"""FB-2 analysis: Table A2-1, A2-2, A2-3 and Figure A2 from an evaluate.py output.

Example:
  python scripts/analyze_variance.py --eval results/eval/bashapes_test
Writes results/analysis/a2_{dataset}/ (tables .csv/.md, fig_a2.png, report.md, manifest.json).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sna.core.config import PROJECT_ROOT
from sna.core.provenance import execution_metadata, protocol, sha256, write_json
from sna.evaluation import stats
from sna.evaluation.variance import (between, cross_gnn, methods_in, prediction_changes,
                                     ref_random, table_a2_1, table_a2_2, table_a2_3, verdicts,
                                     within)

LABEL = {'kappa': 'κ', 'jaccard': 'Jaccard', 'kendall_tau_b': 'τ_b'}


def fmt(row, p=False):
    if row is None or not np.isfinite(row['mean']):
        return 'NaN'
    text = f"{row['mean']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
    if p and 'p_holm' in row and np.isfinite(row.get('p_holm', np.nan)):
        text += f" p={row['p_holm']:.3g}"
    return text


def pick(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    return sel.iloc[0] if len(sel) else None


def md_a2_1(t, metric, budget, scopes, changes):
    t = t[(t.metric == metric) & (t.budget == budget)]
    sym = LABEL[metric]
    lines = [f'### Bảng A2-1 — {sym}@{budget}', '',
             f'| Phương pháp | I_within (GNN cố định) | I_within (5 GNN) | I_gnn | I_within − I_gnn (Holm) | {sym} với Nearest-First | n node |',
             '|---|---|---|---|---|---|---:|']
    for method in [m for m in t.method.unique() if m != 'random']:
        f = pick(t, scope='fixed', method=method, quantity='I_within') if 'fixed' in scopes else None
        a = pick(t, scope='all', method=method, quantity='I_within') if 'all' in scopes else None
        g = pick(t, scope='all', method=method, quantity='I_gnn') if 'all' in scopes else None
        d = pick(t, scope='all', method=method, quantity='delta_within_minus_gnn') if 'all' in scopes else None
        ref_scope = 'all' if 'all' in scopes else 'fixed'
        r = pick(t, scope=ref_scope, method=method, quantity='ref_nearest')
        n = max(int(x['n']) for x in (f, a) if x is not None)
        lines.append(f'| {method} | {fmt(f)} | {fmt(a)} | {fmt(g)} | {fmt(d, p=True)} | {fmt(r)} | {n} |')
    rnd = pick(t, method='random', quantity='ref_random')
    lines += ['', f'Mốc ngẫu nhiên Random–Random: {fmt(rnd)}.']
    if changes and len(changes['gnn_seeds']) > 1:
        lines.append(f"Tỷ lệ đổi dự đoán giữa các seed GNN: {changes['pair_change_rate']:.1%} số cặp (g, g'); "
                     f"{changes['node_change_rate']:.1%} số node đổi lớp ở ít nhất một seed "
                     f"({changes['n_nodes']} node). I_gnn chỉ tính trên cặp cùng dự đoán.")
    return lines


def md_a2_2(t, metric, budget, scope):
    t = t[(t.metric == metric) & (t.budget == budget) & (t.scope == scope)]
    sym = LABEL[metric]
    title = 'GNN cố định' if scope == 'fixed' else '5 GNN'
    lines = [f'### Bảng A2-2 — {sym}@{budget}, {title}', '',
             '| Cặp | D_between | D_between_avg | I_within A | I_within B | Δ_A = I_within A − D | Δ_B = I_within B − D | Δ_min | D_avg − D |',
             '|---|---|---|---|---|---|---|---|---|']
    for (a, b), part in t.groupby(['method_a', 'method_b'], sort=False):
        q = lambda name: pick(part, quantity=name)
        lines.append(f"| {a} – {b} | {fmt(q('D_between'))} | {fmt(q('D_between_avg'))} | "
                     f"{fmt(q('I_within_a'))} | {fmt(q('I_within_b'))} | {fmt(q('delta_a'), True)} | "
                     f"{fmt(q('delta_b'), True)} | {fmt(q('delta_min'), True)} | {fmt(q('delta_avg'), True)} |")
    lines += ['', 'Giá trị: trung bình trên node [CI 95% bootstrap]; p là Wilcoxon theo cặp node, hiệu chỉnh Holm trong bảng.']
    return lines


def figure(ag, metric, budget, scope, gnn_seeds, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ag = ag[ag.budget == budget]
    methods = methods_in(ag)
    bars = {'within': [], 'cross-GNN': [], 'between': []}
    for m in methods:
        bars['within'].append(stats.summarize(within(ag, m, metric, gnn_seeds).to_numpy()))
        cg = cross_gnn(ag, m, metric) if scope == 'all' else pd.Series(dtype=float)
        bars['cross-GNN'].append(stats.summarize(cg.to_numpy()))
        others = [between(ag, m, o, metric, gnn_seeds) for o in methods if o != m]
        per_node = pd.concat(others, axis=1).mean(axis=1) if others else pd.Series(dtype=float)
        bars['between'].append(stats.summarize(per_node.to_numpy()))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    width = 0.26
    for i, (name, vals) in enumerate(bars.items()):
        means = np.array([v['mean'] for v in vals])
        err = np.array([[v['mean'] - v['ci_lo'], v['ci_hi'] - v['mean']] for v in vals]).T
        ax.bar(x + (i - 1) * width, np.nan_to_num(means), width, yerr=np.nan_to_num(err),
               capsize=3, label=name)
    rnd = stats.summarize(ref_random(ag, metric).to_numpy())['mean']
    if np.isfinite(rnd):
        ax.axhline(rnd, color='grey', ls='--', lw=1, label='Random–Random')
    ax.set_xticks(x, methods)
    ax.set_ylabel(f'{LABEL[metric]}@{budget}')
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    spec = protocol()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, required=True, help='evaluate.py output directory')
    parser.add_argument('--out', type=Path, default=None,
                        help='default: results/analysis/a2_{dataset}')
    parser.add_argument('--metrics', nargs='+', default=list(LABEL), choices=list(LABEL))
    parser.add_argument('--budgets', type=int, nargs='+', default=spec['budgets'])
    parser.add_argument('--main-budget', type=int, default=spec['main_budget'])
    parser.add_argument('--fixed-gnn-seed', type=int, default=spec['fixed_gnn_seed'])
    args = parser.parse_args(argv)

    manifest_in = json.loads((args.eval / 'manifest.json').read_text(encoding='utf-8'))
    dataset = manifest_in['dataset']
    out = args.out or PROJECT_ROOT / 'results' / 'analysis' / f'a2_{dataset}'
    out.mkdir(parents=True, exist_ok=True)
    ag = pd.read_csv(args.eval / 'agreement.csv')
    metrics = pd.read_csv(args.eval / 'metrics.csv')
    gnn_seeds = sorted(int(g) for g in metrics.gnn_seed.unique())
    scopes = {}
    if args.fixed_gnn_seed in gnn_seeds:
        scopes['fixed'] = [args.fixed_gnn_seed]
    if len(gnn_seeds) > 1:
        scopes['all'] = None
    if not scopes:
        raise SystemExit(f'Fixed GNN seed {args.fixed_gnn_seed} absent and only one GNN seed present')

    a1, a2, a3 = [], [], []
    for scope, seeds in scopes.items():
        for metric in args.metrics:
            for k in args.budgets:
                a1 += table_a2_1(ag, dataset, metric, k, scope, seeds)
                a2 += table_a2_2(ag, dataset, metric, k, scope, seeds)
        a3 += [{**r, 'scope': scope} for r in table_a2_3(metrics, dataset, args.main_budget, seeds)]
    t1, t2, t3 = pd.DataFrame(a1), pd.DataFrame(a2), pd.DataFrame(a3)
    changes = prediction_changes(metrics)
    t1.to_csv(out / 'a2_1.csv', index=False)
    t2.to_csv(out / 'a2_2.csv', index=False)
    t3.to_csv(out / 'a2_3.csv', index=False)

    k = args.main_budget
    fig_scope = 'all' if 'all' in scopes else 'fixed'
    figure(ag, 'kappa', k, fig_scope, scopes[fig_scope], out / 'fig_a2.png',
           f'{dataset}: within / cross-GNN / between (κ@{k}, {fig_scope})')

    lines = [f'# FB-2 — {dataset}', '',
             f"Nguồn: `{args.eval.as_posix()}` (manifest status `{manifest_in['status']}`, "
             f"{manifest_in['n_nodes']} node, GNN seed {gnn_seeds}). Sinh bằng `scripts/analyze_variance.py`; không sửa tay.", '']
    if manifest_in.get('missing_protocol_runs'):
        lines += [f"**Cảnh báo:** thiếu {len(manifest_in['missing_protocol_runs'])} run so với giao thức, "
                  'bảng chưa phải bản cuối.', '']
    for metric in args.metrics:
        lines += md_a2_1(t1, metric, k, scopes, changes) + ['']
        for scope in scopes:
            lines += md_a2_2(t2, metric, k, scope) + ['']
    if not t3.empty:
        lines += [f'### Bảng A2-3 — Fid-@{k}: độ lệch theo seed explainer so với chênh lệch giữa phương pháp', '',
                  '| Phạm vi | Phương pháp / cặp | Đại lượng | Trung bình [CI] | n |', '|---|---|---|---|---:|']
        for r in t3.to_dict('records'):
            lines.append(f"| {r['scope']} | {r['method']} | {r['quantity']} | {fmt(r)} | {r['n']} |")
        lines.append('')
    rules = verdicts(a1, a2)
    lines += ['## Đọc theo quy tắc Mục 7.4 (tự động, cần người kiểm lại)', '']
    lines += [f'- {v}' for v in rules if f'@{k}]' in v] or ['- Không có quy tắc nào được kích hoạt.']
    lines += ['', '![Hình A2](fig_a2.png)', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')

    outputs = ['a2_1.csv', 'a2_2.csv', 'a2_3.csv', 'fig_a2.png', 'report.md']
    write_json(out / 'manifest.json', {
        **execution_metadata(), 'mode': 'analyze_variance', 'dataset': dataset,
        'eval_dir': str(args.eval.as_posix()),
        'inputs_sha256': {name: sha256(args.eval / name)
                          for name in ('manifest.json', 'agreement.csv', 'metrics.csv')},
        'eval_git_commit': manifest_in.get('git_commit'), 'scopes': list(scopes),
        'fixed_gnn_seed': args.fixed_gnn_seed, 'gnn_seeds': gnn_seeds,
        'metrics': args.metrics, 'budgets': args.budgets, 'main_budget': k,
        'prediction_changes': changes, 'verdicts': rules,
        'outputs_sha256': {name: sha256(out / name) for name in outputs}})
    print(f'A2 written to {out}')
    for v in rules:
        if f'@{k}]' in v and v.startswith('[') and 'kappa' in v:
            print('  ' + v)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
