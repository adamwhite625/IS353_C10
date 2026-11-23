"""D1 Seed-Ensemble (Mục 10 của kế hoạch chốt): gộp thứ hạng của 5 seed explainer.

Với mỗi (dataset, g, phương pháp ∈ {gnnexplainer, pgexplainer}, node), giải thích gộp có điểm
``-(thứ hạng trung bình qua 5 seed explainer)`` trên cùng tập cạnh ứng viên. Phá hòa bằng hoán vị có seed
với khóa ``{method}:avgrank`` (đúng khóa mà bộ đánh giá dùng cho ``D_between_avg``, Mục 7.3), nên top-k của
bản gộp ở đây trùng với top-k đã dùng trong ``a2_2_between.csv``.

Đầu ra (``--out``, mặc định ``results/analysis/d1``):
  d1_node_metrics.csv   metric của bản gộp theo (g, node, budget)  [kiểm toán]
  d1_metrics.csv        trung bình theo node + CI bootstrap: bản gộp và bản một seed (trung bình 5 seed)
  d1_vs_single_seed.csv hiệu cặp theo node: gộp - {trung bình 5 seed đơn, seed đơn tốt nhất, tệ nhất}
  d1_stability.csv      κ@k giữa hai bản gộp RỜI NHAU (1v1, 2v2, 2v3) so với 1v1 = I_within
  manifest.json         git_commit, git_dirty, argv, sha256 đầu vào/đầu ra
và ``docs/BANG_SO_LIEU_D1_SEED_ENSEMBLE.md``.

Thiết kế ổn định: với 5 seed không thể có hai nhóm 5 rời nhau. Dùng các phép tách rời nhau của {0..4}:
1v1 (10 cặp; chính là I_within), 2v2 (15 phép tách, dùng 4 seed), 2v3 (10 phép tách, dùng cả 5 seed).
Không dùng leave-one-seed-out vì hai bản gộp 4 seed chia sẻ 3 seed, κ bị thổi phồng cơ học.

Ví dụ:
  PYTHONPATH=src python scripts/analyze_ensemble.py \
      --scores-root D:/.../final_scores --eval-root D:/.../final_eval
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from sna.core.provenance import execution_metadata, sha256, write_json  # noqa: E402
from sna.data import extract_candidate_edges  # noqa: E402
from sna.evaluation.core import fidelity_selected, ground_truth_metrics, kappa_at_k, rank_edges, top_indices  # noqa: E402
from sna.evaluation.runner_v2 import (GScores, LocalGraph, OrderBook, config_from_protocol, load_data,  # noqa: E402
                                      load_model, load_nodes, load_score_files, tie_seed,
                                      verify_checkpoint_hash)
from sna.evaluation.stats import bootstrap_ci, compare_many, compare_paired  # noqa: E402

DATASETS = ['bashapes', 'cora']
METHODS = ['gnnexplainer', 'pgexplainer']
METRICS = ['fid_drop', 'fid_inv', 'roc_auc', 'average_precision', 'precision']
LOWER_IS_BETTER = {'fid_inv'}      # Inv gần 0 / nhỏ hơn = các cạnh chọn đủ hơn
BUDGETS = (1, 3, 5, 10)
MAIN_BUDGET = 5
DESIGNS = (('1v1', 1, 1), ('2v2', 2, 2), ('2v3', 2, 3))
SEED = 0
LABEL = {'gnnexplainer': 'GNNExplainer', 'pgexplainer': 'PGExplainer'}
METRIC_LABEL = {'fid_drop': 'Fid-@k Drop', 'fid_inv': 'Fid+@k Inv', 'roc_auc': 'AUROC',
                'average_precision': 'AP', 'precision': 'Precision@k'}


# ---------------------------------------------------------------------------
# Hàm thuần (có test)
# ---------------------------------------------------------------------------

def ranks_from_order(order):
    """Thứ hạng 0 = tốt nhất của từng cạnh, từ thứ tự chọn (order[i] = chỉ số cạnh hạng i)."""
    order = np.asarray(order, dtype=int)
    r = np.empty(len(order))
    r[order] = np.arange(len(order))
    return r


def ensemble_scores(orders):
    """Điểm gộp = -(thứ hạng trung bình) từ danh sách thứ tự (mỗi thứ tự đã phá hòa)."""
    if not len(orders):
        raise ValueError('Cần ít nhất một thứ tự để gộp')
    return -np.mean([ranks_from_order(o) for o in orders], axis=0)


def ensemble_for(book, method, seeds, node):
    """Điểm gộp của ``seeds`` cho (method, node); dùng thứ tự phá hòa của OrderBook."""
    return ensemble_scores([book.get(method, e, node)[1] for e in seeds])


def disjoint_splits(seeds, size_a, size_b):
    """Mọi phép tách (A, B) rời nhau, |A|=size_a, |B|=size_b; mỗi phép tách đúng một lần."""
    seeds = tuple(sorted(seeds))
    out, seen = [], set()
    for a in combinations(seeds, size_a):
        rest = [s for s in seeds if s not in a]
        for b in combinations(rest, size_b):
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
    return out


def split_kappa(scores_a, scores_b, key_a, key_b, budgets, tie_for):
    """κ@k giữa hai bản gộp; ``tie_for(key)`` cho seed phá hòa của từng nhóm."""
    oa = rank_edges(scores_a, tie_for(key_a))
    ob = rank_edges(scores_b, tie_for(key_b))
    m = len(scores_a)
    return {b: kappa_at_k(top_indices(oa, b), top_indices(ob, b), b, m)['kappa'] for b in budgets}


def direction(metric):
    return 'lower_better' if metric in LOWER_IS_BETTER else 'higher_better'


def pick_extreme_seed(per_seed_means, metric, which):
    """Seed đơn 'tốt nhất'/'tệ nhất' theo trung bình toàn dataset (chọn hậu nghiệm). per_seed_means: {e: mean}."""
    items = {e: v for e, v in per_seed_means.items() if not np.isnan(v)}
    if not items:
        return None
    best_is_max = metric not in LOWER_IS_BETTER
    pick_max = best_is_max if which == 'best' else not best_is_max
    return (max if pick_max else min)(items, key=items.get)


# ---------------------------------------------------------------------------
# Một đơn vị công việc: (dataset, g)
# ---------------------------------------------------------------------------

def _nan_row(**base):
    return {**base, **{c: float('nan') for c in METRICS}}


def process_unit(args):
    dataset, g, scores_root, eval_root = args
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    cfg = config_from_protocol(dataset, scores_root=Path(scores_root), out=Path('.'))
    nodes, nodes_sha = load_nodes(cfg.nodes_path)
    data, data_sha = load_data(cfg)
    mask = data.get(f'{cfg.split}_mask')
    if mask is None or not all(bool(mask[n]) for n in nodes):
        raise ValueError(f'{dataset}: có node ngoài split {cfg.split}')
    files, file_errors = load_score_files(cfg, g)
    if file_errors:
        raise ValueError(f'{dataset} g={g}: thiếu CSV điểm: {file_errors}')
    csv_hashes = {f.path.name: f.checkpoint_hashes for f in files.values()}
    ckpt_sha = verify_checkpoint_hash(cfg, g, csv_hashes, nodes_sha)     # dừng nếu lệch hash
    model, ck = load_model(cfg.checkpoint_path(g), cfg.device)
    with torch.no_grad():
        logits = model(data['x'], data['edge_index'])
        pred_all = logits.argmax(-1)
        probs_all = logits.softmax(-1)
    if 'pred' in ck and not torch.equal(pred_all, torch.as_tensor(ck['pred'])):
        raise ValueError(f'g={g}: dự đoán tính lại khác ck["pred"]')
    pred = {n: int(pred_all[n]) for n in nodes}

    def graph_pairs(node):
        cand = extract_candidate_edges(data['edge_index'], node, num_hops=2, num_nodes=len(data['x']))
        return {tuple(p) for p in cand.candidate_pairs}

    gs = GScores(cfg, g, files, nodes, pred, graph_pairs)
    if gs.errors or gs.invalid_nodes:
        raise ValueError(f'{dataset} g={g}: GScores có lỗi: {gs.errors[:3]}')
    book = OrderBook(cfg, gs)
    gt_mask = data.get('gt_edge_mask') if dataset == 'bashapes' else None
    seeds_all = {m: gs.seeds_of(m) for m in METHODS}

    single = pd.read_csv(Path(eval_root) / f'{dataset}_test' / 'metrics.csv')
    single = single[(single.g == g) & (single.kind == 'explainer') & (single.budget == MAIN_BUDGET)
                    & single.method.isin(METHODS)].set_index(['method', 'explainer_seed', 'node_id'])

    metric_rows, stab_rows = [], []
    sanity = dict(n_checked=0, n_tie_seed_mismatch=0, max_abs_diff_precision=0.0, max_abs_diff_auc=0.0,
                  n_fid_checked=0, max_abs_diff_fid_drop=0.0)
    local = {}
    for node in nodes:
        if node not in gs.pairs:
            continue
        pairs, target = gs.pairs[node], pred[node]
        m = len(pairs)
        orig = float(probs_all[node, target])
        for method in METHODS:
            seeds = seeds_all[method]
            base = dict(dataset=dataset, g=g, method=method, node_id=node, n_candidates=m,
                        n_seeds=len(seeds))
            if m == 0:
                for b in BUDGETS:
                    metric_rows.append(_nan_row(**base, budget=b, status='empty_candidates', tie_seed=-1))
                continue
            if len(seeds) != len(cfg.explainer_seeds) or not all(book.has(method, e, node) for e in seeds):
                for b in BUDGETS:
                    metric_rows.append(_nan_row(**base, budget=b, status='incomplete_seeds', tie_seed=-1))
                continue
            # --- bản gộp đủ seed ---
            scores = ensemble_for(book, method, seeds, node)
            tseed = tie_seed(cfg, node, f'{method}:avgrank', 0, g)
            order = rank_edges(scores, tseed)
            if node not in local:
                local[node] = LocalGraph(data, node)
            lg = local[node]
            lpairs = lg.pairs(pairs)
            for b in BUDGETS:
                sel = top_indices(order, b)
                fid = fidelity_selected(model, lg.data, lg.node, target, lpairs, sel, orig)
                gt = ground_truth_metrics(pairs, scores, data['edge_index'], gt_mask, sel)
                metric_rows.append({**base, 'budget': b, 'status': 'success', 'tie_seed': tseed,
                                    'fid_drop': fid['fid_drop'], 'fid_inv': fid['fid_inv'],
                                    'roc_auc': gt['roc_auc'], 'average_precision': gt['average_precision'],
                                    'precision': gt['precision']})
            # --- kiểm tra chéo với bộ đánh giá (budget chính): seed phá hòa, precision, AUROC, fid (e0) ---
            for e in seeds:
                s, o, ts = book.get(method, e, node)
                row = single.loc[(method, e, node)]
                sanity['n_checked'] += 1
                sanity['n_tie_seed_mismatch'] += int(int(row.tie_seed) != int(ts))
                gt = ground_truth_metrics(pairs, s, data['edge_index'], gt_mask, top_indices(o, MAIN_BUDGET))
                for key, col in (('max_abs_diff_precision', 'precision'), ('max_abs_diff_auc', 'roc_auc')):
                    if not np.isnan(gt[col]):
                        sanity[key] = max(sanity[key], abs(gt[col] - float(row[col])))
                if e == seeds[0]:
                    fid = fidelity_selected(model, lg.data, lg.node, target, lpairs,
                                            top_indices(o, MAIN_BUDGET), orig)
                    sanity['n_fid_checked'] += 1
                    sanity['max_abs_diff_fid_drop'] = max(sanity['max_abs_diff_fid_drop'],
                                                          abs(fid['fid_drop'] - float(row.fid_drop)))
            # --- ổn định giữa hai bản gộp rời nhau ---
            for design, sa, sb in DESIGNS:
                splits = disjoint_splits(seeds, sa, sb)
                cache = {}

                def sc(group):
                    if group not in cache:
                        cache[group] = ensemble_for(book, method, group, node)
                    return cache[group]

                def tie_for(group):
                    return tie_seed(cfg, node, f'{method}:ens{"".join(map(str, group))}', 0, g)

                if design == '1v1':       # đúng thứ tự của bộ đánh giá (= I_within)
                    vals = []
                    for (a,), (b_,) in splits:
                        oa, ob = book.get(method, a, node)[1], book.get(method, b_, node)[1]
                        vals.append({bu: kappa_at_k(top_indices(oa, bu), top_indices(ob, bu), bu, m)['kappa']
                                     for bu in BUDGETS})
                else:
                    vals = [split_kappa(sc(a), sc(b_), a, b_, BUDGETS, tie_for) for a, b_ in splits]
                for bu in BUDGETS:
                    arr = np.array([v[bu] for v in vals], dtype=float)
                    stab_rows.append(dict(dataset=dataset, g=g, method=method, design=design, budget=bu,
                                          node_id=node, n_splits=len(vals),
                                          kappa=float(arr.mean()) if not np.isnan(arr).any() else float('nan')))
    prov = dict(data_sha256=data_sha, nodes_sha256=nodes_sha, checkpoint_sha256=ckpt_sha,
                score_csv_sha256={f'{mm}_e{e}': f.sha for (mm, e), f in sorted(files.items())
                                  if mm in METHODS})
    return dict(dataset=dataset, g=g, metrics=metric_rows, stability=stab_rows, sanity=sanity, prov=prov,
                seconds=round(time.perf_counter() - t0, 1))


# ---------------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------------

def node_means(df, value, by=('node_id',)):
    return df.groupby(list(by))[value].mean()


def ci_row(values, seed=SEED):
    c = bootstrap_ci(np.asarray(values, dtype=float), n_boot=2000, seed=seed)
    return dict(mean=c['estimate'], ci_low=c['ci_low'], ci_high=c['ci_high'], n=c['n'])


def load_single(eval_root, dataset):
    s = pd.read_csv(Path(eval_root) / f'{dataset}_test' / 'metrics.csv')
    s = s[(s.kind == 'explainer') & s.method.isin(METHODS) & (s.status == 'success')]
    return s[['dataset', 'g', 'method', 'explainer_seed', 'node_id', 'budget', *METRICS]]


def build_metric_tables(ens, singles):
    """d1_metrics.csv và d1_vs_single_seed.csv. ens/singles: DataFrame theo dataset (đã lọc success)."""
    metrics_rows, vs_rows = [], []
    for dataset in DATASETS:
        e_df = ens[(ens.dataset == dataset) & (ens.status == 'success')]
        s_df = singles[singles.dataset == dataset]
        family = []
        for method in METHODS:
            for budget in BUDGETS:
                for metric in METRICS:
                    e = e_df[(e_df.method == method) & (e_df.budget == budget)]
                    s = s_df[(s_df.method == method) & (s_df.budget == budget)]
                    if e[metric].notna().sum() == 0:
                        continue                                  # cora không có ground truth
                    ens_node = node_means(e, metric)              # trung bình qua g
                    single_node = node_means(s, metric)           # trung bình qua g và e
                    metrics_rows.append(dict(dataset=dataset, method=method, variant='ensemble', budget=budget,
                                             metric=metric, **ci_row(ens_node.values)))
                    metrics_rows.append(dict(dataset=dataset, method=method, variant='single_seed_mean',
                                             budget=budget, metric=metric, **ci_row(single_node.values)))
                    per_seed = {}
                    for ee, part in s.groupby('explainer_seed'):
                        per_seed[int(ee)] = node_means(part, metric).reindex(ens_node.index)
                    refs = {'single_mean': single_node.reindex(ens_node.index)}
                    chosen = {}
                    for which in ('best', 'worst'):
                        pick = pick_extreme_seed({k: float(v.mean()) for k, v in per_seed.items()}, metric, which)
                        chosen[which] = pick
                        refs[f'single_{which}'] = per_seed[pick]
                    for ref_name, ref in refs.items():
                        cmp = compare_paired(ens_node.values, ref.values, seed=SEED)
                        vs_rows.append(dict(dataset=dataset, method=method, budget=budget, metric=metric,
                                            direction=direction(metric), reference=ref_name,
                                            reference_seed=chosen.get(ref_name.replace('single_', ''), ''),
                                            **cmp))
    vs = pd.DataFrame(vs_rows)
    vs['p_holm'] = np.nan
    for (_, _), idx in vs.groupby(['dataset', 'budget']).groups.items():       # Holm trong (dataset, budget)
        from sna.evaluation.stats import holm_adjust
        vs.loc[idx, 'p_holm'] = holm_adjust(vs.loc[idx, 'p_raw'].to_numpy())
    vs['family'] = 'dataset x budget'
    vs['better_for_ensemble'] = np.where(vs.direction == 'lower_better', vs.mean_diff < 0, vs.mean_diff > 0)
    return pd.DataFrame(metrics_rows), vs


def build_stability_table(stab, a2_within):
    rows = []
    for dataset in DATASETS:
        d = stab[stab.dataset == dataset]
        for scope, gsel in (('5gnn', None), ('g1000', 1000)):
            ds = d if gsel is None else d[d.g == gsel]
            for method in METHODS:
                for budget in BUDGETS:
                    part = ds[(ds.method == method) & (ds.budget == budget)]
                    nodes_by = {des: node_means(part[part.design == des], 'kappa') for des, _, _ in DESIGNS}
                    fam_start = len(rows)
                    for des, sa, sb in DESIGNS:
                        series = nodes_by[des]
                        row = dict(dataset=dataset, scope=scope, method=method, budget=budget, design=des,
                                   size_a=sa, size_b=sb, **ci_row(series.values),
                                   n_splits=int(part[part.design == des].n_splits.max()))
                        if des != '1v1':
                            cmp = compare_paired(series.values, nodes_by['1v1'].reindex(series.index).values,
                                                 seed=SEED)
                            row.update({f'vs_1v1_{k}': cmp[k] for k in
                                        ('mean_diff', 'ci_low', 'ci_high', 'p_raw', 'n')})
                        rows.append(row)
    out = pd.DataFrame(rows)
    out['vs_1v1_p_holm'] = np.nan
    for _, idx in out[out.design != '1v1'].groupby(['dataset', 'scope', 'budget']).groups.items():
        from sna.evaluation.stats import holm_adjust
        out.loc[idx, 'vs_1v1_p_holm'] = holm_adjust(out.loc[idx, 'vs_1v1_p_raw'].to_numpy())
    # đối chiếu 1v1 với I_within của a2_1 (budget 5, κ)
    out['a2_1_I_within'] = np.nan
    a = a2_within[(a2_within.metric == 'kappa')]
    for i, r in out[out.design == '1v1'].iterrows():
        col = 'I_within_5gnn' if r.scope == '5gnn' else 'I_within_g1000'
        hit = a[(a.dataset == r.dataset) & (a.method == r.method) & (a.budget == r.budget)]
        if len(hit):
            out.loc[i, 'a2_1_I_within'] = float(hit[col].iloc[0])
    return out


# ---------------------------------------------------------------------------
# Báo cáo Markdown
# ---------------------------------------------------------------------------

def f3(x):
    return 'NaN' if pd.isna(x) else f'{x:.3f}'


def fci(r):
    return f'{f3(r["mean"])} [{f3(r["ci_low"])}; {f3(r["ci_high"])}]'


def sig_mark(p):
    return '' if pd.isna(p) else ('*' if p < 0.05 else '')


def build_markdown(metrics, vs, stab, recap, sanity, argv_note):
    L = []
    L.append('# Bảng số liệu D1 Seed-Ensemble\n')
    L.append('Sinh tự động bởi `scripts/analyze_ensemble.py`; số liệu trong `results/analysis/d1/`. '
             'Giải thích gộp = −(thứ hạng trung bình qua 5 seed explainer), cùng tập cạnh ứng viên, '
             'phá hòa bằng hoán vị có seed (khóa `{method}:avgrank`, trùng `D_between_avg` của A2). '
             'Fidelity tính trên đồ thị con 3-hop như bộ đánh giá mặc định. Trung bình theo node '
             '(trước đó trung bình qua 5 GNN seed); CI 95% bootstrap theo node (B=2000). '
             'Fid- Drop và AP/AUROC/Precision: cao hơn tốt hơn; Fid+ Inv: thấp hơn (gần 0) tốt hơn.\n')
    L.append(f'Kiểm tra chéo với bộ đánh giá: {sanity["n_checked"]} cặp (g, phương pháp, seed, node) ở k=5, '
             f'{sanity["n_tie_seed_mismatch"]} lệch seed phá hòa; sai khác lớn nhất precision '
             f'{sanity["max_abs_diff_precision"]:.2e}, AUROC {sanity["max_abs_diff_auc"]:.2e}; '
             f'Fid-@5 (seed e0, {sanity["n_fid_checked"]} cặp) {sanity["max_abs_diff_fid_drop"]:.2e}.\n')

    L.append('## 1. Metric tại k = 5: bản gộp so với một seed (trung bình 5 seed)\n')
    for dataset in DATASETS:
        L.append(f'### {dataset}\n')
        L.append('| Phương pháp | Metric | Gộp (D1) | Một seed (TB 5 seed) | Gộp − TB | CI 95% hiệu | p_Holm | '
                 'Gộp − tốt nhất | Gộp − tệ nhất |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for method in METHODS:
            for metric in METRICS:
                sub = vs[(vs.dataset == dataset) & (vs.method == method) & (vs.budget == MAIN_BUDGET)
                         & (vs.metric == metric)].set_index('reference')
                if sub.empty:
                    continue
                m = metrics[(metrics.dataset == dataset) & (metrics.method == method)
                            & (metrics.budget == MAIN_BUDGET) & (metrics.metric == metric)].set_index('variant')
                a, b, c, w = sub.loc['single_mean'], sub.loc['single_best'], None, sub.loc['single_worst']
                L.append(f'| {LABEL[method]} | {METRIC_LABEL[metric]} | {fci(m.loc["ensemble"])} | '
                         f'{fci(m.loc["single_seed_mean"])} | {a.mean_diff:+.3f} | '
                         f'[{a.ci_low:+.3f}; {a.ci_high:+.3f}] | {f3(a.p_holm)}{sig_mark(a.p_holm)} | '
                         f'{b.mean_diff:+.3f} (p_Holm {f3(b.p_holm)}) | {w.mean_diff:+.3f} (p_Holm {f3(w.p_holm)}) |')
        L.append('')
    L.append('`*` = p_Holm < 0,05 (Holm trong từng dataset × budget). "Tốt nhất/tệ nhất" chọn hậu nghiệm theo '
             'trung bình toàn dataset nên so với "tốt nhất" thiên về bất lợi cho D1.\n')

    L.append('## 2. Các budget khác (Gộp − TB một seed)\n')
    L.append('| Dataset | Phương pháp | Metric | k=1 | k=3 | k=5 | k=10 |')
    L.append('|---|---|---|---|---|---|---|')
    for dataset in DATASETS:
        for method in METHODS:
            for metric in METRICS:
                cells = []
                for b in BUDGETS:
                    r = vs[(vs.dataset == dataset) & (vs.method == method) & (vs.budget == b)
                           & (vs.metric == metric) & (vs.reference == 'single_mean')]
                    cells.append('-' if r.empty else f'{r.mean_diff.iloc[0]:+.3f}{sig_mark(r.p_holm.iloc[0])}')
                if any(c != '-' for c in cells):
                    L.append(f'| {dataset} | {LABEL[method]} | {METRIC_LABEL[metric]} | ' + ' | '.join(cells) + ' |')
    L.append('')

    L.append('## 3. Độ ổn định: κ@5 giữa hai bản gộp rời nhau (trung bình 5 GNN)\n')
    L.append('| Dataset | Phương pháp | 1v1 (= I_within) | I_within (a2_1) | 2v2 | 2v3 | 2v2 − 1v1 (p_Holm) | '
             '2v3 − 1v1 (p_Holm) |')
    L.append('|---|---|---|---|---|---|---|---|')
    s5 = stab[(stab.scope == '5gnn') & (stab.budget == MAIN_BUDGET)]
    for dataset in DATASETS:
        for method in METHODS:
            r = s5[(s5.dataset == dataset) & (s5.method == method)].set_index('design')
            L.append(f'| {dataset} | {LABEL[method]} | {fci(r.loc["1v1"])} | {f3(r.loc["1v1"].a2_1_I_within)} | '
                     f'{fci(r.loc["2v2"])} | {fci(r.loc["2v3"])} | '
                     f'{r.loc["2v2"].vs_1v1_mean_diff:+.3f} ({f3(r.loc["2v2"].vs_1v1_p_holm)}) | '
                     f'{r.loc["2v3"].vs_1v1_mean_diff:+.3f} ({f3(r.loc["2v3"].vs_1v1_p_holm)}) |')
    L.append('')
    L.append('Chỉ có 5 seed nên không có hai bản gộp 5 seed rời nhau: 2v2 dùng 4 seed, 2v3 dùng đủ 5 seed '
             '(nhưng hai nhóm khác kích cỡ). Ablation S=5 chưa chạy (cần seed 5–9). '
             'Bảng đầy đủ các k và g=1000: `d1_stability.csv`.\n')

    L.append('## 4. Bằng chứng bổ sung D_between_avg (A2, κ@5, từ `a2_2_between.csv`)\n')
    L.append('| Dataset | Cặp phương pháp | D_between (từng seed) | D_between_avg (sau gộp seed) |')
    L.append('|---|---|---|---|')
    for _, r in recap.iterrows():
        L.append(f'| {r.dataset} | {r.pair} | {f3(r.D_between)} | {f3(r.D_between_avg)} |')
    L.append('')

    L.append('## 5. Diễn giải\n')
    L.extend(interpretation(vs, stab, recap))
    L.append('')
    L.append(argv_note)
    return '\n'.join(L) + '\n'


def interpretation(vs, stab, recap):
    main = vs[(vs.budget == MAIN_BUDGET) & (vs.reference == 'single_mean')]
    out = []
    for dataset in DATASETS:
        d = main[main.dataset == dataset]
        pos = d[(d.better_for_ensemble) & (d.p_holm < 0.05)]
        neg = d[(~d.better_for_ensemble) & (d.p_holm < 0.05) & (d.mean_diff != 0)]
        none = d[~(d.p_holm < 0.05)]
        out.append(f'- {dataset}, k=5, so với trung bình một seed: {len(pos)}/{len(d)} so sánh (phương pháp × metric) '
                   f'có D1 tốt hơn có ý nghĩa (Holm<0,05), {len(neg)} kém hơn có ý nghĩa, {len(none)} không phân '
                   f'biệt được.' + (' Tốt hơn: ' + ', '.join(f'{LABEL[r.method]} {METRIC_LABEL[r.metric]} '
                                                           f'({r.mean_diff:+.3f})' for _, r in pos.iterrows())
                                   if len(pos) else '')
                   + (' Kém hơn: ' + ', '.join(f'{LABEL[r.method]} {METRIC_LABEL[r.metric]} ({r.mean_diff:+.3f})'
                                              for _, r in neg.iterrows()) if len(neg) else ''))
    sig = main[(main.p_holm < 0.05)]
    if len(sig):
        out.append(f'- Độ lớn: mọi hiệu có ý nghĩa ở k=5 đều nhỏ (|Gộp − TB| lớn nhất = {sig.mean_diff.abs().max():.3f} '
                   f'trên thang metric 0–1); ý nghĩa thống kê không đồng nghĩa lợi ích thực tế đáng kể. '
                   f'Các hiệu còn lại đều trong khoảng ±{main[~(main.p_holm < 0.05)].mean_diff.abs().max():.3f}.')
    best = vs[(vs.budget == MAIN_BUDGET) & (vs.reference == 'single_best')]
    n_beat = int(((best.better_for_ensemble) & (best.p_holm < 0.05)).sum())
    n_lose = int(((~best.better_for_ensemble) & (best.p_holm < 0.05) & (best.mean_diff != 0)).sum())
    out.append(f'- So với seed đơn tốt nhất (chọn hậu nghiệm, k=5): D1 thắng có ý nghĩa {n_beat}/{len(best)}, '
               f'thua có ý nghĩa {n_lose}/{len(best)}; phần còn lại không phân biệt được.')
    s5 = stab[(stab.scope == '5gnn') & (stab.budget == MAIN_BUDGET)]
    for dataset in DATASETS:
        parts = []
        for method in METHODS:
            r = s5[(s5.dataset == dataset) & (s5.method == method)].set_index('design')
            parts.append(f'{LABEL[method]} κ@5 1v1 {r.loc["1v1"]["mean"]:.3f} → 2v2 {r.loc["2v2"]["mean"]:.3f} → '
                         f'2v3 {r.loc["2v3"]["mean"]:.3f}')
        out.append(f'- Độ ổn định ({dataset}): ' + '; '.join(parts) + '.')
    out.append('- κ tăng theo kích thước nhóm gộp là kết quả cơ học của việc lấy trung bình (giảm phương sai seed), '
               'nên chỉ chứng tỏ bản gộp ổn định hơn một seed, KHÔNG chứng tỏ giải thích đúng hơn: các metric ở '
               'bảng 1 (cùng k) gần như không đổi.')
    rc = recap[recap.pair == 'gnnexplainer vs pgexplainer']
    out.append('- D_between_avg so với D_between (GNNExplainer vs PGExplainer): ' +
               '; '.join(f'{r.dataset} {r.D_between:.3f} → {r.D_between_avg:.3f}' for _, r in rc.iterrows()) +
               ' (chưa có CI theo node cho D_between_avg trong A2). Gộp seed không làm hai phương pháp xích lại gần nhau; '
               'bất đồng giữa GNNExplainer và PGExplainer không phải do nhiễu seed.')
    out.append('- Giới hạn: chỉ 5 seed explainer (S=5 không có phần ablation); Saliency tất định nên không đưa vào D1; '
               'chọn seed "tốt nhất/tệ nhất" là hậu nghiệm; phân tích theo node coi các node là độc lập '
               'dù chúng dùng chung GNN; Cora chỉ có metric fidelity (không ground truth).')
    return out


# ---------------------------------------------------------------------------
# Điều phối
# ---------------------------------------------------------------------------

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scores-root', type=Path, required=True)
    p.add_argument('--eval-root', type=Path, required=True,
                   help='thư mục chứa {ds}_test/metrics.csv của bộ đánh giá chính')
    p.add_argument('--a2-dir', type=Path, default=ROOT / 'results' / 'analysis' / 'a2')
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'analysis' / 'd1')
    p.add_argument('--doc', type=Path, default=ROOT / 'docs' / 'BANG_SO_LIEU_D1_SEED_ENSEMBLE.md')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    gseeds = list(config_from_protocol('bashapes').gnn_seeds)
    units = [(ds, g, str(a.scores_root), str(a.eval_root)) for ds in a.datasets for g in gseeds]
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        results = list(ex.map(process_unit, units))
    for r in results:
        print(f"{r['dataset']} g={r['g']}: {r['seconds']}s sanity={r['sanity']}", flush=True)

    ens = pd.DataFrame([row for r in results for row in r['metrics']])
    stab = pd.DataFrame([row for r in results for row in r['stability']])
    singles = pd.concat([load_single(a.eval_root, ds) for ds in a.datasets], ignore_index=True)
    sanity = dict(n_checked=sum(r['sanity']['n_checked'] for r in results),
                  n_tie_seed_mismatch=sum(r['sanity']['n_tie_seed_mismatch'] for r in results),
                  n_fid_checked=sum(r['sanity']['n_fid_checked'] for r in results),
                  **{k: max(r['sanity'][k] for r in results) for k in
                     ('max_abs_diff_precision', 'max_abs_diff_auc', 'max_abs_diff_fid_drop')})
    if sanity['n_tie_seed_mismatch'] or max(sanity['max_abs_diff_precision'], sanity['max_abs_diff_auc']) > 1e-9 \
            or sanity['max_abs_diff_fid_drop'] > 1e-5:
        print(f'CẢNH BÁO kiểm tra chéo với bộ đánh giá không khớp: {sanity}', file=sys.stderr)

    metrics_tbl, vs_tbl = build_metric_tables(ens, singles)
    a2_within = pd.read_csv(a.a2_dir / 'a2_1_within_cross_gnn.csv')
    stab_tbl = build_stability_table(stab, a2_within)
    a2_between = pd.read_csv(a.a2_dir / 'a2_2_between.csv')
    recap = (a2_between[(a2_between.budget == MAIN_BUDGET) & (a2_between.metric == 'kappa')
                        & a2_between.dataset.isin(a.datasets)]
             .drop_duplicates(['dataset', 'pair'])[['dataset', 'pair', 'D_between', 'D_between_avg', 'n']])

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    files = {'d1_node_metrics.csv': ens, 'd1_metrics.csv': metrics_tbl, 'd1_vs_single_seed.csv': vs_tbl,
             'd1_stability.csv': stab_tbl}
    for name, df in files.items():
        df.to_csv(out / name, index=False)
    md = build_markdown(metrics_tbl, vs_tbl, stab_tbl, recap, sanity,
                        'Lệnh: `PYTHONPATH=src python scripts/analyze_ensemble.py --scores-root <final_scores> '
                        '--eval-root <final_eval> --workers 4`.')
    a.doc.parent.mkdir(parents=True, exist_ok=True)
    a.doc.write_text(md, encoding='utf-8')
    inputs = {}
    for r in results:
        key = f"{r['dataset']}/g{r['g']}"
        inputs[key] = r['prov']
    for ds in a.datasets:
        inputs[f'eval/{ds}_test/metrics.csv'] = sha256(a.eval_root / f'{ds}_test' / 'metrics.csv')
    for name in ('a2_1_within_cross_gnn.csv', 'a2_2_between.csv'):
        inputs[f'a2/{name}'] = sha256(a.a2_dir / name)
    outputs = {n: sha256(out / n) for n in files}
    outputs[str(a.doc.name)] = sha256(a.doc)
    write_json(out / 'manifest.json', {
        **execution_metadata(), 'schema': 'd1_seed_ensemble', 'script': 'scripts/analyze_ensemble.py',
        'methods': METHODS, 'budgets': list(BUDGETS), 'main_budget': MAIN_BUDGET, 'stability_designs': [d[0] for d in DESIGNS],
        'bootstrap': {'unit': 'node', 'repeats': 2000, 'seed': SEED}, 'holm_family': 'dataset x budget',
        'local_fidelity': True, 'tie_key': '{method}:avgrank', 'sanity_vs_evaluator': sanity,
        'n_incomplete_or_empty_node_rows': int((ens.status != 'success').sum()),
        'inputs_sha256': inputs, 'outputs_sha256': outputs,
        'seconds': round(time.perf_counter() - t0, 1)})
    print(f'Xong: {out} ({round(time.perf_counter() - t0, 1)}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
