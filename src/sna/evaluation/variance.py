"""FB-2: separate explainer instability, GNN sensitivity and method disagreement.

Reads the evaluator v2 outputs (agreement.csv, metrics.csv) and builds
Table A2-1, Table A2-2, Table A2-3 and Figure A2 (KE_HOACH_CUOI_KY_FINAL.md, §7).

Every quantity is first averaged within a node, then summarised over nodes
(mean, SD ddof=1, bootstrap CI). Paired differences use the same nodes on both
sides; Holm is applied within one table (one scope × metric × budget).

Scopes:
- ``fixed``: only the fixed GNN (step 1 of the teacher's order);
- ``all``  : all GNN seeds (step 2); cross-GNN terms exist only here.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from . import stats

EXPLAINERS = ('saliency', 'gnnexplainer', 'pgexplainer')
AGREEMENT_METRICS = ('kappa', 'jaccard', 'kendall_tau_b')


def _node_mean(rows, metric):
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby('node_id')[metric].mean()


def _scope(rows, gnn_seeds):
    if gnn_seeds is None:
        return rows
    return rows[rows.gnn_seed_a.isin(gnn_seeds) & rows.gnn_seed_b.isin(gnn_seeds)]


def within(ag, method, metric, gnn_seeds=None):
    rows = ag[(ag.kind == 'within') & (ag.method_a == method)]
    return _node_mean(_scope(rows, gnn_seeds), metric)


def cross_gnn(ag, method, metric, ensemble=False):
    kind = 'cross_gnn_ensemble' if ensemble else 'cross_gnn'
    rows = ag[(ag.kind == kind) & (ag.method_a == method) & ag.same_pred.astype(bool)]
    return _node_mean(rows, metric)


def between(ag, a, b, metric, gnn_seeds=None, ensemble=False):
    kind = 'between_ensemble' if ensemble else 'between'
    rows = ag[(ag.kind == kind) & (((ag.method_a == a) & (ag.method_b == b)) |
                                   ((ag.method_a == b) & (ag.method_b == a)))]
    return _node_mean(_scope(rows, gnn_seeds), metric)


def ref_nearest(ag, method, metric, gnn_seeds=None):
    rows = ag[(ag.kind == 'ref_nearest') & (ag.method_a == method) & (ag.variant_a == 'single')]
    return _node_mean(_scope(rows, gnn_seeds), metric)


def ref_random(ag, metric):
    return _node_mean(ag[ag.kind == 'ref_random'], metric)


def _aligned(a, b):
    both = pd.concat([a.rename('a'), b.rename('b')], axis=1)
    return both['a'].to_numpy(float), both['b'].to_numpy(float)


def _row(base, quantity, values, **extra):
    return {**base, 'quantity': quantity, **stats.summarize(np.asarray(values, float)), **extra}


def _paired_row(base, quantity, a, b):
    x, y = _aligned(a, b)
    r = stats.paired_compare(x, y)
    s = stats.summarize(x - y)
    return {**base, 'quantity': quantity, 'mean': r['delta'], 'sd': s['sd'], 'n': r['n'],
            'n_nan': s['n_nan'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'], 'p': r['p'],
            'test_reason': r['reason']}


def _holm(rows):
    idx = [i for i, r in enumerate(rows) if 'p' in r]
    adjusted = stats.holm([rows[i]['p'] for i in idx])
    for i, p in zip(idx, adjusted):
        rows[i]['p_holm'] = float(p)
    return rows


def prediction_changes(metrics):
    """Share of GNN-seed pairs (and of nodes) whose predicted class differs."""
    preds = metrics[['gnn_seed', 'node_id', 'target_class']].drop_duplicates()
    if preds.duplicated(['gnn_seed', 'node_id']).any():
        raise ValueError('metrics.csv has several target classes for one (gnn_seed, node)')
    wide = preds.pivot(index='node_id', columns='gnn_seed', values='target_class')
    gs = list(wide.columns)
    pair_changes = [(wide[a] != wide[b]).to_numpy() for a, b in combinations(gs, 2)]
    pair_rate = float(np.mean(pair_changes)) if pair_changes else float('nan')
    node_rate = float((wide.nunique(axis=1) > 1).mean()) if len(gs) > 1 else float('nan')
    return {'gnn_seeds': [int(g) for g in gs], 'n_nodes': int(len(wide)),
            'pair_change_rate': pair_rate, 'node_change_rate': node_rate}


def methods_in(ag):
    present = set(ag.method_a) | set(ag.method_b)
    return [m for m in EXPLAINERS if m in present]


def table_a2_1(ag, dataset, metric, budget, scope, gnn_seeds):
    ag = ag[ag.budget == budget]
    base = dict(dataset=dataset, scope=scope, metric=metric, budget=budget)
    rows = []
    for method in methods_in(ag):
        b = {**base, 'method': method}
        w = within(ag, method, metric, gnn_seeds)
        rows.append(_row(b, 'I_within', w))
        rows.append(_row(b, 'ref_nearest', ref_nearest(ag, method, metric, gnn_seeds)))
        if scope == 'all':
            g = cross_gnn(ag, method, metric)
            rows.append(_row(b, 'I_gnn', g))
            rows.append(_row(b, 'I_gnn_ensemble', cross_gnn(ag, method, metric, ensemble=True)))
            if len(w) and len(g):
                rows.append(_paired_row(b, 'delta_within_minus_gnn', w, g))
    rows.append(_row({**base, 'method': 'random'}, 'ref_random', ref_random(ag, metric)))
    return _holm(rows)


def table_a2_2(ag, dataset, metric, budget, scope, gnn_seeds):
    ag = ag[ag.budget == budget]
    base = dict(dataset=dataset, scope=scope, metric=metric, budget=budget)
    rows = []
    for a, b in combinations(methods_in(ag), 2):
        p = {**base, 'method_a': a, 'method_b': b}
        d = between(ag, a, b, metric, gnn_seeds)
        d_avg = between(ag, a, b, metric, gnn_seeds, ensemble=True)
        wa, wb = within(ag, a, metric, gnn_seeds), within(ag, b, metric, gnn_seeds)
        rows.append(_row(p, 'D_between', d))
        rows.append(_row(p, 'D_between_avg', d_avg))
        rows.append(_row(p, 'I_within_a', wa))
        rows.append(_row(p, 'I_within_b', wb))
        if len(d):
            if len(wa):
                rows.append(_paired_row(p, 'delta_a', wa, d))
            if len(wb):
                rows.append(_paired_row(p, 'delta_b', wb, d))
            if len(wa) and len(wb):
                # Per node min(I_within(A), I_within(B)); NaN if either side is undefined.
                x, y = _aligned(wa, wb)
                lowest = pd.Series(np.minimum(x, y), index=pd.concat([wa, wb], axis=1).index)
                rows.append(_paired_row(p, 'delta_min', lowest, d))
            if len(d_avg):
                rows.append(_paired_row(p, 'delta_avg', d_avg, d))
    return _holm(rows)


def table_a2_3(metrics, dataset, budget, gnn_seeds=None):
    """Fid-@k spread across explainer seeds vs spread between methods (per node)."""
    m = metrics[(metrics.status == 'success') & (metrics.variant == 'single') &
                (metrics.tie_index == 0) & (metrics.budget == budget)]
    if gnn_seeds is not None:
        m = m[m.gnn_seed.isin(gnn_seeds)]
    base = dict(dataset=dataset, metric='fid_drop', budget=budget)
    rows = []
    per = m.groupby(['method', 'gnn_seed', 'node_id']).fid_drop
    sd = per.std(ddof=1).groupby(['method', 'node_id']).mean()
    mean = per.mean().groupby(['method', 'node_id']).mean()
    for method in [x for x in EXPLAINERS if x in set(m.method)]:
        if method in sd.index.get_level_values(0):
            rows.append(_row({**base, 'method': method}, 'sd_across_explainer_seeds',
                             sd.loc[method].to_numpy()))
    for a, b in combinations([x for x in EXPLAINERS if x in set(m.method)], 2):
        diff = (mean.loc[a] - mean.loc[b]).abs()
        rows.append(_row({**base, 'method': f'{a}|{b}'}, 'abs_diff_between_methods', diff.to_numpy()))
    return rows


def verdicts(a2_1, a2_2):
    """Pre-registered reading rules of §7.4 applied mechanically (Vietnamese text)."""
    out = []
    t2 = pd.DataFrame(a2_2)
    t1 = pd.DataFrame(a2_1)

    def get(df, **kw):
        sel = df
        for k, v in kw.items():
            sel = sel[sel[k] == v]
        return sel.iloc[0] if len(sel) else None

    for (scope, metric, budget, a, b), part in (t2.groupby(['scope', 'metric', 'budget',
                                                            'method_a', 'method_b'])
                                                if not t2.empty else []):
        da, db = get(part, quantity='delta_a'), get(part, quantity='delta_b')
        avg = get(part, quantity='delta_avg')
        head = f'[{scope}, {metric}@{budget}] {a} – {b}: '
        if da is not None and db is not None:
            if da.ci_lo > 0 and db.ci_lo > 0:
                out.append(head + 'bất đồng thực sự (CI của Δ_A và Δ_B đều > 0).')
            else:
                weak = [m for m, r in ((a, da), (b, db)) if not r.ci_lo > 0]
                out.append(head + 'chưa đủ bằng chứng bất đồng vượt mức tự lệch của '
                           + ', '.join(weak) + ' (CI của Δ chứa 0 hoặc âm).')
        if avg is not None and avg.ci_lo > 0:
            out.append(head + 'D_between_avg > D_between: một phần bất đồng là nhiễu seed, gộp seed (D1) giúp ích.')
    if not t1.empty:
        for (metric, budget, method), part in t1[t1.scope == 'all'].groupby(['metric', 'budget', 'method']):
            d = get(part, quantity='delta_within_minus_gnn')
            w = get(part, quantity='I_within')
            g = get(part, quantity='I_gnn')
            head = f'[all, {metric}@{budget}] {method}: '
            if d is not None and d.ci_lo > 0:
                out.append(head + 'I_gnn thấp hơn rõ I_within: giải thích nhạy với việc huấn luyện lại GNN.')
            if (method == 'saliency' and w is not None and g is not None
                    and np.isclose(w['mean'], 1.0) and d is not None and d.ci_lo > 0):
                out.append(head + 'tất định theo seed explainer (I_within = 1) nhưng vẫn phụ thuộc trọng số GNN.')
    return out
