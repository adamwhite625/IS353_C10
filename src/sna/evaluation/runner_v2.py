"""Final-protocol evaluator (F3): many (GNN seed, explainer seed) runs in one pass.

Input  : results/scores/{ds}/gcn_s{g}/{method}_e{e}.csv (schema v2, run_explainers.py)
         results/checkpoints/{ds}/gcn_s{g}.pt (+ .json), data/{ds}/test_nodes.json
Output : metrics.csv, agreement.csv, summary.csv, summary_by_config.csv,
         agreement_summary.csv, errors.csv, manifest.json

Provenance is a hard gate: the evaluator stops (HashMismatchError) when a CSV
names a checkpoint whose bytes differ, when the checkpoint metadata disagrees,
or when data / node list / CSV hashes differ from what the run logs recorded.
Metric conventions are those of ``evaluation/core.py``; nothing is recomputed
differently here. Fidelity is evaluated on the exact 3-hop subgraph of the
target (equal to the full graph for a 2-layer GCN; checked per node).
"""
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph
from sna.core.provenance import execution_metadata, sha256, write_json
from sna.data import extract_candidate_edges
from sna.models import GCN
from . import stats
from .core import (agreement, baseline_scores, fidelity_selected, ground_truth_from_labels,
                   pair_labels, rank_edges, top_indices, validate_scores)

EXPLAINERS = ('saliency', 'gnnexplainer', 'pgexplainer')
BASELINES = ('random', 'nearest_first')
DETERMINISTIC = {'saliency'}
REQUIRED = ['run_id', 'dataset', 'split', 'arch', 'gnn_seed', 'checkpoint_sha256', 'method',
            'explainer_seed', 'seed', 'node_id', 'target_class', 'edge_id', 'u', 'v',
            'importance_score', 'runtime_sec', 'status']
METRICS = ['fid_drop', 'fid_inv', 'orig_prob', 'roc_auc', 'average_precision', 'precision', 'recall']
AGREEMENT_METRICS = ['kappa', 'jaccard', 'kendall_tau_b']
FILE_RE = re.compile(r'^(saliency|gnnexplainer|pgexplainer)_e(\d+)\.csv$')
DIR_RE = re.compile(r'^gcn_s(\d+)$')
LOGIT_TOL = 1e-5


class HashMismatchError(ValueError):
    """Provenance check failed; evaluation must not continue."""


def tie_seed(*parts):
    text = ':'.join(map(str, parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')


def protocol_tie_seed(dataset, g, node, method, variant, e, t):
    """v2 tie-breaking seed. Baseline repeats do not depend on the GNN."""
    if variant == 'baseline':
        return tie_seed(dataset, node, method, 'r', t)
    return tie_seed(dataset, g, node, method, e, t)


def legacy_tie_seed(dataset, g, node, method, variant, e, t):
    """Midterm runner.py rule (seed_for), used only to reproduce T06."""
    return tie_seed(dataset, node, method, t if variant == 'baseline' else e)


def prepare_legacy_midterm(scores_csv, data_dir, workdir, datasets=('bashapes', 'cora'),
                           n_nodes=20, gnn_seed=0):
    """Lay the midterm T05 CSV out as v2 runs so T06 can be reproduced (GATE 0 / F3).

    The midterm CSV has no checkpoint hash; the hash of data/{ds}/gcn_ckpt.pt is
    attached here, which is an explicit assumption recorded in the returned dict.
    """
    workdir = Path(workdir)
    df = pd.read_csv(scores_csv)
    df = df[df.method.isin(EXPLAINERS)]
    out = {}
    for ds in datasets:
        ckpt = Path(data_dir) / ds / 'gcn_ckpt.pt'
        ck_dir = workdir / 'checkpoints' / ds
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / f'gcn_s{gnn_seed}.pt').write_bytes(ckpt.read_bytes())
        digest = sha256(ckpt)
        val = json.loads((Path(data_dir) / ds / 'val_nodes.json').read_text(encoding='utf-8-sig'))
        nodes = [int(n) for n in val['nodes'][:n_nodes]]
        nodes_file = workdir / ds / 'nodes.json'
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text(json.dumps({'nodes': nodes}), encoding='utf-8')
        part = df[(df.dataset == ds) & df.node_id.isin(nodes)]
        for (method, seed), run in part.groupby(['method', 'seed']):
            run = run.assign(split='val', arch='gcn', gnn_seed=gnn_seed, checkpoint_sha256=digest,
                             explainer_seed=int(seed), runtime_sec=float('nan'), status='success',
                             run_id=f'{ds}_gcn_g{gnn_seed}_{method}_e{int(seed)}_val')
            target = workdir / 'scores' / ds / f'gcn_s{gnn_seed}' / f'{method}_e{int(seed)}.csv'
            target.parent.mkdir(parents=True, exist_ok=True)
            run[REQUIRED].to_csv(target, index=False)
        out[ds] = {'scores': workdir / 'scores' / ds, 'checkpoints': workdir / 'checkpoints',
                   'nodes_file': nodes_file, 'assumed_checkpoint_sha256': digest}
    return out


def load_node_list(path, data_sha256=None):
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    nodes = raw['nodes'] if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and data_sha256 and raw.get('data_sha256') not in (None, data_sha256):
        raise HashMismatchError(f'{path}: node list was built for another data.pt')
    nodes = [int(n) for n in nodes]
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError(f'{path}: node list is empty or has duplicates')
    return nodes


def load_checkpoint(path, data, data_sha256):
    """Load a GCN checkpoint, verify JSON hash, data hash and saved predictions."""
    path = Path(path)
    digest = sha256(path)
    meta = path.with_suffix('.json')
    if meta.is_file():
        recorded = json.loads(meta.read_text(encoding='utf-8')).get('checkpoint_sha256')
        if recorded is not None and recorded != digest:
            raise HashMismatchError(f'{path}: sha256 {digest} != {meta.name} {recorded}')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('data_sha256') not in (None, data_sha256):
        raise HashMismatchError(f'{path}: trained on data {ck["data_sha256"]}, not {data_sha256}')
    model = GCN(ck['in_dim'], ck['hidden'], ck['n_classes'], ck.get('dropout', 0.0))
    model.load_state_dict(ck['state_dict'])
    model.eval().requires_grad_(False)
    with torch.no_grad():
        logits = model(data['x'], data['edge_index'])
    pred = logits.argmax(-1)
    if 'pred' in ck and not torch.equal(pred, torch.as_tensor(ck['pred']).cpu()):
        raise ValueError(f'{path}: recomputed predictions differ from saved pred')
    return model, pred, logits.softmax(-1), digest


def discover_runs(scores_dir):
    """{(g, method, e): csv path} plus run_log.json content per GNN seed."""
    runs, logs = {}, {}
    scores_dir = Path(scores_dir)
    for gdir in sorted(p for p in scores_dir.iterdir() if p.is_dir()):
        m = DIR_RE.match(gdir.name)
        if not m:
            continue
        g = int(m.group(1))
        log = gdir / 'run_log.json'
        if log.is_file():
            logs[g] = json.loads(log.read_text(encoding='utf-8'))
        for csv in sorted(gdir.glob('*.csv')):
            fm = FILE_RE.match(csv.name)
            if fm:
                runs[(g, fm.group(1), int(fm.group(2)))] = csv
    if not runs:
        raise FileNotFoundError(f'No {{method}}_e{{e}}.csv under {scores_dir}/gcn_s*/')
    return runs, logs


def logged_csv_hashes(log):
    out = {}
    for entry in (log or {}).get('runs_by_id', {}).values():
        if entry.get('csv_file') and entry.get('csv_sha256'):
            out[entry['csv_file']] = entry['csv_sha256']
    return out


def read_run(path, dataset, g, method, e, split='test'):
    text_cols = ['run_id', 'dataset', 'split', 'arch', 'checkpoint_sha256', 'method', 'status']
    df = pd.read_csv(path, dtype={c: str for c in text_cols})
    missing = set(REQUIRED) - set(df)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    expect = {'dataset': dataset, 'gnn_seed': g, 'method': method, 'explainer_seed': e,
              'arch': 'gcn', 'split': split}
    for col, value in expect.items():
        if set(df[col].unique()) != {value}:
            raise ValueError(f'{path}: column {col} must be {value!r}, got {sorted(df[col].unique())}')
    if df['checkpoint_sha256'].nunique() != 1:
        raise HashMismatchError(f'{path}: several checkpoint hashes in one run')
    unknown = set(df.status) - {'success', 'failed', 'empty_candidates'}
    if unknown:
        raise ValueError(f'{path}: unknown status {unknown}')
    ok = df[df.status == 'success']
    validate_scores(ok.importance_score)
    if (ok.u >= ok.v).any():
        raise ValueError(f'{path}: success rows must use canonical pairs u < v')
    if ok.duplicated(['node_id', 'edge_id']).any() or ok.duplicated(['node_id', 'u', 'v']).any():
        raise ValueError(f'{path}: duplicate edges for a node')
    return df


def _subgraph(data, node, hops):
    subset, sub_edges, mapping, _ = k_hop_subgraph(int(node), hops, data['edge_index'],
                                                    relabel_nodes=True,
                                                    num_nodes=data['x'].size(0))
    local = {int(n): i for i, n in enumerate(subset.tolist())}
    return {'x': data['x'][subset], 'edge_index': sub_edges}, int(mapping[0]), local


def _gap(sorted_scores, k):
    return float(sorted_scores[k - 1] - sorted_scores[k]) if 0 < k < len(sorted_scores) else float('nan')


def _tie_fraction(scores):
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    return float((counts[inverse] > 1).mean()) if len(scores) else float('nan')


def _ensemble(orders, m):
    """Mean position over seeds; lower mean rank = more important."""
    ranks = np.zeros((len(orders), m))
    for i, order in enumerate(orders):
        ranks[i, order] = np.arange(m)
    return -ranks.mean(axis=0)


def evaluate(dataset, data_path, scores_dir, checkpoints_dir, nodes_file, output,
             budgets=(1, 3, 5, 10), tie_seeds=(0, 1, 2, 3, 4), baseline_seeds=(0, 1, 2, 3, 4),
             expected_data_sha256=None, max_nodes=None, ensemble=True, split='test',
             fidelity_hops=3, metadata=None, explainer_seeds=(0, 1, 2, 3, 4),
             primary_gnn_seed=1000, tie_seed_fn=protocol_tie_seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    budgets, tie_seeds, baseline_seeds = list(budgets), list(tie_seeds), list(baseline_seeds)
    if not tie_seeds or tie_seeds[0] != 0:
        raise ValueError('tie_seeds must start with 0 (tie order 0 is the main result)')
    data_sha = sha256(data_path)
    if expected_data_sha256 and data_sha != expected_data_sha256:
        raise HashMismatchError(f'{data_path}: sha256 {data_sha} != protocol {expected_data_sha256}')
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    num_nodes = data['x'].size(0)
    nodes = load_node_list(nodes_file, data_sha)
    node_list_sha = sha256(nodes_file)
    if max_nodes:
        nodes = nodes[:max_nodes]
    labels_by_pair = pair_labels(data['edge_index'], data.get('gt_edge_mask'))
    undirected = {(min(u, v), max(u, v)) for u, v in data['edge_index'].t().tolist() if u != v}
    degree = defaultdict(int)
    for u, v in undirected:
        degree[u] += 1
        degree[v] += 1

    runs, logs = discover_runs(scores_dir)
    errors, warnings_ = [], []
    file_hashes, checkpoints = {}, {}
    frames = {}
    for (g, method, e), path in sorted(runs.items()):
        digest = sha256(path)
        rel = f'gcn_s{g}/{path.name}'
        file_hashes[rel] = digest
        log = logs.get(g)
        if log is None:
            warnings_.append(f'{rel}: no run_log.json')
        else:
            for key, value in (('data_sha256', data_sha), ('node_list_sha256', node_list_sha)):
                if log.get(key) not in (None, value):
                    raise HashMismatchError(f'gcn_s{g}/run_log.json: {key} {log[key]} != {value}')
            logged = logged_csv_hashes(log).get(path.name)
            if logged is None:
                warnings_.append(f'{rel}: not listed in run_log.json')
            elif logged != digest:
                raise HashMismatchError(f'{rel}: sha256 {digest} != run_log {logged}')
        frames[(g, method, e)] = read_run(path, dataset, g, method, e, split)

    models = {}
    for g in sorted({g for g, _, _ in frames}):
        ckpt = Path(checkpoints_dir) / dataset / f'gcn_s{g}.pt'
        model, pred, prob, digest = load_checkpoint(ckpt, data, data_sha)
        claimed = {frames[key].checkpoint_sha256.iloc[0] for key in frames if key[0] == g}
        if claimed != {digest}:
            raise HashMismatchError(f'{ckpt}: sha256 {digest} but CSVs claim {sorted(claimed)}')
        models[g] = (model, pred, prob)
        checkpoints[g] = {'path': str(ckpt.as_posix()), 'sha256': digest}

    metrics, store, preds = [], {}, {g: models[g][1] for g in models}
    for node in nodes:
        cand = extract_candidate_edges(data['edge_index'], node, num_hops=2, num_nodes=num_nodes)
        pairs = [tuple(map(int, p)) for p in cand.candidate_pairs]
        by_eid = dict(zip(map(int, cand.edge_ids), pairs))
        canon = sorted(range(len(pairs)), key=lambda i: pairs[i])
        pairs = [pairs[i] for i in canon]
        eid_of = {p: eid for eid, p in by_eid.items()}
        m = len(pairs)
        labels = None if labels_by_pair is None else [labels_by_pair[p] for p in pairs]
        n_motif = None if labels is None else int(sum(labels))
        sub, local_node, local = _subgraph(data, node, fidelity_hops)
        local_pairs = [(local[u], local[v]) for u, v in pairs]
        nearest = baseline_scores(pairs, node, 'nearest_first') if m else np.zeros(0)
        for g, (model, pred, prob) in models.items():
            target = int(pred[node])
            base = dict(dataset=dataset, split=split, gnn_seed=g,
                        checkpoint_sha256=checkpoints[g]['sha256'], node_id=node,
                        target_class=target, true_label=int(data['y'][node]),
                        pred_correct=bool(target == int(data['y'][node])), m=m,
                        n_motif_candidates=n_motif, node_degree=degree[node])
            full_prob = float(prob[node, target])
            fid_cache = {}
            if m:
                with torch.no_grad():
                    sub_prob = float(model(sub['x'], sub['edge_index'])[local_node].softmax(-1)[target])
                use_sub = abs(sub_prob - full_prob) < LOGIT_TOL
                if not use_sub:
                    errors.append(dict(dataset=dataset, gnn_seed=g, node_id=node, method='fidelity',
                                       reason=f'subgraph_prob_mismatch:{abs(sub_prob - full_prob):.2e}'))

            def fid(selected):
                key = frozenset(int(i) for i in selected)
                if key not in fid_cache:
                    if use_sub:
                        fid_cache[key] = fidelity_selected(model, sub, local_node, target,
                                                           local_pairs, selected, full_prob)
                    else:
                        fid_cache[key] = fidelity_selected(model, data, node, target, pairs,
                                                           selected, full_prob)
                return fid_cache[key]

            runs_here = {}
            for (rg, method, e), df in frames.items():
                if rg != g:
                    continue
                part = df[df.node_id == node]
                status = set(part.status)
                row_id = dict(method=method, variant='single', explainer_seed=e)
                if part.empty:
                    errors.append(dict(dataset=dataset, gnn_seed=g, node_id=node, method=method,
                                       explainer_seed=e, reason='missing_node_in_run'))
                    continue
                if status != {'success'}:
                    kind = 'empty_candidates' if status == {'empty_candidates'} else 'failed'
                    if kind == 'failed' or m:
                        errors.append(dict(dataset=dataset, gnn_seed=g, node_id=node, method=method,
                                           explainer_seed=e, reason=f'run_status_{kind}'))
                    metrics.append({**base, **row_id, 'tie_index': 0, 'status': kind})
                    continue
                if set(part.target_class) != {target}:
                    errors.append(dict(dataset=dataset, gnn_seed=g, node_id=node, method=method,
                                       explainer_seed=e, reason='target_class_differs_from_checkpoint'))
                    continue
                got = dict(zip(part.edge_id.astype(int), zip(part.u.astype(int), part.v.astype(int))))
                if got != by_eid:
                    errors.append(dict(dataset=dataset, gnn_seed=g, node_id=node, method=method,
                                       explainer_seed=e, reason='candidate_edges_differ_from_graph'))
                    continue
                score_of = dict(zip(zip(part.u.astype(int), part.v.astype(int)),
                                    part.importance_score.astype(float)))
                runs_here[(method, 'single', e)] = np.array([score_of[p] for p in pairs])
            if not m:
                continue
            if ensemble:
                for method in EXPLAINERS:
                    seeds = sorted(e for (mm, v, e) in runs_here if mm == method and v == 'single')
                    if len(seeds) > 1 and method not in DETERMINISTIC:
                        orders = [rank_edges(runs_here[(method, 'single', e)],
                                             tie_seed_fn(dataset, g, node, method, 'single', e, 0))
                                  for e in seeds]
                        runs_here[(method, 'ensemble', -1)] = _ensemble(orders, m)
            for b in BASELINES:
                runs_here[(b, 'baseline', -1)] = nearest if b == 'nearest_first' else np.zeros(m)

            for (method, variant, e), scores in runs_here.items():
                sorted_scores = np.sort(scores)[::-1]
                gt_full = ground_truth_from_labels(labels, scores, [])
                extra = dict(tie_frac=_tie_fraction(scores), roc_auc=gt_full['roc_auc'],
                             average_precision=gt_full['average_precision'],
                             auc_reason=gt_full['auc_reason'])
                repeats = baseline_seeds if variant == 'baseline' else tie_seeds
                cell = store.setdefault((g, node), {})
                for t in repeats:
                    seed = tie_seed_fn(dataset, g, node, method, variant, e, t)
                    order = rank_edges(scores, seed)
                    # Agreement uses tie order 0 for explainers and every baseline repeat.
                    if variant == 'baseline':
                        cell[(method, variant, t)] = (scores, order)
                    elif t == 0:
                        cell[(method, variant, e)] = (scores, order)
                    for k in budgets:
                        selected = top_indices(order, k)
                        gt = ground_truth_from_labels(labels, scores, selected)
                        metrics.append({
                            **base, 'method': method, 'variant': variant, 'explainer_seed': e,
                            'tie_index': t, 'tie_seed': seed, 'budget': k, 'k_eff': len(selected),
                            **fid(selected), **extra,
                            'precision': gt['precision'], 'recall': gt['recall'],
                            'n_1hop_in_topk': int(sum(node in pairs[i] for i in selected)),
                            'gap_k_k1': _gap(sorted_scores, len(selected)),
                            # Selected edges in rank order, so A3 can reuse the exact tie-break.
                            'topk_edge_ids': ';'.join(str(eid_of[pairs[i]]) for i in selected),
                            'status': 'success'})

    agreements = _agreements(dataset, store, preds, nodes, budgets, baseline_seeds)
    metrics_df = pd.DataFrame(metrics)
    agree_df = pd.DataFrame(agreements)
    errors_df = pd.DataFrame(errors, columns=['dataset', 'gnn_seed', 'node_id', 'method',
                                              'explainer_seed', 'reason'])
    outputs = {'metrics.csv': metrics_df, 'agreement.csv': agree_df, 'errors.csv': errors_df,
               'summary.csv': summarize_metrics(metrics_df, pooled=True),
               'summary_by_config.csv': summarize_metrics(metrics_df, pooled=False),
               'agreement_summary.csv': summarize_agreement(agree_df)}
    for name, df in outputs.items():
        df.to_csv(output / name, index=False)

    expected = expected_runs(sorted(models), explainer_seeds, primary_gnn_seed)
    manifest = {
        **(metadata or execution_metadata()), 'mode': 'evaluate_v2', 'dataset': dataset,
        'split': split, 'status': 'completed_with_errors' if errors else 'completed',
        'data_path': str(Path(data_path).as_posix()), 'data_sha256': data_sha,
        'node_list_path': str(Path(nodes_file).as_posix()), 'node_list_sha256': node_list_sha,
        'n_nodes': len(nodes), 'max_nodes': max_nodes, 'checkpoints': checkpoints,
        'score_files_sha256': file_hashes,
        'runs': sorted(f'g{g}/{method}/e{e}' for g, method, e in frames),
        'missing_protocol_runs': sorted(f'g{g}/{mt}/e{e}' for g, mt, e in expected - set(frames)),
        'protocol': {'budgets': budgets, 'tie_seeds': tie_seeds, 'baseline_repeat_seeds': baseline_seeds,
                     'tie_policy': 'seeded uniform permutation per (dataset, g, node, method, e, t); '
                                   'one order reused across budgets',
                     'baseline_policy': 'repeats averaged within node before aggregation',
                     'fidelity': f'features fixed; exact {fidelity_hops}-hop subgraph of the target',
                     'agreement': 'kappa@k (main), jaccard@k, kendall tau-b on tie_index 0',
                     'ensemble': 'mean rank position over explainer seeds (D1)' if ensemble else None,
                     'statistics': f'bootstrap over nodes B={stats.BOOTSTRAP_REPEATS}, '
                                   f'{stats.CONFIDENCE:.0%}, sd ddof=1'},
        'warnings': warnings_, 'error_count': len(errors),
        'rows': {name: len(df) for name, df in outputs.items()},
        'outputs_sha256': {name: sha256(output / name) for name in outputs},
    }
    write_json(output / 'manifest.json', manifest)
    return manifest


def expected_runs(gnn_seeds, explainer_seeds=(0, 1, 2, 3, 4), primary=1000):
    """Runs the locked protocol asks for (used to report gaps, not to fail)."""
    out = set()
    for g in gnn_seeds:
        for method in ('gnnexplainer', 'pgexplainer'):
            out |= {(g, method, e) for e in explainer_seeds}
        out |= {(g, 'saliency', e) for e in ((0, 1) if g == primary else (0,))}
    return out


def _seeds(store, g, node, method, variant):
    return sorted(k[2] for k in store.get((g, node), {}) if k[:2] == (method, variant))


def _pick(store, g, node, method, variant, e):
    """Run for (method, variant, e); deterministic methods fall back to their first seed."""
    if (method, variant, e) in store.get((g, node), {}):
        return (g, node, method, variant, e), False
    if method in DETERMINISTIC:
        seeds = _seeds(store, g, node, method, 'single')
        if seeds:
            return (g, node, method, 'single', seeds[0]), True
    return None, False


def _has(store, key):
    return key[2:] in store.get(key[:2], {})


def _agreements(dataset, store, preds, nodes, budgets, baseline_seeds):
    rows = []
    gs = sorted(preds)

    def add(kind, ka, kb, broadcast=False):
        (sa, oa), (sb, ob) = store[ka[:2]][ka[2:]], store[kb[:2]][kb[2:]]
        same = bool(int(preds[ka[0]][ka[1]]) == int(preds[kb[0]][kb[1]]))
        for k in budgets:
            out = agreement(sa, sb, k, order_a=oa, order_b=ob)
            rows.append(dict(dataset=dataset, kind=kind, node_id=ka[1], m=len(sa), budget=k,
                             k_eff=min(k, len(sa)), method_a=ka[2], variant_a=ka[3],
                             gnn_seed_a=ka[0], explainer_seed_a=ka[4], method_b=kb[2],
                             variant_b=kb[3], gnn_seed_b=kb[0], explainer_seed_b=kb[4],
                             same_pred=same, broadcast=broadcast,
                             **{key: out[key] for key in ('kappa', 'kappa_reason', 'jaccard',
                                                          'kendall_tau_b', 'tau_reason')}))

    for node in nodes:
        for g in gs:
            for method in EXPLAINERS:
                for ea, eb in combinations(_seeds(store, g, node, method, 'single'), 2):
                    add('within', (g, node, method, 'single', ea), (g, node, method, 'single', eb))
            for variant in ('single', 'ensemble'):
                for ma, mb in combinations(EXPLAINERS, 2):
                    seeds = sorted(set(_seeds(store, g, node, ma, variant)) |
                                   set(_seeds(store, g, node, mb, variant)))
                    if variant == 'ensemble':
                        seeds = [-1]
                    for e in seeds:
                        ka, fa = _pick(store, g, node, ma, variant, e)
                        kb, fb = _pick(store, g, node, mb, variant, e)
                        if ka and kb and not (fa and fb):
                            add('between' if variant == 'single' else 'between_ensemble',
                                ka, kb, fa or fb)
            for run in [k for k in store.get((g, node), {}) if k[0] in EXPLAINERS]:
                for r in baseline_seeds:
                    ref = (g, node, 'nearest_first', 'baseline', r)
                    if _has(store, ref):
                        add('ref_nearest', (g, node, *run), ref)
        g0 = gs[0] if gs else None
        for ra, rb in combinations(baseline_seeds, 2):
            ka, kb = (g0, node, 'random', 'baseline', ra), (g0, node, 'random', 'baseline', rb)
            if _has(store, ka) and _has(store, kb):
                add('ref_random', ka, kb)
        for ga, gb in combinations(gs, 2):
            for method in EXPLAINERS:
                for variant in ('single', 'ensemble'):
                    seeds = sorted(set(_seeds(store, ga, node, method, variant)) &
                                   set(_seeds(store, gb, node, method, variant)))
                    for e in seeds:
                        add('cross_gnn' if variant == 'single' else 'cross_gnn_ensemble',
                            (ga, node, method, variant, e), (gb, node, method, variant, e))
    return rows


def _node_level(df, keys):
    """Average repeats (baseline) or configs within node, per metric."""
    cols = [c for c in METRICS if c in df]
    return df.groupby(keys + ['node_id'], dropna=False)[cols].mean().reset_index()


def summarize_metrics(metrics, pooled=True):
    if metrics.empty or 'budget' not in metrics:
        return pd.DataFrame()
    ok = metrics[metrics.status == 'success']
    # Explainers: tie order 0 is the main result; baselines: average their repeats.
    ok = ok[(ok.variant == 'baseline') | (ok.tie_index == 0)]
    keys = ['dataset', 'method', 'variant', 'budget']
    if not pooled:
        keys = keys[:3] + ['gnn_seed', 'explainer_seed', 'budget']
    per_config = _node_level(ok, keys if not pooled else keys + ['gnn_seed', 'explainer_seed'])
    per_node = _node_level(per_config, keys)
    rows = []
    for group, part in per_node.groupby(keys, dropna=False):
        for metric in [c for c in METRICS if c in part]:
            rows.append({**dict(zip(keys, group)), 'metric': metric,
                         **stats.summarize(part[metric].to_numpy())})
    return pd.DataFrame(rows)


def summarize_agreement(agree):
    if agree.empty:
        return pd.DataFrame()
    df = agree[~agree.kind.str.startswith('cross_gnn') | agree.same_pred]
    keys = ['dataset', 'kind', 'method_a', 'variant_a', 'method_b', 'variant_b', 'budget']
    per_node = df.groupby(keys + ['node_id'])[AGREEMENT_METRICS].mean().reset_index()
    rows = []
    for group, part in per_node.groupby(keys):
        for metric in AGREEMENT_METRICS:
            rows.append({**dict(zip(keys, group)), 'metric': metric,
                         **stats.summarize(part[metric].to_numpy())})
    out = pd.DataFrame(rows)
    out['note'] = np.where(out.kind.str.startswith('cross_gnn'), 'same_pred_only', '')
    return out
