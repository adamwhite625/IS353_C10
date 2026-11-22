"""F3: final-protocol evaluator (runner_v2) on the committed data and checkpoints."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import json
import numpy as np
import pandas as pd
import pytest
import torch
from sna.core.provenance import sha256
from sna.data import extract_candidate_edges
from sna.evaluation.core import fidelity_selected, rank_edges, top_indices
from sna.evaluation.runner_v2 import (HashMismatchError, REQUIRED, evaluate, legacy_tie_seed,
                                      load_checkpoint, prepare_legacy_midterm, protocol_tie_seed)

DS = 'bashapes'
DATA = ROOT / 'data' / DS / 'data.pt'
CKPTS = ROOT / 'results' / 'checkpoints'
NODES = [300, 305, 401]          # motif nodes with both motif and base candidates
BUDGETS = [1, 5]


def write_run(scores_dir, g, method, e, nodes, seed_offset=0, sha=None, drop_node=None):
    data = torch.load(DATA, weights_only=False)
    _, pred, _, digest = load_checkpoint(CKPTS / DS / f'gcn_s{g}.pt', data, sha256(DATA))
    rng = np.random.default_rng(1000 * g + 10 * e + seed_offset)
    rows = []
    for node in nodes:
        if node == drop_node:
            continue
        cand = extract_candidate_edges(data['edge_index'], node, num_hops=2,
                                       num_nodes=data['x'].size(0))
        scores = rng.random(len(cand.candidate_pairs))
        for eid, (u, v), s in zip(cand.edge_ids, cand.candidate_pairs, scores):
            rows.append(dict(run_id=f'{DS}_gcn_g{g}_{method}_e{e}_test', dataset=DS, split='test',
                             arch='gcn', gnn_seed=g, checkpoint_sha256=sha or digest, method=method,
                             explainer_seed=e, seed=e, node_id=node, target_class=int(pred[node]),
                             edge_id=int(eid), u=int(u), v=int(v), importance_score=float(s),
                             runtime_sec=0.01, status='success'))
    path = Path(scores_dir) / f'gcn_s{g}' / f'{method}_e{e}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[REQUIRED].to_csv(path, index=False)
    return path


@pytest.fixture
def grid(tmp_path):
    scores = tmp_path / 'scores'
    for g in (1000, 1001):
        for method in ('gnnexplainer', 'pgexplainer'):
            for e in (0, 1):
                write_run(scores, g, method, e, NODES)
        write_run(scores, g, 'saliency', 0, NODES)
    # Saliency is deterministic: e = 1 on the fixed GNN carries identical scores.
    write_run(scores, 1000, 'saliency', 1, NODES, seed_offset=-10)
    nodes_file = tmp_path / 'nodes.json'
    nodes_file.write_text(json.dumps({'nodes': NODES, 'data_sha256': sha256(DATA)}))
    return tmp_path, scores, nodes_file


def run(tmp_path, scores, nodes_file, **kw):
    return evaluate(DS, DATA, scores, CKPTS, nodes_file, tmp_path / 'eval', budgets=BUDGETS,
                    tie_seeds=kw.pop('tie_seeds', [0, 1]), baseline_seeds=[0, 1, 2],
                    metadata={}, explainer_seeds=[0, 1], **kw)


def test_grid_outputs_pairs_and_provenance(grid):
    tmp_path, scores, nodes_file = grid
    manifest = run(tmp_path, scores, nodes_file)
    assert manifest['status'] == 'completed' and manifest['error_count'] == 0
    assert set(manifest['checkpoints']) == {1000, 1001}
    assert len(manifest['score_files_sha256']) == 11
    out = tmp_path / 'eval'
    agree = pd.read_csv(out / 'agreement.csv')
    n, k = len(NODES), len(BUDGETS)
    counts = agree.groupby('kind').size().to_dict()
    # within: g1000 has saliency/gnn/pg seed pairs, g1001 has gnn/pg.
    assert counts['within'] == n * k * 5
    # between: 3 method pairs × 2 explainer seeds × 2 GNNs (saliency broadcast on e=1 at g1001).
    assert counts['between'] == n * k * 12
    assert counts['between_ensemble'] == n * k * 2 * 3
    # cross-GNN: gnn/pg × seeds {0, 1} + saliency e0, plus gnn/pg ensembles.
    assert counts['cross_gnn'] == n * k * 5 and counts['cross_gnn_ensemble'] == n * k * 2
    assert counts['ref_random'] == n * k * 3
    within_sal = agree[(agree.kind == 'within') & (agree.method_a == 'saliency')]
    assert len(within_sal) == n * k
    metrics = pd.read_csv(out / 'metrics.csv')
    assert set(metrics.variant) == {'single', 'ensemble', 'baseline'}
    # per (node, budget): single runs (6 at g1000, 5 at g1001) × 2 tie orders,
    # 2 ensembles × 2 tie orders per g, 2 baselines × 3 repeats per g.
    assert len(metrics) == n * k * ((6 + 5) * 2 + 2 * (2 * 2) + 2 * (2 * 3))
    summary = pd.read_csv(out / 'summary.csv')
    assert {'mean', 'sd', 'n', 'ci_lo', 'ci_hi'} <= set(summary)
    assert (summary[summary.metric == 'fid_drop'].n == n).all()


def test_saliency_seed_copy_gives_kappa_one(tmp_path):
    scores = tmp_path / 'scores'
    a = write_run(scores, 1000, 'saliency', 0, NODES)
    b = pd.read_csv(a).assign(explainer_seed=1, seed=1,
                              run_id=f'{DS}_gcn_g1000_saliency_e1_test')
    b.to_csv(a.with_name('saliency_e1.csv'), index=False)
    nodes_file = tmp_path / 'nodes.json'
    nodes_file.write_text(json.dumps({'nodes': NODES}))
    run(tmp_path, scores, nodes_file, ensemble=False)
    agree = pd.read_csv(tmp_path / 'eval' / 'agreement.csv')
    within = agree[agree.kind == 'within']
    assert len(within) == len(NODES) * len(BUDGETS)
    assert np.allclose(within.kappa, 1.0) and np.allclose(within.jaccard, 1.0)


def test_metrics_equal_core_on_full_graph(grid):
    tmp_path, scores, nodes_file = grid
    run(tmp_path, scores, nodes_file, ensemble=False)
    metrics = pd.read_csv(tmp_path / 'eval' / 'metrics.csv',
                          dtype={'topk_edge_ids': str, 'checkpoint_sha256': str})
    data = torch.load(DATA, weights_only=False)
    model, pred, _, _ = load_checkpoint(CKPTS / DS / 'gcn_s1001.pt', data, sha256(DATA))
    single = metrics[(metrics.gnn_seed == 1001) & (metrics.variant == 'single')]
    rows = pd.concat([single[single.budget == k].sample(4, random_state=0) for k in BUDGETS])
    for r in rows.itertuples():
        df = pd.read_csv(scores / 'gcn_s1001' / f'{r.method}_e{r.explainer_seed}.csv')
        part = df[df.node_id == r.node_id]
        pairs = sorted(zip(part.u, part.v))
        by_pair = dict(zip(zip(part.u, part.v), part.importance_score))
        s = np.array([by_pair[p] for p in pairs])
        order = rank_edges(s, protocol_tie_seed(DS, 1001, r.node_id, r.method, 'single',
                                                r.explainer_seed, r.tie_index))
        chosen = top_indices(order, r.budget)
        ref = fidelity_selected(model, data, r.node_id, int(pred[r.node_id]), pairs, chosen)
        eid = dict(zip(zip(part.u, part.v), part.edge_id))
        assert r.topk_edge_ids == ';'.join(str(eid[pairs[i]]) for i in chosen)
        assert r.true_label == int(data['y'][r.node_id])
        assert r.checkpoint_sha256 == sha256(CKPTS / DS / 'gcn_s1001.pt')
        assert abs(ref['fid_drop'] - r.fid_drop) < 1e-6
        assert abs(ref['fid_inv'] - r.fid_inv) < 1e-6


def test_csv_checkpoint_hash_mismatch_stops(grid):
    tmp_path, scores, nodes_file = grid
    write_run(scores, 1001, 'pgexplainer', 1, NODES, sha='0' * 64)
    with pytest.raises(HashMismatchError):
        run(tmp_path, scores, nodes_file)


def test_run_log_csv_hash_mismatch_stops(grid):
    tmp_path, scores, nodes_file = grid
    log = {'data_sha256': sha256(DATA), 'runs_by_id': {
        'x': {'csv_file': 'gnnexplainer_e0.csv', 'csv_sha256': 'f' * 64}}}
    (scores / 'gcn_s1000' / 'run_log.json').write_text(json.dumps(log))
    with pytest.raises(HashMismatchError):
        run(tmp_path, scores, nodes_file)


def test_wrong_protocol_data_hash_stops(grid):
    tmp_path, scores, nodes_file = grid
    with pytest.raises(HashMismatchError):
        run(tmp_path, scores, nodes_file, expected_data_sha256='a' * 64)


def test_missing_node_is_reported(grid):
    tmp_path, scores, nodes_file = grid
    write_run(scores, 1000, 'gnnexplainer', 1, NODES, drop_node=NODES[0])
    manifest = run(tmp_path, scores, nodes_file)
    assert manifest['status'] == 'completed_with_errors'
    errors = pd.read_csv(tmp_path / 'eval' / 'errors.csv')
    assert (errors.reason == 'missing_node_in_run').sum() == 1


def test_tie_seeds_must_start_with_zero(grid):
    tmp_path, scores, nodes_file = grid
    with pytest.raises(ValueError):
        run(tmp_path, scores, nodes_file, tie_seeds=[1, 2])


@pytest.mark.parametrize('ds', ['bashapes', 'cora'])
def test_reproduces_midterm_t06(tmp_path, ds):
    """F3 acceptance: T05 CSV through runner_v2 matches the archived T06 tables."""
    t06 = ROOT / 'notebooks/05_duc_kha_evaluation/results'
    legacy = prepare_legacy_midterm(ROOT / 'notebooks/01_midterm/edge_scores.csv', ROOT / 'data',
                                    tmp_path, datasets=[ds])[ds]
    manifest = evaluate(ds, ROOT / 'data' / ds / 'data.pt', legacy['scores'], legacy['checkpoints'],
                        legacy['nodes_file'], tmp_path / 'eval', tie_seeds=[0], ensemble=False,
                        split='val', tie_seed_fn=legacy_tie_seed, metadata={},
                        explainer_seeds=[42], primary_gnn_seed=0)
    assert manifest['error_count'] == 0
    old = pd.read_csv(t06 / 'summary.csv')
    new = pd.read_csv(tmp_path / 'eval' / 'summary.csv')
    j = old[old.dataset == ds].merge(new, on=['dataset', 'method', 'budget', 'metric'])
    assert len(j) == (old.dataset == ds).sum()
    assert (j['n_valid_nodes'] == j['n']).all()
    assert np.allclose(j['mean_x'], j['mean_y'], atol=1e-6, equal_nan=True)
    assert np.allclose(j['std'], j['sd'], atol=1e-6, equal_nan=True)
    old_a = pd.read_csv(t06 / 'agreement_summary.csv')
    new_a = pd.read_csv(tmp_path / 'eval' / 'agreement_summary.csv')
    ja = old_a[old_a.dataset == ds].merge(new_a[new_a.kind == 'between'],
                                          on=['dataset', 'method_a', 'method_b', 'budget', 'metric'])
    assert len(ja) == 24
    assert np.allclose(ja['mean_x'], ja['mean_y'], atol=1e-6, equal_nan=True)
