"""F3: bộ đánh giá v2 (sna.evaluation.runner_v2). Dữ liệu tổng hợp nhỏ, không cần tài sản Drive."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import torch

from sna.data import extract_candidate_edges
from sna.evaluation import core
from sna.evaluation.runner import read_scores, sha256, seed_for
from sna.evaluation.runner_v2 import (EvalConfig, EvaluationError, ProvenanceError, run, main,
                                      config_from_protocol)
from sna.models import GCN

ROOT = Path(__file__).resolve().parents[1]
N = 14
BUDGETS = (1, 3, 5)


def build_graph():
    rng = np.random.default_rng(3)
    und = set()
    for u in range(N - 2):                       # node N-1 bị cô lập => không có cạnh ứng viên
        und.add((u, u + 1))
    while len(und) < 24:
        a, b = sorted(rng.choice(N - 2, 2, replace=False).tolist())
        und.add((a, b))
    und = sorted(und)
    edge_index = torch.tensor([[u for u, v in und] + [v for u, v in und],
                               [v for u, v in und] + [u for u, v in und]])
    gt = torch.tensor([(u + v) % 3 == 0 for u, v in und] * 2)
    x = torch.tensor(rng.normal(size=(N, 3)), dtype=torch.float)
    test = torch.zeros(N, dtype=torch.bool)
    test[[1, 3, 5, 6, N - 1]] = True
    return {'x': x, 'edge_index': edge_index, 'gt_edge_mask': gt,
            'train_mask': ~test, 'val_mask': torch.zeros(N, dtype=torch.bool), 'test_mask': test}


def make_checkpoint(path, data, seed):
    torch.manual_seed(seed)
    model = GCN(3, 6, 3).eval()
    with torch.no_grad():
        pred = model(data['x'], data['edge_index']).argmax(-1)
    torch.save({'state_dict': model.state_dict(), 'in_dim': 3, 'hidden': 6, 'n_classes': 3,
                'pred': pred}, path)
    return pred


class World:
    """Dữ liệu + checkpoint + CSV v2 tổng hợp, dựng trong tmp_path."""

    def __init__(self, tmp_path, gnn_seeds=(1000,), same_checkpoint=False, differing_pred=False):
        self.root = tmp_path
        self.data = build_graph()
        torch.save(self.data, tmp_path / 'data.pt')
        (tmp_path / 'ck').mkdir()
        self.nodes = [1, 3, 5, 6, N - 1]
        self.nodes_path = tmp_path / 'test_nodes.json'
        self.nodes_path.write_text(json.dumps({'nodes': self.nodes}))
        self.preds, self.sha = {}, {}
        seeds_for_weights = {}
        for i, g in enumerate(gnn_seeds):
            seeds_for_weights[g] = 0 if same_checkpoint else i
        if differing_pred and len(gnn_seeds) > 1:
            base = None
            for trial in range(1, 200):
                tmp = tmp_path / 'probe.pt'
                p0 = make_checkpoint(tmp, self.data, 0)
                p1 = make_checkpoint(tmp, self.data, trial)
                same = [bool(p0[n] == p1[n]) for n in self.nodes]
                if not all(same) and any(same):
                    seeds_for_weights[gnn_seeds[1]] = trial
                    break
            else:
                raise RuntimeError('không tìm được cặp checkpoint đổi dự đoán một phần')
        for g in gnn_seeds:
            path = tmp_path / 'ck' / f'gcn_s{g}.pt'
            self.preds[g] = make_checkpoint(path, self.data, seeds_for_weights[g])
            self.sha[g] = sha256(path)
        self.scores_root = tmp_path / 'scores'
        self.rng = np.random.default_rng(11)
        self.pairs = {n: extract_candidate_edges(self.data['edge_index'], n, 2, N).candidate_pairs
                      for n in self.nodes}
        self.written = {}

    def write(self, g, method, e, scores=None, declared_sha=None, mutate=None):
        """Ghi CSV v2. ``scores``: {node: mảng điểm theo thứ tự pairs}; mặc định ngẫu nhiên (không hòa)."""
        rows = []
        for n in self.nodes:
            pairs = self.pairs[n]
            if not pairs:
                rows.append(dict(node_id=n, target_class=int(self.preds[g][n]), edge_id=-1, u=-1, v=-1,
                                 importance_score=float('nan'), status='empty_candidates'))
                continue
            s = (scores[n] if scores is not None else self.rng.permutation(len(pairs)) + self.rng.random(len(pairs)) * .5)
            for i, ((u, v), val) in enumerate(zip(pairs, s)):
                rows.append(dict(node_id=n, target_class=int(self.preds[g][n]), edge_id=i, u=u, v=v,
                                 importance_score=float(val), status='success'))
        df = pd.DataFrame(rows)
        df = df.assign(run_id=f'bashapes_gcn_g{g}_{method}_e{e}_test', dataset='bashapes', split='test',
                       arch='gcn', gnn_seed=g, checkpoint_sha256=declared_sha or self.sha[g],
                       method=method, explainer_seed=e, seed=e, runtime_sec=0.0)
        if mutate:
            df = mutate(df)
        folder = self.scores_root / 'bashapes' / f'gcn_s{g}'
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / f'{method}_e{e}.csv', index=False)
        self.written[(g, method, e)] = {n: (df[(df.node_id == n) & (df.status == 'success')]
                                             .importance_score.to_numpy()) for n in self.nodes}
        return folder / f'{method}_e{e}.csv'

    def cfg(self, gnn_seeds=(1000,), explainer_seeds=(0, 1), methods=('gnnexplainer',), **kw):
        return EvalConfig(dataset='bashapes', data_path=self.root / 'data.pt',
                          nodes_path=self.nodes_path, out=self.root / 'out', scores_root=self.scores_root,
                          checkpoints=self.root / 'ck', budgets=BUDGETS, main_budget=5,
                          gnn_seeds=tuple(gnn_seeds), explainer_seeds=tuple(explainer_seeds),
                          methods=tuple(methods), baseline_repeats=(0, 1),
                          data_sha256_override=sha256(self.root / 'data.pt'), **kw)


def independent_kappa(a, b, k, m):
    """Cài đặt độc lập của κ@k từ điểm thô (không hòa)."""
    k = min(k, m)
    if m == 0 or k == m:
        return float('nan')
    top_a = set(np.argsort(-np.asarray(a))[:k])
    top_b = set(np.argsort(-np.asarray(b))[:k])
    return (len(top_a & top_b) - k * k / m) / (k - k * k / m)


# ---------------------------------------------------------------------------

def test_checkpoint_hash_mismatch_stops(tmp_path):
    w = World(tmp_path)
    w.write(1000, 'gnnexplainer', 0, declared_sha='0' * 64)
    w.write(1000, 'gnnexplainer', 1)
    with pytest.raises(ProvenanceError, match='LỆCH'):
        run(w.cfg())
    assert not (tmp_path / 'out' / 'manifest.json').exists()


def test_checkpoint_json_and_run_log_mismatch_stop(tmp_path):
    w = World(tmp_path)
    for e in (0, 1):
        w.write(1000, 'gnnexplainer', e)
    (tmp_path / 'ck' / 'gcn_s1000.json').write_text(json.dumps({'checkpoint_sha256': 'f' * 64}))
    with pytest.raises(ProvenanceError, match='gcn_s1000.json'):
        run(w.cfg())
    (tmp_path / 'ck' / 'gcn_s1000.json').write_text(json.dumps({'checkpoint_sha256': w.sha[1000]}))
    folder = tmp_path / 'scores' / 'bashapes' / 'gcn_s1000'
    (folder / 'run_log.json').write_text(json.dumps({'checkpoint_sha256': 'e' * 64}))
    with pytest.raises(ProvenanceError, match='run_log.json'):
        run(w.cfg())
    (folder / 'run_log.json').write_text(json.dumps({'checkpoint_sha256': w.sha[1000],
                                                     'node_list_sha256': 'd' * 64}))
    with pytest.raises(ProvenanceError, match='file node'):
        run(w.cfg())


def test_data_hash_mismatch_stops(tmp_path):
    w = World(tmp_path)
    for e in (0, 1):
        w.write(1000, 'gnnexplainer', e)
    cfg = w.cfg()
    cfg.data_sha256_override = '1' * 64
    with pytest.raises(ProvenanceError, match='data'):
        run(cfg)
    # Không override: protocol_final.yaml có hash khác dữ liệu tổng hợp => dừng
    cfg.data_sha256_override = None
    with pytest.raises(ProvenanceError, match='protocol'):
        run(cfg)


def test_cli_returns_error_code_on_hash_mismatch(tmp_path, capsys):
    w = World(tmp_path)
    w.write(1000, 'gnnexplainer', 0, declared_sha='0' * 64)
    code = main(['--dataset', 'bashapes', '--data', str(tmp_path / 'data.pt'),
                 '--scores-root', str(w.scores_root), '--checkpoints', str(tmp_path / 'ck'),
                 '--nodes', str(w.nodes_path), '--out', str(tmp_path / 'o'), '--gnn-seeds', '1000',
                 '--explainer-seeds', '0', '--methods', 'gnnexplainer'])
    assert code == 2   # lỗi hash (protocol hash của data tổng hợp cũng sẽ lệch); đều dừng, không có kết quả
    assert not (tmp_path / 'o' / 'manifest.json').exists()


def test_two_seeds_give_within_pair_and_hand_computed_kappa(tmp_path):
    w = World(tmp_path)
    w.write(1000, 'gnnexplainer', 0)
    w.write(1000, 'gnnexplainer', 1)
    manifest = run(w.cfg())
    assert manifest['status'] == 'completed' and manifest['error_count'] == 0
    ag = pd.read_csv(tmp_path / 'out' / 'agreement.csv')
    within = ag[ag.kind == 'within']
    assert set(zip(within.e_a, within.e_b)) == {(0, 1)}     # đúng 1 cặp e < e'
    assert set(within.node_id) == set(w.nodes)             # mọi node, kể cả node rỗng
    a, b = w.written[(1000, 'gnnexplainer', 0)], w.written[(1000, 'gnnexplainer', 1)]
    for n in w.nodes[:-1]:
        for k in BUDGETS:
            row = within[(within.node_id == n) & (within.budget == k)].iloc[0]
            expect = independent_kappa(a[n], b[n], k, len(w.pairs[n]))
            if math.isnan(expect):
                assert math.isnan(row.kappa) and row.kappa_reason
            else:
                assert row.kappa == pytest.approx(expect)
            top_a, top_b = set(np.argsort(-a[n])[:k]), set(np.argsort(-b[n])[:k])
            if top_a | top_b:
                assert row.jaccard == pytest.approx(len(top_a & top_b) / len(top_a | top_b))
    # bảng tóm tắt: SD ddof=1 giữa node
    summ = pd.read_csv(tmp_path / 'out' / 'agreement_summary.csv')
    r = summ[(summ.kind == 'within') & (summ.budget == 3) & (summ.method_a == 'gnnexplainer')
             & (summ.metric == 'kappa')].iloc[0]
    vals = within[(within.budget == 3)].groupby('node_id').kappa.mean().dropna()
    assert r['mean'] == pytest.approx(vals.mean()) and r['std'] == pytest.approx(vals.std(ddof=1))
    assert r.n_valid_nodes == len(vals)


def test_identical_csvs_give_kappa_and_jaccard_one(tmp_path):
    w = World(tmp_path)
    shared = {n: np.random.default_rng(n).permutation(len(w.pairs[n])) + 0.25 for n in w.nodes if w.pairs[n]}
    shared[N - 1] = np.zeros(0)
    w.write(1000, 'gnnexplainer', 0, scores=shared)
    w.write(1000, 'gnnexplainer', 1, scores=shared)
    run(w.cfg())
    ag = pd.read_csv(tmp_path / 'out' / 'agreement.csv')
    within = ag[(ag.kind == 'within') & (ag.node_id != N - 1)]
    assert (within.jaccard == 1).all()
    defined = within[within.kappa.notna()]
    assert len(defined) > 0 and np.allclose(defined.kappa, 1.0)
    undefined = within[within.kappa.isna()]
    assert (undefined.k_eff == undefined.n_candidates).all() and (undefined.kappa_reason != '').all()
    assert np.allclose(within.kendall_tau_b.dropna(), 1.0)


def test_empty_candidates_node_does_not_break(tmp_path):
    w = World(tmp_path)
    for e in (0, 1):
        w.write(1000, 'gnnexplainer', e)
    manifest = run(w.cfg())
    assert manifest['status'] == 'completed'
    metrics = pd.read_csv(tmp_path / 'out' / 'metrics.csv')
    iso = metrics[metrics.node_id == N - 1]
    assert len(iso) > 0 and (iso.status == 'empty_candidates').all()
    assert iso[['fid_drop', 'fid_inv', 'roc_auc', 'precision']].isna().all().all()
    assert (iso.reason == 'empty_candidates').all()
    ag = pd.read_csv(tmp_path / 'out' / 'agreement.csv')
    iso_ag = ag[ag.node_id == N - 1]
    assert len(iso_ag) > 0 and iso_ag.kappa.isna().all() and (iso_ag.kappa_reason == 'no_candidates').all()
    # node bình thường vẫn có giá trị
    assert metrics[(metrics.node_id == 3) & (metrics.kind == 'explainer')].fid_drop.notna().all()


def test_fidelity_matches_hand_computation(tmp_path):
    """Fid- và Fid+ cho top-1 tính tay bằng cách lọc cạnh, độc lập với fidelity_selected."""
    w = World(tmp_path)
    w.write(1000, 'gnnexplainer', 0)
    w.write(1000, 'gnnexplainer', 1)
    run(w.cfg())
    metrics = pd.read_csv(tmp_path / 'out' / 'metrics.csv')
    model = GCN(3, 6, 3)
    ck = torch.load(tmp_path / 'ck' / 'gcn_s1000.pt', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    edges = w.data['edge_index']
    n = 3
    scores = w.written[(1000, 'gnnexplainer', 0)][n]
    best = w.pairs[n][int(np.argmax(scores))]
    target = int(w.preds[1000][n])

    def prob(e):
        with torch.no_grad():
            return float(model(w.data['x'], e)[n].softmax(-1)[target])
    is_best = torch.tensor([(min(u, v), max(u, v)) == best for u, v in edges.t().tolist()])
    original = prob(edges)
    expected_drop = original - prob(edges[:, ~is_best])
    expected_inv = original - prob(edges[:, is_best])
    row = metrics[(metrics.method == 'gnnexplainer') & (metrics.explainer_seed == 0)
                  & (metrics.node_id == n) & (metrics.budget == 1)].iloc[0]
    assert row.fid_drop == pytest.approx(expected_drop, abs=1e-6)
    assert row.fid_inv == pytest.approx(expected_inv, abs=1e-6)
    # AUROC theo nhãn motif: so với sklearn trực tiếp
    from sklearn.metrics import roc_auc_score
    labels = np.array([(u + v) % 3 == 0 for u, v in w.pairs[n]], dtype=int)
    if 0 < labels.sum() < len(labels):
        assert row.roc_auc == pytest.approx(roc_auc_score(labels, scores))
    else:
        assert math.isnan(row.roc_auc) and row.auc_reason


def test_baselines_present_with_repeats_and_shared_tie_order(tmp_path):
    w = World(tmp_path)
    for e in (0, 1):
        w.write(1000, 'gnnexplainer', e)
    run(w.cfg())
    metrics = pd.read_csv(tmp_path / 'out' / 'metrics.csv')
    base = metrics[metrics.kind == 'baseline']
    assert set(base.method) == {'random', 'nearest_first'}
    assert set(base.explainer_seed) == {0, 1}              # 2 repeat theo cấu hình thử
    one = base[(base.method == 'random') & (base.node_id == 3) & (base.explainer_seed == 0)]
    assert one.tie_seed.nunique() == 1 and len(one) == len(BUDGETS)   # một thứ tự cho mọi budget
    assert one.tie_seed.iloc[0] == seed_for('bashapes', 3, 'random:g1000', 0)


def test_cross_gnn_filters_changed_predictions_and_counts_pairs(tmp_path):
    w = World(tmp_path, gnn_seeds=(1000, 1001), differing_pred=True)
    for g in (1000, 1001):
        for e in (0, 1):
            w.write(g, 'gnnexplainer', e)
    manifest = run(w.cfg(gnn_seeds=(1000, 1001)))
    assert manifest['status'] == 'completed'
    ag = pd.read_csv(tmp_path / 'out' / 'agreement.csv')
    cross = ag[ag.kind == 'cross_gnn']
    assert set(zip(cross.g_a, cross.g_b)) == {(1000, 1001)}
    assert set(cross.e_a) == {0, 1}
    same = [n for n in w.nodes if w.preds[1000][n] == w.preds[1001][n]]
    changed = [n for n in w.nodes if n not in same]
    assert changed and same
    # node đổi dự đoán: NaN với lý do; node giữ dự đoán: có giá trị = công thức độc lập
    assert (cross[cross.node_id.isin(changed)].kappa.isna()).all()
    assert (cross[cross.node_id.isin(changed)].kappa_reason == 'pred_changed').all()
    assert not cross[cross.node_id.isin(changed)].pred_same.any()
    for n in same:
        if not w.pairs[n]:
            continue
        for k in BUDGETS:
            for e in (0, 1):
                row = cross[(cross.node_id == n) & (cross.budget == k) & (cross.e_a == e)].iloc[0]
                expect = independent_kappa(w.written[(1000, 'gnnexplainer', e)][n],
                                           w.written[(1001, 'gnnexplainer', e)][n], k, len(w.pairs[n]))
                assert (math.isnan(expect) and math.isnan(row.kappa)) or row.kappa == pytest.approx(expect)
    pc = pd.read_csv(tmp_path / 'out' / 'pred_change.csv').iloc[0]
    assert pc.n_pred_changed == len(changed) and pc.n_nodes == len(w.nodes)
    assert pc.pred_change_rate == pytest.approx(len(changed) / len(w.nodes))
    # mỗi g có within riêng
    within = ag[ag.kind == 'within']
    assert set(within.g_a) == {1000, 1001}


def test_parts_then_aggregate_equals_single_run(tmp_path):
    w = World(tmp_path, gnn_seeds=(1000, 1001), differing_pred=True)
    for g in (1000, 1001):
        for e in (0, 1):
            w.write(g, 'gnnexplainer', e)
            w.write(g, 'saliency', 0) if e == 0 else None
    methods = ('saliency', 'gnnexplainer')
    one = w.cfg(gnn_seeds=(1000, 1001), methods=methods)
    one.out = tmp_path / 'single'
    run(one)
    split = w.cfg(gnn_seeds=(1000, 1001), methods=methods)
    split.out = tmp_path / 'split'
    for g in (1000, 1001):
        part = w.cfg(gnn_seeds=(g,), methods=methods)
        part.out = tmp_path / 'split'
        run(part, 'parts')
    with pytest.raises(EvaluationError, match='Thiếu phần'):
        bad = w.cfg(gnn_seeds=(1000, 1002), methods=methods)
        bad.out = tmp_path / 'split'
        run(bad, 'aggregate')
    run(split, 'aggregate')
    for name in ('metrics.csv', 'agreement.csv', 'summary.csv', 'agreement_summary.csv', 'pred_change.csv'):
        a = pd.read_csv(tmp_path / 'single' / name)
        b = pd.read_csv(tmp_path / 'split' / name)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
    # saliency chỉ e0 mà vẫn có between với gnnexplainer ở mọi e (dùng e0 của saliency)
    ag = pd.read_csv(tmp_path / 'split' / 'agreement.csv')
    btw = ag[(ag.kind == 'between') & (ag.method_a == 'saliency')]
    assert set(btw.e_a) == {0} and set(btw.e_b) == {0, 1}
    # between_avg tồn tại, và là thứ hạng trung bình
    assert (ag.kind == 'between_avg').any()
    assert set(ag.kind) >= {'within', 'cross_gnn', 'between', 'between_avg', 'ref_nearest_first', 'ref_random'}


def test_between_avg_hand_computed(tmp_path):
    """Thứ hạng trung bình qua 2 seed của A so với B: kiểm tra tay trên một node."""
    w = World(tmp_path)
    n = 3
    m = len(w.pairs[n])
    assert m >= 6
    base = {x: np.arange(len(w.pairs[x]), dtype=float) + 0.5 for x in w.nodes if w.pairs[x]}
    base[N - 1] = np.zeros(0)
    rev = {x: -v for x, v in base.items()}
    # A: e0 = base, e1 = đảo; trung bình thứ hạng hòa => kết quả phụ thuộc phá hòa nhưng B = base rõ ràng
    w.write(1000, 'gnnexplainer', 0, scores=base)
    w.write(1000, 'gnnexplainer', 1, scores=rev)
    w.write(1000, 'pgexplainer', 0, scores=base)
    w.write(1000, 'pgexplainer', 1, scores=base)
    run(w.cfg(methods=('gnnexplainer', 'pgexplainer')))
    ag = pd.read_csv(tmp_path / 'out' / 'agreement.csv')
    avg = ag[(ag.kind == 'between_avg') & (ag.node_id == n)]
    assert set(avg.method_a) == {'gnnexplainer'} and set(avg.method_b) == {'pgexplainer'}
    # Ā = trung bình thứ hạng của (base, rev): mọi cạnh có trung bình (m-1)/2 => hòa hoàn toàn
    row = avg[avg.budget == 3].iloc[0]
    assert row.kendall_tau_b != row.kendall_tau_b      # điểm trung bình hằng => τ_b NaN
    # between thường (g, e): e0 giống nhau => κ = 1; e1 đảo => κ thấp hơn
    btw = ag[(ag.kind == 'between') & (ag.node_id == n) & (ag.budget == 3)]
    assert btw[btw.e_a == 0].kappa.iloc[0] == pytest.approx(1.0)
    assert btw[btw.e_a == 1].kappa.iloc[0] < 1.0


def test_missing_csv_is_reported_not_silent(tmp_path):
    w = World(tmp_path)
    w.write(1000, 'gnnexplainer', 0)          # thiếu e1
    manifest = run(w.cfg())
    assert manifest['status'] == 'completed_with_errors'
    errors = pd.read_csv(tmp_path / 'out' / 'errors.csv')
    assert errors.reason.str.contains('missing_csv').any()


def test_manifest_contents(tmp_path):
    w = World(tmp_path)
    for e in (0, 1):
        w.write(1000, 'gnnexplainer', e)
    manifest = run(w.cfg())
    on_disk = json.loads((tmp_path / 'out' / 'manifest.json').read_text(encoding='utf-8'))
    assert on_disk == json.loads(json.dumps(manifest))
    for key in ('git_commit', 'git_dirty', 'argv', 'utc', 'python', 'versions', 'device', 'data_sha256',
                'checkpoints', 'score_csv_sha256', 'nodes_sha256', 'node_ids', 'protocol', 'error_count',
                'status'):
        assert key in manifest, key
    assert manifest['checkpoints']['1000']['sha256'] == w.sha[1000]
    assert set(manifest['versions']) >= {'torch', 'torch-geometric', 'numpy', 'pandas'}
    assert len(manifest['score_csv_sha256']) == 2
    assert manifest['protocol']['budgets'] == list(BUDGETS)


# ---------------------------------------------------------------------------
# Dữ liệu thật (bỏ qua nếu thiếu tệp)
# ---------------------------------------------------------------------------

REAL_SCORES = Path(r'D:/WorkSpace/UIT/SocialNetwork/DoAn/repo/results/scores')


@pytest.mark.skipif(not (REAL_SCORES / 'bashapes/gcn_s1000/pgexplainer_e1.csv').is_file()
                    or not (ROOT / 'results/checkpoints/bashapes/gcn_s1000.pt').is_file()
                    or not (ROOT / 'data/bashapes/data.pt').is_file(),
                    reason='thiếu CSV PGExplainer lượt cũ hoặc checkpoint/data')
def test_real_bashapes_matches_core_computed_directly(tmp_path):
    cfg = config_from_protocol('bashapes', scores_root=REAL_SCORES, out=tmp_path / 'real',
                               nodes_limit=8, gnn_seeds=(1000,), explainer_seeds=(0, 1),
                               methods=('pgexplainer',))
    manifest = run(cfg)
    assert manifest['status'] == 'completed'
    metrics = pd.read_csv(tmp_path / 'real' / 'metrics.csv')
    ag = pd.read_csv(tmp_path / 'real' / 'agreement.csv')
    data = torch.load(ROOT / 'data/bashapes/data.pt', weights_only=False)
    ck = torch.load(ROOT / 'results/checkpoints/bashapes/gcn_s1000.pt', weights_only=False)
    model = GCN(ck['in_dim'], ck['hidden'], ck['n_classes'], ck.get('dropout', 0.0))
    model.load_state_dict(ck['state_dict'])
    model.eval()
    nodes = json.loads((ROOT / 'data/bashapes/test_nodes.json').read_text())['nodes'][:8]
    frames, orders = {}, {}
    for e in (0, 1):
        df = read_scores(REAL_SCORES / f'bashapes/gcn_s1000/pgexplainer_e{e}.csv')
        frames[e] = df[df.node_id.isin(nodes)].sort_values(['node_id', 'u', 'v'])
    checked = 0
    for e in (0, 1):
        for node in nodes:
            part = frames[e][frames[e].node_id == node]
            pairs = list(zip(part.u, part.v))
            scores = part.importance_score.to_numpy(float)
            target = int(part.target_class.iloc[0])
            order = core.rank_edges(scores, seed_for('bashapes', node, 'pgexplainer:g1000', e))
            orders[(e, node)] = (scores, order)
            for k in (1, 3, 5, 10):
                sel = core.top_indices(order, k)
                fid = core.fidelity_selected(model, data, node, target, pairs, sel)
                gt = core.ground_truth_metrics(pairs, scores, data['edge_index'], data['gt_edge_mask'], sel)
                row = metrics[(metrics.kind == 'explainer') & (metrics.explainer_seed == e)
                              & (metrics.node_id == node) & (metrics.budget == k)].iloc[0]
                assert row.fid_drop == pytest.approx(fid['fid_drop'], abs=1e-6)
                assert row.fid_inv == pytest.approx(fid['fid_inv'], abs=1e-6)
                for key in ('roc_auc', 'average_precision', 'precision', 'recall'):
                    assert (math.isnan(gt[key]) and math.isnan(row[key])) or row[key] == pytest.approx(gt[key])
                checked += 1
    assert checked == 2 * len(nodes) * 4
    for node in nodes:
        (sa, oa), (sb, ob) = orders[(0, node)], orders[(1, node)]
        for k in (1, 3, 5, 10):
            expect = core.agreement(sa, sb, k, order_a=oa, order_b=ob)
            row = ag[(ag.kind == 'within') & (ag.node_id == node) & (ag.budget == k)].iloc[0]
            for key in ('jaccard', 'kendall_tau_b', 'kappa'):
                assert (math.isnan(expect[key]) and math.isnan(row[key])) or row[key] == pytest.approx(expect[key])


@pytest.mark.skipif(not (ROOT / 'notebooks/01_midterm/edge_scores.csv').is_file()
                    or not (ROOT / 'notebooks/05_duc_kha_evaluation/results/summary.csv').is_file(),
                    reason='thiếu CSV giữa kỳ hoặc bảng T06')
def test_legacy_midterm_reproduces_t06_to_four_digits(tmp_path):
    cfg = config_from_protocol(
        'bashapes', out=tmp_path / 'legacy', legacy_scores=ROOT / 'notebooks/01_midterm/edge_scores.csv',
        checkpoints=ROOT / 'data/bashapes/gcn_ckpt.pt', nodes_path=ROOT / 'data/bashapes/val_nodes.json',
        nodes_limit=20, split='val', saliency_seeds=(42,), saliency_fixed_gnn_seeds=())
    manifest = run(cfg)
    assert manifest['status'] == 'completed' and manifest['mode'] == 'legacy_midterm'
    ref = pd.read_csv(ROOT / 'notebooks/05_duc_kha_evaluation/results/summary.csv')
    ref = ref[ref.dataset == 'bashapes']
    new = pd.read_csv(tmp_path / 'legacy' / 'summary.csv')
    key = ['dataset', 'method', 'budget', 'metric']
    both = ref.merge(new, on=key, suffixes=('_ref', '_new'))
    assert len(both) == len(ref) == len(new)
    for col in ('mean', 'std'):
        assert np.allclose(both[f'{col}_ref'].fillna(-1), both[f'{col}_new'].fillna(-1), atol=5e-5)
    assert (both.n_valid_nodes_ref == both.n_valid_nodes_new).all()
    assert (both.n_total_nodes_ref == both.n_total_nodes_new).all()
    ref_a = pd.read_csv(ROOT / 'notebooks/05_duc_kha_evaluation/results/agreement_summary.csv')
    ref_a = ref_a[ref_a.dataset == 'bashapes']
    new_a = pd.read_csv(tmp_path / 'legacy' / 'agreement_summary.csv')
    new_a = new_a[new_a.kind.isin(['between', 'ref_nearest_first'])]
    joined = ref_a.merge(new_a, on=['dataset', 'method_a', 'method_b', 'budget', 'metric'], suffixes=('_ref', '_new'))
    assert len(joined) >= 48
    assert np.allclose(joined.mean_ref.fillna(-1), joined.mean_new.fillna(-1), atol=5e-5)
    assert np.allclose(joined.std_ref.fillna(-1), joined.std_new.fillna(-1), atol=5e-5)
