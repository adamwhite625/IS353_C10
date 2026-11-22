"""Test cho F1 (seed PGExplainer), F2 (schema v2 batch_runner) và F9 (forward 3-hop).

Dùng đồ thị nhỏ tự sinh để không phụ thuộc file dữ liệu và chạy nhanh trên CPU.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch
from torch_geometric.utils import to_undirected

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sna.core.seed import set_global_seed
from sna.data.candidate_edges import extract_candidate_edges
from sna.explainers.pgexplainer import (
    PGExplainer,
    PGExplainerModel,
    build_forward_subgraph,
    expand_pair_weights,
)
from sna.models.gcn import GCN
from sna.pipelines.batch_runner import run_batch_explaining
from sna.pipelines.schema import SCHEMA_V2_COLUMNS

NUM_NODES = 41  # node 40 là node cô lập (m = 0)


def _make_graph():
    gen = torch.Generator().manual_seed(123)
    # Chuỗi dài để có cấu trúc > 3 hop, thêm cạnh ngẫu nhiên tạo hub và chu trình.
    chain = torch.stack([torch.arange(0, 39), torch.arange(1, 40)])
    extra_src = torch.randint(0, 40, (40,), generator=gen)
    extra_dst = torch.randint(0, 40, (40,), generator=gen)
    extra = torch.stack([extra_src, extra_dst])
    extra = extra[:, extra[0] != extra[1]]
    edge_index = to_undirected(torch.cat([chain, extra], dim=1), num_nodes=NUM_NODES)
    x = torch.randn(NUM_NODES, 6, generator=gen)
    train_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    train_mask[:30] = True
    return {'x': x, 'edge_index': edge_index, 'train_mask': train_mask}


def _make_model():
    torch.manual_seed(0)
    model = GCN(6, 16, 3, dropout=0.0).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def graph():
    return _make_graph()


@pytest.fixture(scope="module")
def model():
    return _make_model()


def _mlp_state(pg):
    return [p.detach().clone() for p in pg.mlp.parameters()]


# ----------------------------------------------------------------- F1 ----

def test_f1_mlp_init_ignores_prior_rng_state(model):
    torch.rand(1000)  # trạng thái RNG trước đó khác nhau
    a = _mlp_state(PGExplainerModel(model=model, seed=7))
    torch.rand(37)
    b = _mlp_state(PGExplainerModel(model=model, seed=7))
    c = _mlp_state(PGExplainerModel(model=model, seed=8))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_f1_same_as_runner_ordering(model):
    """run_explainers.py gọi set_global_seed rồi mới tạo model: phải cho cùng trọng số."""
    set_global_seed(5)
    runner_style = _mlp_state(PGExplainerModel(model=model))
    torch.rand(10)
    seeded = _mlp_state(PGExplainerModel(model=model, seed=5))
    assert all(torch.equal(x, y) for x, y in zip(runner_style, seeded))


def _train_and_score(model, graph, seed, node=3, **kw):
    pg = PGExplainerModel(model=model, seed=seed)
    pg.train_on_graph(graph, epochs=3, seed=seed, **kw)
    cand = extract_candidate_edges(graph['edge_index'], node, num_hops=2, num_nodes=NUM_NODES)
    res = PGExplainer(trained_model=pg).explain(model, graph, node, 0, cand, seed=seed)
    assert res.status == "success", res.error_message
    return torch.tensor(res.scores)


def test_f1_end_to_end_reproducible_scores(model, graph):
    torch.rand(99)
    s1 = _train_and_score(model, graph, seed=11)
    torch.rand(5)
    s2 = _train_and_score(model, graph, seed=11)
    s3 = _train_and_score(model, graph, seed=12)
    assert torch.equal(s1, s2)  # max_diff == 0 trên CPU
    assert not torch.equal(s1, s3)


def test_f1_auto_train_path_reproducible(model, graph):
    """PGExplainer().explain tự huấn luyện (đường gọi qua registry.explain_edges)."""
    cand = extract_candidate_edges(graph['edge_index'], 4, num_hops=2, num_nodes=NUM_NODES)
    torch.rand(50)
    r1 = PGExplainer().explain(model, graph, 4, 1, cand, config={'epochs': 2}, seed=3)
    torch.rand(3)
    r2 = PGExplainer().explain(model, graph, 4, 1, cand, config={'epochs': 2}, seed=3)
    assert r1.status == r2.status == "success"
    assert r1.scores == r2.scores


def test_f1_training_restores_deterministic_flag(model, graph):
    before = torch.are_deterministic_algorithms_enabled()
    PGExplainerModel(model=model, seed=0).train_on_graph(graph, epochs=1, seed=0)
    assert torch.are_deterministic_algorithms_enabled() == before


BASHAPES = ROOT / "data" / "bashapes"


@pytest.mark.skipif(not (BASHAPES / "data.pt").exists() or not (BASHAPES / "gcn_ckpt.pt").exists(),
                    reason="Cần data/bashapes/data.pt và gcn_ckpt.pt (bộ giữa kỳ)")
def test_f1_real_bashapes_3hop_reproducible():
    """Trên BA-Shapes thật, batch 3-hop đủ lớn để scatter chạy song song; hai lượt phải trùng nhau."""
    data = torch.load(BASHAPES / "data.pt", weights_only=False, map_location="cpu")
    ck = torch.load(BASHAPES / "gcn_ckpt.pt", weights_only=False, map_location="cpu")
    m = GCN(ck['in_dim'], ck['hidden'], ck['n_classes'], ck.get('dropout', 0.0))
    m.load_state_dict(ck['state_dict'])
    m.eval()
    weights = []
    for _ in range(2):
        pg = PGExplainerModel(model=m, seed=0)
        pg.train_on_graph(data, epochs=3, seed=0)
        weights.append(torch.cat([p.detach().flatten() for p in pg.mlp.parameters()]))
    assert torch.equal(weights[0], weights[1])


# ----------------------------------------------------------------- F9 ----

def _target_logit_full(model, graph, node, weight=None):
    return model(graph['x'], graph['edge_index'], edge_weight=weight)[node]


def _target_logit_sub(model, graph, node, cand, hops, pair_weights=None):
    fwd = build_forward_subgraph(graph['edge_index'], node, cand, hops, NUM_NODES)
    if pair_weights is None:
        pair_weights = torch.ones(cand.num_candidate_edges)
    w = expand_pair_weights(pair_weights, fwd['edge_pair_idx'])
    out = model(graph['x'][fwd['sub_nodes']], fwd['local_edge_index'], edge_weight=w)
    return out[fwd['local_target']], fwd


def test_f9_three_hop_matches_full_graph(model, graph):
    for node in range(40):
        cand = extract_candidate_edges(graph['edge_index'], node, num_hops=2, num_nodes=NUM_NODES)
        sub, _ = _target_logit_sub(model, graph, node, cand, hops=3)
        full = _target_logit_full(model, graph, node)
        assert (sub - full).abs().max().item() < 1e-5, node


def test_f9_three_hop_matches_full_graph_with_mask(model, graph):
    """Mask trên cạnh ứng viên (cạnh khác = 1) cho cùng logit ở đồ thị con và đồ thị đầy đủ."""
    gen = torch.Generator().manual_seed(1)
    ei = graph['edge_index']
    for node in range(0, 40, 3):
        cand = extract_candidate_edges(ei, node, num_hops=2, num_nodes=NUM_NODES)
        pw = torch.rand(cand.num_candidate_edges, generator=gen)
        sub, _ = _target_logit_sub(model, graph, node, cand, hops=3, pair_weights=pw)
        pair_id = {p: i for i, p in enumerate(cand.candidate_pairs)}
        full_w = torch.tensor([
            float(pw[pair_id[(min(u, v), max(u, v))]]) if (min(u, v), max(u, v)) in pair_id else 1.0
            for u, v in ei.t().tolist()
        ])
        full = _target_logit_full(model, graph, node, weight=full_w)
        assert (sub - full).abs().max().item() < 1e-5, node


def test_f9_two_hop_is_inexact(model, graph):
    """Cách giữa kỳ (2-hop) lệch logit ở ít nhất một node: đây là lý do cần F9."""
    worst = 0.0
    for node in range(40):
        cand = extract_candidate_edges(graph['edge_index'], node, num_hops=2, num_nodes=NUM_NODES)
        sub, _ = _target_logit_sub(model, graph, node, cand, hops=2)
        worst = max(worst, (sub - _target_logit_full(model, graph, node)).abs().max().item())
    assert worst > 1e-3


def test_f9_two_hop_maps_every_edge_to_a_candidate(graph):
    """forward_hops=2 dùng đúng tập cạnh của cách làm giữa kỳ (mọi cạnh đều là ứng viên)."""
    cand = extract_candidate_edges(graph['edge_index'], 7, num_hops=2, num_nodes=NUM_NODES)
    fwd = build_forward_subgraph(graph['edge_index'], 7, cand, 2, NUM_NODES)
    assert (fwd['edge_pair_idx'] >= 0).all()
    assert fwd['local_edge_index'].size(1) == cand.subgraph_edge_index.size(1)


def test_f9_rejects_forward_hops_below_candidates(graph):
    cand = extract_candidate_edges(graph['edge_index'], 0, num_hops=2, num_nodes=NUM_NODES)
    with pytest.raises(ValueError):
        build_forward_subgraph(graph['edge_index'], 0, cand, 1, NUM_NODES)


def test_f9_training_runs_with_both_settings(model, graph):
    for hops in (2, 3):
        scores = _train_and_score(model, graph, seed=0, forward_hops=hops)
        assert torch.isfinite(scores).all()


# ----------------------------------------------------------------- F2 ----

def _load_runner_schema():
    spec = importlib.util.spec_from_file_location("run_explainers", ROOT / "scripts" / "run_explainers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCHEMA_V2_COLUMNS


def test_f2_schema_matches_run_explainers():
    assert SCHEMA_V2_COLUMNS == _load_runner_schema()


def test_f2_batch_runner_schema_v2(model, graph, tmp_path):
    ckpt = tmp_path / "gcn_s1000.pt"
    torch.save({'state_dict': model.state_dict()}, ckpt)
    nodes = [3, 10, 40]  # 40 là node cô lập
    df, df_evals = run_batch_explaining(
        model, graph, 'toy', nodes, methods=['saliency', 'pge'], output_dir=tmp_path,
        seed=2, split='test', gnn_seed=1001, checkpoint_path=ckpt,
        pg_config={'epochs': 2},
    )
    assert list(df.columns) == SCHEMA_V2_COLUMNS
    assert df_evals.empty
    assert set(df.method) == {'saliency', 'pgexplainer'}
    assert (df.seed == 2).all() and (df.explainer_seed == 2).all() and (df.gnn_seed == 1001).all()
    assert df.checkpoint_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(df.run_id) == {'toy_gcn_g1001_saliency_e2_test', 'toy_gcn_g1001_pgexplainer_e2_test'}

    ok = df[df.status == 'success']
    assert (ok.u < ok.v).all() and ok.importance_score.notna().all()
    assert (df.runtime_sec >= 0).all()
    empty = df[df.node_id == 40]
    assert set(empty.status) == {'empty_candidates'} and len(empty) == 2

    for (method, node), part in ok.groupby(['method', 'node_id']):
        cand = extract_candidate_edges(graph['edge_index'], node, num_hops=2, num_nodes=NUM_NODES)
        assert list(zip(part.u, part.v)) == cand.candidate_pairs
    assert (tmp_path / "edge_scores.csv").exists()
    assert not (tmp_path / "evaluation.csv").exists()


def test_f2_rejects_wrong_checkpoint_hash(model, graph, tmp_path):
    ckpt = tmp_path / "c.pt"
    torch.save({'a': 1}, ckpt)
    with pytest.raises(ValueError):
        run_batch_explaining(model, graph, 'toy', [3], methods=['saliency'],
                             checkpoint_path=ckpt, checkpoint_sha256='0' * 64)
