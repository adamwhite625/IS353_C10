"""Kiểm thử các phương pháp giải thích cạnh (Saliency, GNNExplainer, PGExplainer, Baselines).
Chạy trên cả hai dataset BA-Shapes và Cora với GPU CUDA (RTX 3050 Laptop).
"""

import sys
import pathlib
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sna


def test_explainers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== KIỂM THỬ EXPLAINERS TRÊN THIẾT BỊ: {device} ===")

    for ds_name in ('bashapes', 'cora'):
        print(f"\n==========================================")
        print(f"--- ĐANG KIỂM THỬ TRÊN DATASET: {ds_name.upper()} ---")
        print(f"==========================================")

        # 1. Nạp dữ liệu và mô hình
        data, pred, val_nodes = sna.load_dataset(ds_name)
        model, _, _ = sna.load_gcn_checkpoint(ds_name)
        model = model.to(device)

        # Chọn node validation đầu tiên
        test_node = val_nodes[0]
        target_class = int(pred[test_node])
        true_label = int(data['y'][test_node])

        print(f"Node kiểm thử: {test_node}")
        print(f"Nhãn thật (y): {true_label} | Dự đoán GCN (pred): {target_class}")

        # 2. Trích xuất tập cạnh ứng viên 2-hop theo BG05, BG07
        cand = sna.extract_candidate_edges(data['edge_index'], test_node, num_hops=2)
        m = cand.num_candidate_edges
        print(f"Số cạnh ứng viên (m): {m}")

        # 3. Saliency
        print("\n[1/4] Đang chạy Saliency...")
        edge_ids, scores_sal, rt_sal, st_sal, err_sal = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': 'saliency'},
            seed=0
        )
        assert st_sal == "success", f"Saliency lỗi: {err_sal}"
        assert len(scores_sal) == m, f"Số điểm ({len(scores_sal)}) khác số cạnh ({m})"
        print(f"  -> Saliency OK | Runtime: {rt_sal:.4f}s | Max score: {max(scores_sal):.4f}")

        # 4. GNNExplainer
        print("\n[2/4] Đang chạy GNNExplainer (50 epochs nhanh)...")
        edge_ids, scores_gnn, rt_gnn, st_gnn, err_gnn = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': 'gnnexplainer', 'epochs': 50, 'lr': 0.02},
            seed=0
        )
        assert st_gnn == "success", f"GNNExplainer lỗi: {err_gnn}"
        assert len(scores_gnn) == m, f"Số điểm ({len(scores_gnn)}) khác số cạnh ({m})"
        print(f"  -> GNNExplainer OK | Runtime: {rt_gnn:.4f}s | Max score: {max(scores_gnn):.4f}")

        # 5. PGExplainer
        print("\n[3/4] Đang huấn luyện & chạy PGExplainer (10 epochs nhanh)...")
        edge_ids, scores_pg, rt_pg, st_pg, err_pg = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': 'pgexplainer', 'epochs': 10, 'lr': 0.01},
            seed=0
        )
        assert st_pg == "success", f"PGExplainer lỗi: {err_pg}"
        assert len(scores_pg) == m, f"Số điểm ({len(scores_pg)}) khác số cạnh ({m})"
        print(f"  -> PGExplainer OK | Runtime: {rt_pg:.4f}s | Max score: {max(scores_pg):.4f}")

        # 6. Baselines (Random & Nearest-First)
        print("\n[4/4] Đang kiểm tra Baselines...")
        for b_name in ('random', 'nearest_first'):
            _, b_scores, b_rt, b_st, _ = sna.explain_edges(
                model=model,
                data=data,
                node_id=test_node,
                target_class=target_class,
                candidate_edges=cand,
                config={'method': b_name},
                seed=0
            )
            assert b_st == "success"
            print(f"  -> Baseline {b_name:15s} OK | Runtime: {b_rt:.4f}s")

    # 7. Kiểm thử trường hợp biên: Node có đúng 1 cạnh ứng viên (Degree-1 / Leaf node)
    print("\n[5/5] Đang kiểm tra trường hợp biên (Node có 1 cạnh ứng viên - Cora node 3)...")
    cora_data, cora_pred, _ = sna.load_dataset('cora')
    cora_model, _, _ = sna.load_gcn_checkpoint('cora')
    cora_model = cora_model.to(device)
    cand_deg1 = sna.extract_candidate_edges(cora_data['edge_index'], 3, num_hops=2)
    assert cand_deg1.num_candidate_edges == 1
    edge_ids, scores_deg1, _, st_deg1, err_deg1 = sna.explain_edges(
        model=cora_model,
        data=cora_data,
        node_id=3,
        target_class=int(cora_pred[3]),
        candidate_edges=cand_deg1,
        config={'method': 'pgexplainer', 'epochs': 5},
        seed=0
    )
    assert st_deg1 == "success", f"Degree-1 node failed: {err_deg1}"
    assert len(scores_deg1) == 1, f"Expected 1 score, got {len(scores_deg1)}"
    print(f"  -> Degree-1 Node OK | Score: {scores_deg1[0]:.4f}")

    # 8. Kiểm thử mở rộng (Reviewer 2 & 3 Adversarial Edge Cases):
    print("\n[6/6] Đang kiểm tra các trường hợp biên bổ sung (Adversarial Robustness)...")
    import numpy as np
    from sna.explainers.pgexplainer import PGExplainerMLP, PGExplainerModel
    mlp_test = PGExplainerMLP(emb_dim=16)

    # a) Target vector broadcasting: (N, D) + (N, D) + (D,)
    u_2d = torch.randn(5, 16)
    v_2d = torch.randn(5, 16)
    t_1d = torch.randn(16)
    out_broad = mlp_test(u_2d, v_2d, t_1d)
    assert out_broad.shape == torch.Size([5]), f"Expected shape [5], got {out_broad.shape}"

    # b) Target singleton 2D broadcasting: (N, D) + (N, D) + (1, D)
    t_single = torch.randn(1, 16)
    out_single = mlp_test(u_2d, v_2d, t_single)
    assert out_single.shape == torch.Size([5]), f"Expected shape [5], got {out_single.shape}"

    # c) Reviewer 3: Asymmetric broadcasting (N, D) + (1, D) or (D,) and (D,) + (N, D)
    v_1d = torch.randn(16)
    v_single = torch.randn(1, 16)
    u_1d = torch.randn(16)
    u_single = torch.randn(1, 16)
    out_asym1 = mlp_test(u_2d, v_1d, t_1d)
    assert out_asym1.shape == torch.Size([5]), f"Expected [5], got {out_asym1.shape}"
    out_asym2 = mlp_test(u_2d, v_single, t_1d)
    assert out_asym2.shape == torch.Size([5]), f"Expected [5], got {out_asym2.shape}"
    out_asym3 = mlp_test(u_1d, v_2d, t_1d)
    assert out_asym3.shape == torch.Size([5]), f"Expected [5], got {out_asym3.shape}"
    out_asym4 = mlp_test(u_single, v_2d, t_1d)
    assert out_asym4.shape == torch.Size([5]), f"Expected [5], got {out_asym4.shape}"
    out_asym5 = mlp_test(u_single, v_1d, t_1d)
    assert out_asym5.shape == torch.Size([1]), f"Expected [1], got {out_asym5.shape}"

    # d) Reviewer 3: Pre-concatenated 1D and 2D single / batch inputs
    cat_1d = torch.randn(48)
    assert mlp_test(cat_1d).shape == torch.Size([1])
    cat_empty = torch.randn(0, 48)
    assert mlp_test(cat_empty).shape == torch.Size([0])

    # e) Auto-detection của emb_dim trong PGExplainerModel (c1 & conv1)
    auto_pge = PGExplainerModel(model=cora_model)
    assert auto_pge.emb_dim == cora_model.c1.out_channels, f"Expected {cora_model.c1.out_channels}, got {auto_pge.emb_dim}"

    class CustomConv1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Linear(16, 32)
            self.conv1.out_channels = 32
        def forward(self, x, edge_index):
            return self.conv1(x)

    custom_model = CustomConv1Model()
    conv1_pge = PGExplainerModel(model=custom_model)
    assert conv1_pge.emb_dim == 32

    # f) Train mask dạng PyTorch tensor chỉ số nguyên (Integer index tensor)
    synth_data = {
        'x': torch.randn(10, 16),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        'train_mask': torch.tensor([0, 2], dtype=torch.long)
    }
    synth_model = sna.models.gcn.GCN(16, 16, 2)
    pge_synth = PGExplainerModel(synth_model)
    synth_time = pge_synth.train_on_graph(synth_data, epochs=2)
    assert pge_synth.is_trained
    assert synth_time > 0

    # g) Reviewer 3: Train mask dạng NumPy integer array (ngăn ngừa lỗi index corruption [1, 2] thay vì [0, 2])
    synth_data_np = {
        'x': torch.randn(10, 16),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        'train_mask': np.array([0, 2])
    }
    pge_synth_np = PGExplainerModel(synth_model)
    synth_time_np = pge_synth_np.train_on_graph(synth_data_np, epochs=2)
    assert pge_synth_np.is_trained

    # h) Reviewer 3: Đồ thị có Node cô lập vượt quá index max của edge_index (ngăn ngừa PyG k_hop_subgraph IndexError)
    edge_index_small = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    isolated_cand = sna.extract_candidate_edges(edge_index_small, node_id=5)
    assert isolated_cand.num_candidate_edges == 0
    assert isolated_cand.subgraph_nodes.tolist() == [5]
    assert isolated_cand.edge_to_pair_idx.device == edge_index_small.device

    # i) Đồ thị rỗng không có cạnh (Empty edge_index)
    empty_cand = sna.extract_candidate_edges(torch.empty((2, 0), dtype=torch.long), node_id=0)
    assert empty_cand.num_candidate_edges == 0
    print("  -> Tất cả các trường hợp biên bổ sung (Asym Broadcasting, Pre-concatenated, conv1, NumPy train_mask, Isolated node bounds, Empty graph) ĐỀU ĐẠT!")

    print("\n" + "=" * 50)
    print(">>> TẤT CẢ CÁC BÀI KIỂM THỬ EXPLAINERS ĐỀU THÀNH CÔNG! <<<")
    print("=" * 50)


def test_f8_generic_model_without_c1_c2():
    """F8: GNNExplainer and Saliency must call model(x, edge_index, edge_weight=w) without c1/c2."""
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GenericThreeLayerGNN(nn.Module):
        def __init__(self, in_dim: int, hid: int, out_dim: int):
            super().__init__()
            self.blocks = nn.ModuleList([
                GCNConv(in_dim, hid),
                GCNConv(hid, hid),
                GCNConv(hid, out_dim),
            ])

        def forward(self, x, edge_index, edge_weight=None):
            h = F.relu(self.blocks[0](x, edge_index, edge_weight=edge_weight))
            h = F.relu(self.blocks[1](h, edge_index, edge_weight=edge_weight))
            return self.blocks[2](h, edge_index, edge_weight=edge_weight)

    torch.manual_seed(0)
    model = GenericThreeLayerGNN(4, 8, 3).eval()
    assert not hasattr(model, "c1") and not hasattr(model, "c2")

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    data = {"x": torch.randn(4, 4), "edge_index": edge_index}
    cand = sna.extract_candidate_edges(edge_index, node_id=0, num_hops=2, num_nodes=4)
    assert cand.num_candidate_edges == 4

    with torch.no_grad():
        target = int(model(data["x"], edge_index)[0].argmax().item())

    for cfg in (
        {"method": "saliency"},
        {"method": "gnnexplainer", "epochs": 20, "lr": 0.01},
        {"method": "gnnexplainer", "epochs": 20, "lr": 0.01, "size_loss_mode": "sum"},
    ):
        eids, scores, rt, status, err = sna.explain_edges(
            model=model,
            data=data,
            node_id=0,
            target_class=target,
            candidate_edges=cand,
            config=cfg,
            seed=42,
        )
        assert status == "success", f"{cfg} failed on generic model: {err}"
        assert len(eids) == cand.num_candidate_edges
        assert len(scores) == cand.num_candidate_edges
        assert all(torch.isfinite(torch.tensor(scores)).tolist())
        assert rt >= 0.0


def test_f8_empty_candidates_returns_empty_candidates():
    """F8: Both GNNExplainer and Saliency must return status='empty_candidates' when m == 0."""
    torch.manual_seed(0)
    model = sna.models.gcn.GCN(4, 8, 2).eval()
    data_iso = {
        "x": torch.randn(6, 4),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    }
    cand_iso = sna.extract_candidate_edges(data_iso["edge_index"], node_id=5, num_hops=2, num_nodes=6)
    assert cand_iso.num_candidate_edges == 0

    data_empty = {
        "x": torch.randn(3, 4),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    cand_empty = sna.extract_candidate_edges(data_empty["edge_index"], node_id=0, num_hops=2, num_nodes=3)
    assert cand_empty.num_candidate_edges == 0

    for data_case, cand_case, nid in ((data_iso, cand_iso, 5), (data_empty, cand_empty, 0)):
        for method in ("saliency", "gnnexplainer"):
            eids, scores, rt, status, err = sna.explain_edges(
                model=model,
                data=data_case,
                node_id=nid,
                target_class=0,
                candidate_edges=cand_case,
                config={"method": method, "epochs": 10},
                seed=0,
            )
            assert status == "empty_candidates", f"{method} expected empty_candidates, got {status} ({err})"
            assert eids == []
            assert scores == []
            assert err == ""
            assert rt >= 0.0


def test_f8_determinism_and_model_grad_hygiene():
    """F8: Saliency is seed-invariant and does not pollute unfrozen model.grad; GNNExplainer is deterministic per seed."""
    torch.manual_seed(0)
    model = sna.models.gcn.GCN(4, 8, 3).eval()  # intentionally unfrozen params
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    data = {"x": torch.randn(5, 4), "edge_index": edge_index}
    cand = sna.extract_candidate_edges(edge_index, node_id=0, num_hops=2, num_nodes=5)

    _, sal0, _, st0, _ = sna.explain_edges(model, data, 0, 1, cand, {"method": "saliency"}, seed=0)
    _, sal1, _, st1, _ = sna.explain_edges(model, data, 0, 1, cand, {"method": "saliency"}, seed=1)
    assert st0 == "success" and st1 == "success"
    assert sal0 == sal1
    assert all(p.grad is None for p in model.parameters())

    _, gnn_a, _, st_a, _ = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 25, "lr": 0.01}, seed=7
    )
    _, gnn_b, _, st_b, _ = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 25, "lr": 0.01}, seed=7
    )
    _, gnn_c, _, st_c, _ = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 25, "lr": 0.01}, seed=8
    )
    assert st_a == "success" and st_b == "success" and st_c == "success"
    assert max(abs(a - b) for a, b in zip(gnn_a, gnn_b)) == 0.0
    assert max(abs(a - c) for a, c in zip(gnn_a, gnn_c)) > 0.0
    assert all(p.grad is None for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())

    # Invalid size_loss_mode must fail gracefully
    _, _, _, st_bad, err_bad = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "size_loss_mode": "invalid"}, seed=0
    )
    assert st_bad == "failed"
    assert "size_loss_mode" in err_bad

    # Negative epochs must fail gracefully
    _, _, _, st_neg, err_neg = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": -5}, seed=0
    )
    assert st_neg == "failed"
    assert "epochs" in err_neg

    # Invalid lr / coeff_size / coeff_ent must fail gracefully
    for bad_cfg, key in (
        ({"method": "gnnexplainer", "lr": 0.0}, "lr"),
        ({"method": "gnnexplainer", "coeff_size": -0.1}, "coeff_size"),
        ({"method": "gnnexplainer", "coeff_ent": -0.1}, "coeff_ent"),
    ):
        _, _, _, st_inv, err_inv = sna.explain_edges(model, data, 0, 1, cand, bad_cfg, seed=0)
        assert st_inv == "failed"
        assert key in err_inv


def test_f8_works_inside_torch_no_grad_context():
    """F8: Both Saliency and GNNExplainer must succeed with identical scores even inside torch.no_grad()."""
    torch.manual_seed(0)
    model = sna.models.gcn.GCN(4, 8, 3).eval()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    data = {"x": torch.randn(5, 4), "edge_index": edge_index}
    cand = sna.extract_candidate_edges(edge_index, node_id=0, num_hops=2, num_nodes=5)

    _, sal_ref, _, st_sr, _ = sna.explain_edges(model, data, 0, 1, cand, {"method": "saliency"}, seed=0)
    _, gnn_ref, _, st_gr, _ = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 15, "lr": 0.01}, seed=7
    )
    assert st_sr == "success" and st_gr == "success"

    with torch.no_grad():
        _, sal_ng, _, st_sng, err_sng = sna.explain_edges(model, data, 0, 1, cand, {"method": "saliency"}, seed=0)
        _, gnn_ng, _, st_gng, err_gng = sna.explain_edges(
            model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 15, "lr": 0.01}, seed=7
        )
    assert st_sng == "success", f"Saliency failed under torch.no_grad(): {err_sng}"
    assert st_gng == "success", f"GNNExplainer failed under torch.no_grad(): {err_gng}"
    assert sal_ng == sal_ref
    assert max(abs(a - b) for a, b in zip(gnn_ng, gnn_ref)) == 0.0
    assert all(p.grad is None for p in model.parameters())


def test_def_b4_explainer_input_validation_and_nonfinite_guard():
    """DEF-B4: Reject bool/NaN/Inf hyperparams, out-of-range target_class (incl. epochs=0), and NaN inputs/scores."""
    torch.manual_seed(0)
    model = sna.models.gcn.GCN(4, 8, 3).eval()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    data = {"x": torch.randn(5, 4), "edge_index": edge_index}
    cand = sna.extract_candidate_edges(edge_index, node_id=0, num_hops=2, num_nodes=5)

    # 1. GNNExplainer rejects bool/float epochs and bool/NaN/Inf lr, coeff_size, coeff_ent
    for bad_cfg, key in (
        ({"method": "gnnexplainer", "epochs": True}, "epochs"),
        ({"method": "gnnexplainer", "epochs": False}, "epochs"),
        ({"method": "gnnexplainer", "epochs": 1.5}, "epochs"),
        ({"method": "gnnexplainer", "lr": True}, "lr"),
        ({"method": "gnnexplainer", "lr": float("nan")}, "lr"),
        ({"method": "gnnexplainer", "lr": float("inf")}, "lr"),
        ({"method": "gnnexplainer", "coeff_size": True}, "coeff_size"),
        ({"method": "gnnexplainer", "coeff_size": float("nan")}, "coeff_size"),
        ({"method": "gnnexplainer", "coeff_size": float("inf")}, "coeff_size"),
        ({"method": "gnnexplainer", "coeff_ent": False}, "coeff_ent"),
        ({"method": "gnnexplainer", "coeff_ent": float("nan")}, "coeff_ent"),
        ({"method": "gnnexplainer", "coeff_ent": float("inf")}, "coeff_ent"),
    ):
        _, _, _, st, err = sna.explain_edges(model, data, 0, 1, cand, bad_cfg, seed=0)
        assert st == "failed", f"Expected failed for {bad_cfg}, got {st}"
        assert key in err, f"Expected '{key}' in error message, got {err!r}"

    # 2. Out-of-range target_class must fail on Saliency and on GNNExplainer (both epochs=0 and epochs=3)
    for cfg in (
        {"method": "saliency"},
        {"method": "gnnexplainer", "epochs": 0},
        {"method": "gnnexplainer", "epochs": 3},
    ):
        _, _, _, st_oob, err_oob = sna.explain_edges(model, data, 0, 99, cand, cfg, seed=0)
        assert st_oob == "failed", f"Expected failed for out-of-range target_class in {cfg}, got {st_oob}"
        assert "target_class" in err_oob

    # Valid epochs=0 with valid target_class still succeeds
    _, scores_e0, _, st_e0, err_e0 = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 0}, seed=0
    )
    assert st_e0 == "success", err_e0
    assert len(scores_e0) == cand.num_candidate_edges

    # 3. NaN in data['x'] must produce status='failed' (never status='success' with NaN scores)
    x_nan = data["x"].clone()
    x_nan[0, 0] = float("nan")
    data_nan = {"x": x_nan, "edge_index": edge_index}
    for cfg in (
        {"method": "saliency"},
        {"method": "gnnexplainer", "epochs": 0},
        {"method": "gnnexplainer", "epochs": 3},
    ):
        _, sc_nan, _, st_nan, _ = sna.explain_edges(model, data_nan, 0, 1, cand, cfg, seed=0)
        assert st_nan == "failed", f"Expected failed on NaN input for {cfg}, got {st_nan} ({sc_nan})"

    # 4. Asymmetric NaN edge gradient in Saliency (finite uv, NaN vu) must NOT be hidden by Python max()
    class AsymNaNGradModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.ones(1))

        def forward(self, x, edge_index, edge_weight=None):
            # Make edge 0 finite and edge 1 produce NaN gradient via 0 * inf in backward
            w = edge_weight
            bad_term = torch.where(w[1] > 0, w[1] * float("nan"), w[1])
            logits = torch.zeros((x.size(0), 2), device=x.device)
            logits[0, 0] = w[0] * 0.5 + bad_term
            return logits

    asym_model = AsymNaNGradModel().eval()
    _, _, _, st_asym, _ = sna.explain_edges(
        asym_model, data, 0, 0, cand, {"method": "saliency"}, seed=0
    )
    assert st_asym == "failed"

    # 5. GNNExplainer must fail when out[node_id] has NaN independent of edge_weight (epochs>0)
    #    or NaN in a non-target class logit (epochs==0), and must reject bool/np.bool_ seed
    import numpy as np

    class DetachedNaNOutModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_weight=None):
            out = torch.zeros((x.size(0), 2), device=x.device)
            out[0, 0] = float("nan")
            return out

    class OtherClassNaNModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_weight=None):
            out = torch.zeros((x.size(0), 2), device=x.device)
            out[0, 1] = float("nan")
            return out

    _, _, _, st_det, _ = sna.explain_edges(
        DetachedNaNOutModel().eval(), data, 0, 0, cand, {"method": "gnnexplainer", "epochs": 3}, seed=0
    )
    assert st_det == "failed"
    _, _, _, st_oth, _ = sna.explain_edges(
        OtherClassNaNModel().eval(), data, 0, 0, cand, {"method": "gnnexplainer", "epochs": 0}, seed=0
    )
    assert st_oth == "failed"
    for bad_seed in (True, np.bool_(True)):
        _, _, _, st_bs, err_bs = sna.explain_edges(
            model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 2}, seed=bad_seed
        )
        assert st_bs == "failed" and "seed" in err_bs


def test_def_b13_gnnexplainer_accepts_numpy_int_epochs_and_ref_audit_note_dynamic():
    """DEF-B13 polish: GNNExplainer accepts np.int64 epochs (rejects np.bool_); check_gnnexplainer_ref note uses len(ba_val)."""
    import numpy as np

    torch.manual_seed(0)
    model = sna.models.gcn.GCN(4, 8, 3).eval()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    data = {"x": torch.randn(5, 4), "edge_index": edge_index}
    cand = sna.extract_candidate_edges(edge_index, node_id=0, num_hops=2, num_nodes=5)

    # np.int64 epochs must succeed identically to Python int epochs
    _, sc_py, _, st_py, err_py = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": 3}, seed=np.int64(7)
    )
    _, sc_np, _, st_np, err_np = sna.explain_edges(
        model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": np.int64(3)}, seed=np.int64(7)
    )
    assert st_py == "success", err_py
    assert st_np == "success", err_np
    assert sc_py == sc_np

    # np.bool_ and negative np.int64 epochs must still fail with 'epochs' in message
    for bad_ep in (np.bool_(True), np.bool_(False), np.int64(-1)):
        _, _, _, st_bad, err_bad = sna.explain_edges(
            model, data, 0, 1, cand, {"method": "gnnexplainer", "epochs": bad_ep}, seed=0
        )
        assert st_bad == "failed" and "epochs" in err_bad

    # check_gnnexplainer_ref.py dynamic note string check (fast n_nodes=2, top_k=3 on bashapes)
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import check_gnnexplainer_ref

    audit = check_gnnexplainer_ref.audit_set_masks_vs_edge_weight(
        ROOT / "data", ["bashapes"], n_nodes=2, top_k=3
    )
    note = audit["gat_synthetic_check"]["note"]
    assert "2 dev nodes" in note and "first 3 candidate pairs" in note


if __name__ == '__main__':
    test_explainers()
    test_f8_generic_model_without_c1_c2()
    test_f8_empty_candidates_returns_empty_candidates()
    test_f8_determinism_and_model_grad_hygiene()
    test_f8_works_inside_torch_no_grad_context()
    test_def_b4_explainer_input_validation_and_nonfinite_guard()
    test_def_b13_gnnexplainer_accepts_numpy_int_epochs_and_ref_audit_note_dynamic()
