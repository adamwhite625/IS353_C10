"""Unit and end-to-end smoke tests for scripts/run_explainers.py + check_scores.py (PN2)."""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import check_scores
import run_explainers
from sna.models.gcn import GCN


def test_run_explainers_end_to_end_smoke_and_check_scores_pass(tmp_path):
    """Run Saliency, GNNExplainer, PGExplainer via run_explainers.py on synthetic data and verify PASS in check_scores.py."""
    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(0)
    # 6 nodes: 0..4 connected ring + node 5 isolated (tests empty_candidates end-to-end!)
    x = torch.randn(6, 4)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
        ],
        dtype=torch.long,
    )
    train_mask = torch.tensor([True, True, True, False, False, False])
    val_mask = torch.tensor([False, False, False, True, False, False])
    test_mask = torch.tensor([False, False, False, False, True, True])
    torch.save(
        {
            "x": x,
            "edge_index": edge_index,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        },
        ds_dir / "data.pt",
    )
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [4, 5]}), encoding="utf-8")

    model = GCN(in_dim=4, hidden_dim=8, out_dim=2, dropout=0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)

    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds_dir = ckpt_dir / "bashapes"
    ckpt_ds_dir.mkdir(parents=True)
    ckpt_pt = ckpt_ds_dir / "gcn_s1000.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": 4,
            "hidden": 8,
            "n_classes": 2,
            "dropout": 0.0,
            "pred": pred,
        },
        ckpt_pt,
    )
    ckpt_sha = run_explainers.sha256_file(ckpt_pt)
    (ckpt_ds_dir / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": ckpt_sha}), encoding="utf-8"
    )

    out_dir = tmp_path / "results" / "scores"

    # Invocation 1: Khoi runs saliency (protocol-saliency -> e=0,1 on g=1000) & gnnexplainer
    rc1 = run_explainers.main(
        [
            "--dataset", "bashapes",
            "--split", "test",
            "--gnn-seeds", "1000",
            "--explainer-seeds", "0", "1",
            "--methods", "saliency", "gnnexplainer",
            "--data-dir", str(data_dir),
            "--checkpoints-dir", str(ckpt_dir),
            "--out-dir", str(out_dir),
            "--device", "cpu",
            "--protocol-saliency",
            "--gnn-epochs", "15",
        ]
    )
    assert rc1 == 0

    # Invocation 2: Thien runs pgexplainer into same seed directory -> run_log.json merges cleanly
    rc2 = run_explainers.main(
        [
            "--dataset", "bashapes",
            "--split", "test",
            "--gnn-seeds", "1000",
            "--explainer-seeds", "0",
            "--methods", "pgexplainer",
            "--data-dir", str(data_dir),
            "--checkpoints-dir", str(ckpt_dir),
            "--out-dir", str(out_dir),
            "--device", "cpu",
            "--pg-epochs", "3",
        ]
    )
    assert rc2 == 0

    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    assert (seed_dir / "saliency_e0.csv").exists()
    assert (seed_dir / "saliency_e1.csv").exists()
    assert (seed_dir / "gnnexplainer_e0.csv").exists()
    assert (seed_dir / "gnnexplainer_e1.csv").exists()
    assert (seed_dir / "pgexplainer_e0.csv").exists()
    assert (seed_dir / "run_log.json").exists()

    log_data = json.loads((seed_dir / "run_log.json").read_text(encoding="utf-8"))
    assert len(log_data["runs_by_id"]) == 5
    assert log_data["checkpoint_sha256"] == ckpt_sha
    assert log_data["total_failed_nodes"] == 0

    # Verify node 5 got empty_candidates across all methods
    df_gnn = pd.read_csv(seed_dir / "gnnexplainer_e0.csv")
    assert list(df_gnn.columns) == run_explainers.SCHEMA_V2_COLUMNS
    assert df_gnn.loc[df_gnn["node_id"] == 5, "status"].tolist() == ["empty_candidates"]
    assert (df_gnn.loc[df_gnn["node_id"] == 4, "status"] == "success").all()

    # Validate entire directory with check_scores.py -> MUST PASS!
    report = check_scores.validate_scores_dir(
        scores_path=out_dir / "bashapes",
        checkpoints_dir=ckpt_dir,
        data_dir=data_dir,
    )
    assert report["status"] == "PASS", report["errors"]
    assert report["saliency_determinism"]["bashapes/gcn_s1000"]["max_diff"] == 0.0
    assert report["n_files"] == 5


def test_run_explainers_pgexplainer_reproducibility_via_runner(tmp_path):
    """Verify run_explainers.py seeds before creating PGExplainerModel (defensive F1 parity)."""
    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(1)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    train_mask = torch.tensor([True, True, True, False, False])
    torch.save({"x": x, "edge_index": edge_index, "train_mask": train_mask}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [3, 4]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_pt = tmp_path / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        ckpt_pt,
    )

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    for out_dir in (out_a, out_b):
        rc = run_explainers.main(
            [
                "--dataset", "bashapes",
                "--split", "test",
                "--gnn-seeds", "1000",
                "--explainer-seeds", "0",
                "--methods", "pgexplainer",
                "--data-dir", str(data_dir),
                "--checkpoint-path", str(ckpt_pt),
                "--out-dir", str(out_dir),
                "--device", "cpu",
                "--pg-epochs", "4",
            ]
        )
        assert rc == 0

    df_a = pd.read_csv(out_a / "bashapes" / "gcn_s1000" / "pgexplainer_e0.csv")
    df_b = pd.read_csv(out_b / "bashapes" / "gcn_s1000" / "pgexplainer_e0.csv")
    assert (df_a["importance_score"].to_numpy() == df_b["importance_score"].to_numpy()).all()


def test_run_explainers_resume_recovers_missing_run_log(tmp_path):
    """Verify --resume skips completed CSVs and restores run_log.json metadata cleanly."""
    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(2)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    ckpt_pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        ckpt_pt,
    )
    (ckpt_ds / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": run_explainers.sha256_file(ckpt_pt)}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "results" / "scores"
    rc1 = run_explainers.main(
        [
            "--dataset", "bashapes",
            "--split", "test",
            "--gnn-seeds", "1000",
            "--explainer-seeds", "0",
            "--methods", "saliency",
            "--data-dir", str(data_dir),
            "--checkpoints-dir", str(ckpt_dir),
            "--out-dir", str(out_dir),
            "--device", "cpu",
        ]
    )
    assert rc1 == 0
    # Simulate interrupted run where run_log.json was missing before resume
    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    sal_csv = seed_dir / "saliency_e0.csv"
    orig_bytes = sal_csv.read_bytes()
    (seed_dir / "run_log.json").unlink()

    rc2 = run_explainers.main(
        [
            "--dataset", "bashapes",
            "--split", "test",
            "--gnn-seeds", "1000",
            "--explainer-seeds", "0",
            "--methods", "saliency", "gnnexplainer",
            "--data-dir", str(data_dir),
            "--checkpoints-dir", str(ckpt_dir),
            "--out-dir", str(out_dir),
            "--device", "cpu",
            "--gnn-epochs", "10",
            "--resume",
        ]
    )
    assert rc2 == 0
    assert sal_csv.read_bytes() == orig_bytes
    log_obj = json.loads((seed_dir / "run_log.json").read_text(encoding="utf-8"))
    assert "bashapes_gcn_g1000_saliency_e0_test" in log_obj["runs_by_id"]
    assert "bashapes_gcn_g1000_gnnexplainer_e0_test" in log_obj["runs_by_id"]
    assert log_obj["runs_by_id"]["bashapes_gcn_g1000_saliency_e0_test"]["resumed_from_existing_csv"] is True

    report = check_scores.validate_scores_dir(
        scores_path=out_dir / "bashapes",
        checkpoints_dir=ckpt_dir,
        data_dir=data_dir,
    )
    assert report["status"] == "PASS", report["errors"]


def test_run_explainers_resume_rejects_hyperparam_drift_and_preserves_merged_hyperparams(tmp_path):
    """--resume must reject size_loss_mode/context drift; merging Thien's PG run must not clobber Khoi's GNN hyperparams."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(3)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    train_mask = torch.tensor([True, True, True, False, False])
    test_mask = torch.tensor([False, False, False, True, True])
    torch.save(
        {"x": x, "edge_index": edge_index, "train_mask": train_mask, "test_mask": test_mask},
        ds_dir / "data.pt",
    )
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [3, 4]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    for g in (1000, 1001):
        pt = ckpt_ds / f"gcn_s{g}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
            pt,
        )
        (ckpt_ds / f"gcn_s{g}.json").write_text(
            json.dumps({"gnn_seed": g, "sha256": run_explainers.sha256_file(pt)}), encoding="utf-8"
        )

    out_dir = tmp_path / "results" / "scores"

    # 1. Khoi runs gnnexplainer e=0 with size_loss_mode=sum
    rc1 = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "gnnexplainer",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        "--gnn-epochs", "6", "--gnn-size-loss-mode", "sum",
    ])
    assert rc1 == 0

    # 2. Attempting --resume with size_loss_mode=mean MUST fail loudly (no silent mixing!)
    with pytest.raises(ValueError, match="hyperparams"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "0", "1",
            "--methods", "gnnexplainer",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            "--gnn-epochs", "6", "--gnn-size-loss-mode", "mean",
            "--resume",
        ])

    # 3. Attempting --resume with different --max-nodes MUST fail loudly
    with pytest.raises(ValueError, match="Cannot --resume"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "0",
            "--methods", "gnnexplainer", "--max-nodes", "1",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            "--gnn-epochs", "6", "--gnn-size-loss-mode", "sum",
            "--resume",
        ])

    # 4. Thien merges pgexplainer (with default --gnn-size-loss-mode=mean!) -> Khoi's sum hyperparams MUST stay intact
    rc_pg = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "pgexplainer",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        "--pg-epochs", "2",
    ])
    assert rc_pg == 0
    log_1000 = json.loads((out_dir / "bashapes" / "gcn_s1000" / "run_log.json").read_text(encoding="utf-8"))
    assert log_1000["hyperparams"]["gnnexplainer"]["size_loss_mode"] == "sum"
    assert log_1000["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["hyperparams"]["size_loss_mode"] == "sum"
    assert log_1000["runs_by_id"]["bashapes_gcn_g1000_pgexplainer_e0_test"]["hyperparams"]["epochs"] == 2

    # 5. Step 2 (--gnn-seeds 1001 --protocol-saliency) must run ONLY saliency e=0 (not e=1)
    rc_s2 = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1001", "--explainer-seeds", "0", "1",
        "--methods", "saliency", "--protocol-saliency",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ])
    assert rc_s2 == 0
    s2_csvs = sorted(p.name for p in (out_dir / "bashapes" / "gcn_s1001").glob("*.csv"))
    assert s2_csvs == ["saliency_e0.csv"]

    # Everything validates PASS in check_scores.py (including csv_sha256 + split_mask_leakage checks)
    rep = check_scores.validate_scores_dir(
        scores_path=out_dir / "bashapes",
        checkpoints_dir=ckpt_dir,
        data_dir=data_dir,
    )
    assert rep["status"] == "PASS", rep["errors"]


def test_def_b2_run_explainers_blocks_mixed_hyperparams_with_or_without_resume_and_no_default_pollution(tmp_path):
    """DEF-B2: Block mixing sum/mean hyperparams on retained runs (with or without --resume) & do not pollute merged_hyperparams."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(11)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    for g in (1000, 1001):
        pt = ckpt_ds / f"gcn_s{g}.pt"
        # Also verify DEF-B4: saved ck['pred'] as int32 is accepted cleanly!
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": 4,
                "hidden": 8,
                "n_classes": 2,
                "pred": pred.to(dtype=torch.int32),
            },
            pt,
        )
        (ckpt_ds / f"gcn_s{g}.json").write_text(
            json.dumps({"gnn_seed": g, "sha256": run_explainers.sha256_file(pt)}), encoding="utf-8"
        )

    out_dir = tmp_path / "results" / "scores"

    # 1. Saliency-only invocation on fresh g=1001 must NOT populate default gnnexplainer/pgexplainer hyperparams
    rc_sal = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1001", "--explainer-seeds", "0",
        "--methods", "saliency",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ])
    assert rc_sal == 0
    log_1001 = json.loads((out_dir / "bashapes" / "gcn_s1001" / "run_log.json").read_text(encoding="utf-8"))
    assert log_1001["hyperparams"] == {}

    # 2. Run gnnexplainer e=0 with size_loss_mode=sum on g=1000
    rc_sum = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "gnnexplainer",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        "--gnn-epochs", "4", "--gnn-size-loss-mode", "sum",
    ])
    assert rc_sum == 0

    # 3. Adding e=1 with size_loss_mode=mean WITHOUT --resume must fail (retains e=0 with sum!)
    with pytest.raises(ValueError, match="Cannot mix gnnexplainer hyperparams"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "1",
            "--methods", "gnnexplainer",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            "--gnn-epochs", "4", "--gnn-size-loss-mode", "mean",
        ])
    assert not (out_dir / "bashapes" / "gcn_s1000" / "gnnexplainer_e1.csv").exists()

    # 4. Adding e=1 with size_loss_mode=mean WITH --resume must ALSO fail (retains e=0 with sum!)
    with pytest.raises(ValueError, match="Cannot mix gnnexplainer hyperparams"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "1",
            "--methods", "gnnexplainer",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            "--gnn-epochs", "4", "--gnn-size-loss-mode", "mean",
            "--resume",
        ])

    # 5. Retained run with hyperparams=None for gnnexplainer must fail with or without --resume
    log_path_1000 = out_dir / "bashapes" / "gcn_s1000" / "run_log.json"
    log_obj = json.loads(log_path_1000.read_text(encoding="utf-8"))
    log_obj["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["hyperparams"] = None
    log_path_1000.write_text(json.dumps(log_obj), encoding="utf-8")
    for extra_flags in ([], ["--resume"]):
        with pytest.raises(ValueError, match="Cannot mix gnnexplainer hyperparams"):
            run_explainers.main([
                "--dataset", "bashapes", "--split", "test",
                "--gnn-seeds", "1000", "--explainer-seeds", "1",
                "--methods", "gnnexplainer",
                "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
                "--gnn-epochs", "4", "--gnn-size-loss-mode", "sum",
                *extra_flags,
            ])

    # 6. Full overwrite of e=0 WITHOUT --resume (no retained old gnnexplainer runs) succeeds cleanly
    rc_overwrite = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "gnnexplainer",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        "--gnn-epochs", "4", "--gnn-size-loss-mode", "mean",
    ])
    assert rc_overwrite == 0
    rep = check_scores.validate_scores_dir(
        scores_path=out_dir / "bashapes", checkpoints_dir=ckpt_dir, data_dir=data_dir
    )
    assert rep["status"] == "PASS", rep["errors"]


def test_def_b4_run_explainers_cli_nonfinite_flags_and_corrupt_explainer_result_guard(tmp_path, monkeypatch):
    """DEF-B4: Reject NaN/Inf/inverted-temp CLI flags and convert NaN/length-mismatched explainer outputs to status='failed'."""
    import pytest
    from sna.core.types import ExplainerResult
    from sna.data.candidate_edges import extract_candidate_edges

    # 1. CLI float flags must reject NaN, Inf, and pg_temp_min > pg_temp_init
    base = ["--dataset", "bashapes", "--gnn-seeds", "1000"]
    for bad_args in (
        ["--gnn-lr", "nan"],
        ["--gnn-lr", "inf"],
        ["--gnn-coeff-size", "nan"],
        ["--gnn-coeff-ent", "inf"],
        ["--pg-lr", "nan"],
        ["--pg-temp-init", "inf"],
        ["--pg-temp-min", "nan"],
        ["--pg-temp-init", "1.0", "--pg-temp-min", "2.0"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            run_explainers.main(base + bad_args)
        assert exc_info.value.code != 0

    # 2. run_single_configuration must turn NaN scores or length mismatches into status='failed'
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    data = {"x": x, "edge_index": edge_index}
    model = GCN(4, 8, 2, 0.0).eval()
    preds = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    cands = {
        0: extract_candidate_edges(edge_index, 0, num_hops=2, num_nodes=4),
        1: extract_candidate_edges(edge_index, 1, num_hops=2, num_nodes=4),
    }

    class CorruptExplainer:
        def explain(self, model, data, node_id, target_class, candidate_edges, config=None, seed=0):
            if node_id == 0:
                # Claims success but returns NaN score
                scores = [float("nan")] * candidate_edges.num_candidate_edges
                return ExplainerResult(list(candidate_edges.edge_ids), scores, 0.01, "success")
            # Claims success but returns truncated scores (length mismatch)
            return ExplainerResult(list(candidate_edges.edge_ids), [0.5], 0.01, "success")

    monkeypatch.setattr(run_explainers, "SaliencyExplainer", lambda: CorruptExplainer())
    df, stats = run_explainers.run_single_configuration(
        dataset="bashapes",
        split="test",
        arch="gcn",
        gnn_seed=1000,
        explainer_seed=0,
        method="saliency",
        model=model,
        data=data,
        preds=preds,
        node_ids=[0, 1],
        candidates_by_node=cands,
        checkpoint_sha256="a" * 64,
        device=torch.device("cpu"),
        gnn_cfg={},
        pg_cfg={},
    )
    assert stats["n_success_nodes"] == 0
    assert stats["n_failed_nodes"] == 2
    assert set(df["status"].tolist()) == {"failed"}
    assert df["importance_score"].isna().all()
    assert len(stats["failed_details"]) == 2

    # 3. Also reject np.bool_ scores and reordered res.edge_ids (same length as candidate_pairs)
    import numpy as np

    class SubtleCorruptExplainer:
        def explain(self, model, data, node_id, target_class, candidate_edges, config=None, seed=0):
            if node_id == 0:
                scores = [np.bool_(True)] * candidate_edges.num_candidate_edges
                return ExplainerResult(list(candidate_edges.edge_ids), scores, 0.01, "success")
            # Reordered edge_ids with valid length and finite floats
            rev_ids = list(reversed(candidate_edges.edge_ids))
            scores = [0.5] * candidate_edges.num_candidate_edges
            return ExplainerResult(rev_ids, scores, 0.01, "success")

    monkeypatch.setattr(run_explainers, "SaliencyExplainer", lambda: SubtleCorruptExplainer())
    df2, stats2 = run_explainers.run_single_configuration(
        dataset="bashapes",
        split="test",
        arch="gcn",
        gnn_seed=1000,
        explainer_seed=0,
        method="saliency",
        model=model,
        data=data,
        preds=preds,
        node_ids=[0, 1],
        candidates_by_node=cands,
        checkpoint_sha256="a" * 64,
        device=torch.device("cpu"),
        gnn_cfg={},
        pg_cfg={},
    )
    assert stats2["n_success_nodes"] == 0
    assert stats2["n_failed_nodes"] == 2
    assert set(df2["status"].tolist()) == {"failed"}
    assert df2["importance_score"].isna().all()


def test_run_explainers_resume_blocks_unlogged_gnnexplainer_nan_csv_and_corrupt_runs_by_id(tmp_path):
    """Block --resume of gnnexplainer without run_log.json, NaN-corrupted CSVs, and malformed runs_by_id."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(17)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        pt,
    )
    (ckpt_ds / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": run_explainers.sha256_file(pt)}), encoding="utf-8"
    )

    out_dir = tmp_path / "results" / "scores"
    seed_dir = out_dir / "bashapes" / "gcn_s1000"

    # 1. Generate gnnexplainer e=0 with sum, delete run_log.json -> --resume with mean MUST raise ValueError
    rc1 = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "gnnexplainer",
        "--gnn-epochs", "4", "--gnn-size-loss-mode", "sum",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ])
    assert rc1 == 0
    (seed_dir / "run_log.json").unlink()
    with pytest.raises(ValueError, match="missing verified gnnexplainer hyperparams"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "0", "1",
            "--methods", "gnnexplainer",
            "--gnn-epochs", "4", "--gnn-size-loss-mode", "mean",
            "--resume",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        ])
    assert not (seed_dir / "gnnexplainer_e1.csv").exists()

    # 2. --resume must reject a saliency CSV whose method column has a NaN row
    rc_sal = run_explainers.main([
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "saliency",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ])
    assert rc_sal == 0
    sal_csv = seed_dir / "saliency_e0.csv"
    df_sal = pd.read_csv(sal_csv)
    df_sal.loc[0, "method"] = None
    df_sal.to_csv(sal_csv, index=False)
    with pytest.raises(ValueError, match="missing/NaN"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "0",
            "--methods", "saliency", "--resume",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        ])

    # 3. Corrupted non-dict entry inside run_log.json['runs_by_id'] must fail loudly with ValueError (not AttributeError)
    log_path = seed_dir / "run_log.json"
    log_obj = json.loads(log_path.read_text(encoding="utf-8"))
    log_obj["runs_by_id"]["corrupt_entry"] = "not_a_dict"
    log_path.write_text(json.dumps(log_obj), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted runs_by_id"):
        run_explainers.main([
            "--dataset", "bashapes", "--split", "test",
            "--gnn-seeds", "1000", "--explainer-seeds", "0",
            "--methods", "saliency",
            "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
        ])


def test_def_b9_and_b13_resume_rejects_nondict_log_null_node_ids_bool_ints_and_tampered_csv_sha256(tmp_path):
    """DEF-B9 & DEF-B13 (#3): Reject non-dict JSON, null node_ids, bool/true/false int cols, and tampered CSV SHA-256 under --resume."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(23)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        pt,
    )
    (ckpt_ds / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": run_explainers.sha256_file(pt)}), encoding="utf-8"
    )

    out_dir = tmp_path / "results" / "scores"
    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    base_cmd = [
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "saliency",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ]
    assert run_explainers.main(base_cmd) == 0

    log_path = seed_dir / "run_log.json"
    sal_csv = seed_dir / "saliency_e0.csv"
    orig_log_text = log_path.read_text(encoding="utf-8")
    orig_csv_bytes = sal_csv.read_bytes()

    # 1. DEF-B9: Non-dict JSON in run_log.json ([], "corrupt", null, 123) must raise ValueError under --resume
    for bad_json in ("[]", '"corrupt"', "null", "123"):
        log_path.write_text(bad_json, encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            run_explainers.main(base_cmd + ["--resume"])

    # 2. DEF-B9: node_ids: null (or non-list int) must raise ValueError cleanly (never TypeError on len(None))
    for bad_nids in (None, 42):
        obj = json.loads(orig_log_text)
        obj["node_ids"] = bad_nids
        log_path.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot --resume"):
            run_explainers.main(base_cmd + ["--resume"])

    # Restore valid run_log.json
    log_path.write_text(orig_log_text, encoding="utf-8")

    # 3. DEF-B13 (#3): Tampered importance_score in CSV while run_log has original csv_sha256 -> must FAIL --resume!
    df_tampered = pd.read_csv(sal_csv)
    df_tampered.loc[0, "importance_score"] = 999.0
    df_tampered.to_csv(sal_csv, index=False)
    with pytest.raises(ValueError, match="CSV SHA-256"):
        run_explainers.main(base_cmd + ["--resume"])

    # Also reject invalid falsy csv_sha256 ("", False) in run_log.json under --resume
    sal_csv.write_bytes(orig_csv_bytes)
    for bad_sha in ("", False, "bad_sha"):
        obj = json.loads(orig_log_text)
        obj["runs_by_id"]["bashapes_gcn_g1000_saliency_e0_test"]["csv_sha256"] = bad_sha
        log_path.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid csv_sha256"):
            run_explainers.main(base_cmd + ["--resume"])

    # 4. DEF-B13 (#3): Boolean / "true"/"false" in integer columns (gnn_seed, explainer_seed, seed, node_id) must FAIL --resume
    log_path.write_text(orig_log_text, encoding="utf-8")
    for col, bad_val in (
        ("explainer_seed", False),
        ("seed", "false"),
        ("gnn_seed", "true"),
        ("node_id", False),
    ):
        sal_csv.write_bytes(orig_csv_bytes)
        df_b = pd.read_csv(sal_csv)
        df_b[col] = df_b[col].astype(object)
        if col == "node_id":
            # Replace 0 -> False and 1 -> True so set({0, 1}) would falsely match without bool guard!
            df_b[col] = df_b[col].map(lambda v: False if int(v) == 0 else True)
        else:
            df_b[col] = bad_val
        df_b.to_csv(sal_csv, index=False)
        # Update csv_sha256 in run_log so the boolean-column guard is specifically exercised
        obj = json.loads(orig_log_text)
        obj["runs_by_id"]["bashapes_gcn_g1000_saliency_e0_test"]["csv_sha256"] = run_explainers.sha256_file(sal_csv)
        log_path.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(ValueError, match="boolean"):
            run_explainers.main(base_cmd + ["--resume"])


def test_adversarial_resume_rejects_corrupt_top_hp_null_failed_count_bogus_status_and_giant_int(tmp_path):
    """Block corrupt top-level hyperparams, null n_failed_nodes TypeError, bogus CSV status/NaN score, and 400-digit OverflowError under --resume."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    torch.manual_seed(29)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        pt,
    )
    (ckpt_ds / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": run_explainers.sha256_file(pt)}), encoding="utf-8"
    )

    out_dir = tmp_path / "results" / "scores"
    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    base_cmd = [
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "saliency",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ]
    assert run_explainers.main(base_cmd) == 0

    log_path = seed_dir / "run_log.json"
    sal_csv = seed_dir / "saliency_e0.csv"
    orig_log_text = log_path.read_text(encoding="utf-8")
    orig_csv_bytes = sal_csv.read_bytes()

    # 1. Corrupt top-level hyperparams (non-dict, unsupported method, non-dict method value) must raise ValueError
    for bad_hp in ("GARBAGE", {"gnnexplainer": "CORRUPT"}, {"unknown_method": {}}):
        obj = json.loads(orig_log_text)
        obj["hyperparams"] = bad_hp
        log_path.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted hyperparams"):
            run_explainers.main(base_cmd + ["--resume"])

    # 2. Corrupt n_failed_nodes (None, bool, negative, str) in runs_by_id must raise ValueError (never TypeError)
    for bad_nf in (None, True, -1, "0"):
        obj = json.loads(orig_log_text)
        obj["runs_by_id"]["bashapes_gcn_g1000_saliency_e0_test"]["n_failed_nodes"] = bad_nf
        log_path.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(ValueError, match="n_failed_nodes"):
            run_explainers.main(base_cmd + ["--resume"])

    # 3. Missing run_log.json with corrupted status='BOGUS' or NaN importance_score must raise ValueError
    log_path.unlink()
    df_bad_st = pd.read_csv(sal_csv)
    df_bad_st.loc[0, "status"] = "BOGUS"
    df_bad_st.to_csv(sal_csv, index=False)
    with pytest.raises(ValueError, match="status"):
        run_explainers.main(base_cmd + ["--resume"])

    sal_csv.write_bytes(orig_csv_bytes)
    df_bad_sc = pd.read_csv(sal_csv)
    df_bad_sc.loc[0, "importance_score"] = float("nan")
    df_bad_sc.to_csv(sal_csv, index=False)
    with pytest.raises(ValueError, match="importance_score"):
        run_explainers.main(base_cmd + ["--resume"])

    # 4. 400-digit integer in CSV node_id must raise ValueError (never unhandled OverflowError)
    sal_csv.write_bytes(orig_csv_bytes)
    lines = sal_csv.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[9] = "9" * 400
    lines[1] = ",".join(parts)
    sal_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot --resume"):
        run_explainers.main(base_cmd + ["--resume"])


def test_def_b18_atomic_json_no_nan_and_spoofed_retained_method(tmp_path):
    """DEF-B18 (#1 & #2): Reject NaN/Inf in _atomic_write_json and old_log, and derive retained-run method from canonical rid."""
    import pytest

    # 1. _atomic_write_json must reject NaN and Inf with allow_nan=False without writing corrupted files
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        out_json = tmp_path / "bad_out.json"
        with pytest.raises(ValueError):
            run_explainers._atomic_write_json({"hyperparams": {"lr": bad_val}}, out_json)
        assert not out_json.exists()

    # Build minimal synthetic fixture
    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)
    torch.manual_seed(31)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        pt,
    )
    (ckpt_ds / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": run_explainers.sha256_file(pt)}), encoding="utf-8"
    )

    out_dir = tmp_path / "results" / "scores"
    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    gnn_sum_cmd = [
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "gnnexplainer",
        "--gnn-epochs", "4", "--gnn-size-loss-mode", "sum",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ]
    assert run_explainers.main(gnn_sum_cmd) == 0

    log_path = seed_dir / "run_log.json"
    orig_log_text = log_path.read_text(encoding="utf-8")
    rid_gnn0 = "bashapes_gcn_g1000_gnnexplainer_e0_test"

    # 2. Non-finite NaN/Inf in old_log["hyperparams"] or persisted run metadata must raise ValueError (with and without --resume)
    for bad_num in (float("nan"), float("inf")):
        obj = json.loads(orig_log_text)
        obj["hyperparams"]["gnnexplainer"]["lr"] = bad_num
        log_path.write_text(json.dumps(obj, allow_nan=True), encoding="utf-8")
        for extra in (["--resume"], []):
            with pytest.raises(ValueError, match="non-finite"):
                run_explainers.main(
                    [
                        "--dataset", "bashapes", "--split", "test",
                        "--gnn-seeds", "1000", "--explainer-seeds", "0",
                        "--methods", "saliency",
                        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
                        *extra,
                    ]
                )

    # Non-finite in persisted runs_by_id metadata must raise ValueError
    obj_meta = json.loads(orig_log_text)
    obj_meta["runs_by_id"][rid_gnn0]["total_runtime_sec"] = float("nan")
    log_path.write_text(json.dumps(obj_meta, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        run_explainers.main(gnn_sum_cmd + ["--resume"])

    # Non-finite in top-level runtime metadata under --resume must raise ValueError
    obj_top = json.loads(orig_log_text)
    obj_top["last_invocation_runtime_sec"] = float("inf")
    log_path.write_text(json.dumps(obj_top, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        run_explainers.main(gnn_sum_cmd + ["--resume"])

    # 3. Spoofing rinfo["method"] on retained run must NEVER bypass hyperparam mismatch validation
    for spoof_method in ("saliency", "nearest_first", "fake_method", 123):
        obj_sp = json.loads(orig_log_text)
        obj_sp["runs_by_id"][rid_gnn0]["method"] = spoof_method
        log_path.write_text(json.dumps(obj_sp), encoding="utf-8")
        for extra in (["--resume"], []):
            with pytest.raises(ValueError, match="method"):
                run_explainers.main(
                    [
                        "--dataset", "bashapes", "--split", "test",
                        "--gnn-seeds", "1000", "--explainer-seeds", "1",
                        "--methods", "gnnexplainer",
                        "--gnn-epochs", "4", "--gnn-size-loss-mode", "mean",
                        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
                        *extra,
                    ]
                )

    # 4. Omitting rinfo["method"] entirely still derives 'gnnexplainer' from canonical rid and blocks hyperparam mixing
    obj_nom = json.loads(orig_log_text)
    del obj_nom["runs_by_id"][rid_gnn0]["method"]
    log_path.write_text(json.dumps(obj_nom), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot mix gnnexplainer hyperparams"):
        run_explainers.main(
            [
                "--dataset", "bashapes", "--split", "test",
                "--gnn-seeds", "1000", "--explainer-seeds", "1",
                "--methods", "gnnexplainer",
                "--gnn-epochs", "4", "--gnn-size-loss-mode", "mean",
                "--resume",
                "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            ]
        )

    # 5. Non-canonical retained run_id under --resume must be rejected
    obj_noncanon = json.loads(orig_log_text)
    obj_noncanon["runs_by_id"]["non_canonical_id"] = {"method": "saliency", "hyperparams": {}}
    log_path.write_text(json.dumps(obj_noncanon), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical run_id"):
        run_explainers.main(gnn_sum_cmd + ["--resume"])


def test_def_b18_hardened_load_node_list_and_bounds_guard(tmp_path):
    """DEF-B18 (#3): load_node_list & callers reject bool, negative, non-int, and out-of-bounds node IDs."""
    import pytest
    from sna.data.candidate_edges import extract_candidate_edges

    # 1. Direct load_node_list rejections (bool, negative, float, string, None, OOB, malformed JSON, bad max_nodes)
    bad_payloads = [
        ("bool_true", {"nodes": [True]}),
        ("bool_false", {"nodes": [False]}),
        ("bool_after_slice", {"nodes": [0, True]}),
        ("neg_int", {"nodes": [-1]}),
        ("neg_after_slice", {"nodes": [0, -2]}),
        ("float_val", {"nodes": [1.5]}),
        ("str_val", {"nodes": ["0"]}),
        ("none_val", {"nodes": [None]}),
        ("oob_exact", {"nodes": [0, 5]}),
        ("oob_large", {"nodes": [0, 999]}),
        ("empty_list", {"nodes": []}),
        ("dup_ids", {"nodes": [0, 0]}),
        ("missing_key", {"other": [0, 1]}),
        ("nodes_str", {"nodes": "012"}),
        ("root_int", 123),
    ]
    for tag, payload in bad_payloads:
        nf = tmp_path / f"nodes_{tag}.json"
        nf.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            run_explainers.load_node_list(
                dataset="bashapes",
                split="test",
                data_dir=tmp_path,
                nodes_file=nf,
                max_nodes=1 if "after_slice" in tag or "oob" in tag else None,
                num_nodes=5,
            )

    # Directory passed as nodes_file raises FileNotFoundError (not PermissionError)
    with pytest.raises(FileNotFoundError):
        run_explainers.load_node_list("bashapes", "test", tmp_path, nodes_file=tmp_path, max_nodes=None, num_nodes=5)

    # Boolean max_nodes or num_nodes rejected
    good_nf = tmp_path / "good_nodes.json"
    good_nf.write_text(json.dumps({"nodes": [0, 1, 2]}), encoding="utf-8")
    with pytest.raises(ValueError):
        run_explainers.load_node_list("bashapes", "test", tmp_path, nodes_file=good_nf, max_nodes=True, num_nodes=5)
    with pytest.raises(ValueError):
        run_explainers.load_node_list("bashapes", "test", tmp_path, nodes_file=good_nf, max_nodes=1, num_nodes=False)

    # Valid call succeeds cleanly
    nodes_ok, path_ok, sha_ok = run_explainers.load_node_list(
        "bashapes", "test", tmp_path, nodes_file=good_nf, max_nodes=2, num_nodes=5
    )
    assert nodes_ok == [0, 1] and path_ok == good_nf and len(sha_ok) == 64

    # 2. End-to-end main() rejects OOB, negative, and bool nodes in test_nodes.json
    data_dir = tmp_path / "e2e_data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)
    x = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    for bad_nodes in ([0, 4], [-1], [True]):
        (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": bad_nodes}), encoding="utf-8")
        with pytest.raises(ValueError):
            run_explainers.main(
                [
                    "--dataset", "bashapes", "--split", "test",
                    "--gnn-seeds", "1000", "--explainer-seeds", "0",
                    "--methods", "saliency",
                    "--data-dir", str(data_dir),
                    "--checkpoints-dir", str(tmp_path / "ckpts"),
                    "--out-dir", str(tmp_path / "out"),
                ]
            )

    # 3. Direct run_single_configuration also rejects bool, negative, OOB, empty, and duplicate node_ids
    model = GCN(4, 8, 2, 0.0).eval()
    preds = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    cands = {0: extract_candidate_edges(edge_index, 0, num_hops=2, num_nodes=4)}
    for bad_nids in ([True], [-1], [4], [], [0, 0]):
        with pytest.raises(ValueError):
            run_explainers.run_single_configuration(
                dataset="bashapes",
                split="test",
                arch="gcn",
                gnn_seed=1000,
                explainer_seed=0,
                method="saliency",
                model=model,
                data={"x": x, "edge_index": edge_index},
                preds=preds,
                node_ids=bad_nids,
                candidates_by_node=cands,
                checkpoint_sha256="a" * 64,
                device=torch.device("cpu"),
                gnn_cfg={},
                pg_cfg={},
            )


def test_def_b18_adversarial_ckpt_meta_noncanon_foreign_rid_poison_csvfile_and_dir_guards(tmp_path):
    """Adversarial DEF-B18 hardening: malformed ckpt .json, non-canonical/foreign rid without --resume, top-level Inf without --resume, poisoned csv_file, NaN nodes JSON, and CSV dir guard."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)
    torch.manual_seed(37)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        pt,
    )
    ckpt_sha = run_explainers.sha256_file(pt)
    meta_json = ckpt_ds / "gcn_s1000.json"

    # 1. load_checkpoint_strict must reject non-dict .json, falsy/invalid sha256, NaN/Inf in .json, and directory .json
    for tag, bad_text in (
        ("list", "[]"),
        ("str", '"bad"'),
        ("empty_sha", json.dumps({"sha256": ""})),
        ("false_sha", json.dumps({"sha256": False})),
        ("bad_ckpt_sha", json.dumps({"checkpoint_sha256": "not-hex"})),
        ("nan_meta", json.dumps({"sha256": ckpt_sha, "val_acc": float("nan")}, allow_nan=True)),
        ("inf_meta", json.dumps({"sha256": ckpt_sha, "val_acc": float("inf")}, allow_nan=True)),
    ):
        meta_json.write_text(bad_text, encoding="utf-8")
        with pytest.raises(ValueError):
            run_explainers.load_checkpoint_strict(pt, "gcn", torch.device("cpu"))

    meta_json.unlink()
    meta_json.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        run_explainers.load_checkpoint_strict(pt, "gcn", torch.device("cpu"))
    meta_json.rmdir()
    meta_json.write_text(json.dumps({"gnn_seed": 1000, "sha256": ckpt_sha}), encoding="utf-8")

    # 2. load_node_list must reject non-finite NaN/Inf inside node list JSON metadata
    nan_nodes_path = tmp_path / "nan_nodes.json"
    nan_nodes_path.write_text(
        json.dumps({"nodes": [0, 1], "extra": float("nan")}, allow_nan=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Non-finite"):
        run_explainers.load_node_list("bashapes", "test", data_dir, nan_nodes_path, None, num_nodes=5)

    out_dir = tmp_path / "results" / "scores"
    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    gnn_sum_cmd = [
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "gnnexplainer",
        "--gnn-epochs", "4", "--gnn-size-loss-mode", "sum",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ]
    assert run_explainers.main(gnn_sum_cmd) == 0

    log_path = seed_dir / "run_log.json"
    orig_log_text = log_path.read_text(encoding="utf-8")
    rid_gnn0 = "bashapes_gcn_g1000_gnnexplainer_e0_test"

    # 3. Non-canonical rid rejected WITHOUT --resume on same-context merge
    obj_nc = json.loads(orig_log_text)
    obj_nc["runs_by_id"]["weird_noncanon"] = {"method": "saliency", "hyperparams": {"epochs": 999}}
    log_path.write_text(json.dumps(obj_nc), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical run_id"):
        run_explainers.main(
            [
                "--dataset", "bashapes", "--split", "test",
                "--gnn-seeds", "1000", "--explainer-seeds", "1",
                "--methods", "gnnexplainer",
                "--gnn-epochs", "4", "--gnn-size-loss-mode", "sum",
                "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            ]
        )

    # 4. Foreign canonical rid (different dataset/gnn_seed/split) rejected with and without --resume
    obj_for = json.loads(orig_log_text)
    obj_for["runs_by_id"]["cora_gcn_g1001_saliency_e0_val"] = {"method": "saliency", "hyperparams": {}}
    log_path.write_text(json.dumps(obj_for), encoding="utf-8")
    for extra in (["--resume"], []):
        with pytest.raises(ValueError, match="context"):
            run_explainers.main(gnn_sum_cmd + extra)

    # 5. Top-level Inf in old_log rejected WITHOUT --resume on same-context merge
    obj_inf = json.loads(orig_log_text)
    obj_inf["last_invocation_runtime_sec"] = float("inf")
    log_path.write_text(json.dumps(obj_inf, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        run_explainers.main(
            [
                "--dataset", "bashapes", "--split", "test",
                "--gnn-seeds", "1000", "--explainer-seeds", "0",
                "--methods", "saliency",
                "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            ]
        )

    # 6. Poisoned csv_file ('../escape.csv', False, 'wrong.csv') and explicit method=None rejected under --resume
    for bad_cf in ("../escape.csv", False, "wrong.csv"):
        obj_cf = json.loads(orig_log_text)
        obj_cf["runs_by_id"][rid_gnn0]["csv_file"] = bad_cf
        log_path.write_text(json.dumps(obj_cf), encoding="utf-8")
        with pytest.raises(ValueError, match="csv_file"):
            run_explainers.main(gnn_sum_cmd + ["--resume"])

    obj_mnull = json.loads(orig_log_text)
    obj_mnull["runs_by_id"][rid_gnn0]["method"] = None
    log_path.write_text(json.dumps(obj_mnull), encoding="utf-8")
    with pytest.raises(ValueError, match="method"):
        run_explainers.main(gnn_sum_cmd + ["--resume"])

    # 7. _validate_resumable_csv directly rejects non-dict old_top_hyperparams[method]
    log_path.write_text(orig_log_text, encoding="utf-8")
    with pytest.raises(ValueError, match="top-level gnnexplainer hyperparams"):
        run_explainers._validate_resumable_csv(
            csv_path=seed_dir / "gnnexplainer_e0.csv",
            expected_run_id=rid_gnn0,
            dataset="bashapes",
            split="test",
            arch="gcn",
            gnn_seed=1000,
            method="gnnexplainer",
            explainer_seed=0,
            ckpt_sha=ckpt_sha,
            node_ids=[0, 1],
            expected_hp={"epochs": 4, "lr": 0.01, "coeff_size": 0.005, "coeff_ent": 0.1, "size_loss_mode": "sum"},
            prior_run_entry={"method": "gnnexplainer"},
            old_top_hyperparams={"gnnexplainer": "CORRUPT_STRING"},
        )

    # 8. Directory standing in for output CSV path rejected upfront with ValueError (no PermissionError)
    bad_csv_dir = seed_dir / "saliency_e0.csv"
    bad_csv_dir.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        run_explainers.main(
            [
                "--dataset", "bashapes", "--split", "test",
                "--gnn-seeds", "1000", "--explainer-seeds", "0",
                "--methods", "saliency",
                "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            ]
        )


def test_round5_parity_sibling_corrupt_hyperparams_nondict_pt_and_nonfile_run_log(tmp_path, monkeypatch):
    """Round 5 parity Notes #3 & #5: sibling corrupt hyperparams on merge, non-dict .pt guard, and non-file/symlink run_log guard."""
    import pytest

    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)
    torch.manual_seed(41)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    model = GCN(4, 8, 2, 0.0).eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)
    ckpt_dir = tmp_path / "results" / "checkpoints"
    ckpt_ds = ckpt_dir / "bashapes"
    ckpt_ds.mkdir(parents=True)
    pt = ckpt_ds / "gcn_s1000.pt"
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": 4, "hidden": 8, "n_classes": 2, "pred": pred},
        pt,
    )
    ckpt_sha = run_explainers.sha256_file(pt)
    (ckpt_ds / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": ckpt_sha}), encoding="utf-8"
    )

    # Note #5a: load_checkpoint_strict must raise ValueError (not TypeError) on corrupt non-dict .pt
    for tag, bad_obj in (
        ("list", [1, 2, 3]),
        ("str", "not_a_dict"),
        ("tensor", torch.tensor([1, 2, 3])),
        ("none", None),
    ):
        bad_pt = tmp_path / f"corrupt_{tag}.pt"
        torch.save(bad_obj, bad_pt)
        bad_sha = run_explainers.sha256_file(bad_pt)
        bad_pt.with_suffix(".json").write_text(json.dumps({"sha256": bad_sha}), encoding="utf-8")
        with pytest.raises(ValueError, match="expected dict"):
            run_explainers.load_checkpoint_strict(bad_pt, "gcn", torch.device("cpu"))

    # Note #3: On same_context merge, corrupt hyperparams on an UNTOUCHED sibling method MUST raise ValueError
    out_dir = tmp_path / "results" / "scores"
    seed_dir = out_dir / "bashapes" / "gcn_s1000"
    assert (
        run_explainers.main(
            [
                "--dataset", "bashapes", "--split", "test",
                "--gnn-seeds", "1000", "--explainer-seeds", "0",
                "--methods", "gnnexplainer",
                "--gnn-epochs", "4",
                "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
            ]
        )
        == 0
    )

    log_path = seed_dir / "run_log.json"
    orig_log_text = log_path.read_text(encoding="utf-8")
    rid_gnn0 = "bashapes_gcn_g1000_gnnexplainer_e0_test"
    sal_cmd = [
        "--dataset", "bashapes", "--split", "test",
        "--gnn-seeds", "1000", "--explainer-seeds", "0",
        "--methods", "saliency",
        "--data-dir", str(data_dir), "--checkpoints-dir", str(ckpt_dir), "--out-dir", str(out_dir),
    ]
    for bad_hp in ("CORRUPT_STRING", [1, 2], 123, True):
        obj = json.loads(orig_log_text)
        obj["runs_by_id"][rid_gnn0]["hyperparams"] = bad_hp
        log_path.write_text(json.dumps(obj), encoding="utf-8")
        for extra in ([], ["--resume"]):
            with pytest.raises(ValueError, match="hyperparams"):
                run_explainers.main(sal_cmd + extra)

    # Note #5b: run_log_path as a directory, dangling NTFS junction / broken symlink, or simulated broken symlink raises ValueError
    import os
    import subprocess

    def _make_dangling_link(link_path: pathlib.Path, missing_target: pathlib.Path) -> None:
        if os.name == "nt":
            rc_j = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_path), str(missing_target)],
                capture_output=True,
                text=True,
            )
            assert rc_j.returncode == 0, f"mklink /J failed: {rc_j.stdout} {rc_j.stderr}"
        else:
            os.symlink(str(missing_target), str(link_path))

    def _remove_dangling_link(link_path: pathlib.Path) -> None:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "rmdir", str(link_path)], capture_output=True)
        elif link_path.is_symlink():
            link_path.unlink()

    log_path.unlink()
    log_path.mkdir()
    with pytest.raises(ValueError, match="expected regular file"):
        run_explainers.main(sal_cmd)
    log_path.rmdir()

    # Real OS-level dangling NTFS junction / broken symlink at run_log.json (never PermissionError, no .tmp left)
    _make_dangling_link(log_path, tmp_path / "missing_run_log_target")
    try:
        with pytest.raises(ValueError, match="expected regular file"):
            run_explainers.main(sal_cmd)
        assert not log_path.with_suffix(".json.tmp").exists()
    finally:
        _remove_dangling_link(log_path)

    # Real OS-level dangling NTFS junction / broken symlink at output CSV path (never PermissionError)
    sal_csv_path = seed_dir / "saliency_e0.csv"
    if sal_csv_path.exists():
        sal_csv_path.unlink()
    _make_dangling_link(sal_csv_path, tmp_path / "missing_csv_target")
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            run_explainers.main(sal_cmd)
        assert not sal_csv_path.with_suffix(".csv.tmp").exists()
    finally:
        _remove_dangling_link(sal_csv_path)

    # Real OS-level dangling NTFS junction / broken symlink at checkpoint metadata .json
    ckpt_meta_path = ckpt_ds / "gcn_s1000.json"
    orig_meta_text = ckpt_meta_path.read_text(encoding="utf-8")
    ckpt_meta_path.unlink()
    _make_dangling_link(ckpt_meta_path, tmp_path / "missing_meta_target")
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            run_explainers.load_checkpoint_strict(pt, "gcn", torch.device("cpu"))
    finally:
        _remove_dangling_link(ckpt_meta_path)
        ckpt_meta_path.write_text(orig_meta_text, encoding="utf-8")

    # Simulate broken symlink at run_log_path (exists=False, is_file=False, is_symlink=True)
    orig_is_symlink = pathlib.Path.is_symlink

    def _fake_is_symlink(self: pathlib.Path) -> bool:
        if self == log_path:
            return True
        return orig_is_symlink(self)

    monkeypatch.setattr(pathlib.Path, "is_symlink", _fake_is_symlink)
    with pytest.raises(ValueError, match="expected regular file"):
        run_explainers.main(sal_cmd)
