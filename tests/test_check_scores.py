"""Comprehensive unit tests for scripts/check_scores.py (H2 Handover Validator - Khoi)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pandas as pd
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import check_scores
from sna.data.candidate_edges import extract_candidate_edges


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_valid_fixture(tmp_path: pathlib.Path, include_isolated_node: bool = False):
    """Build a minimal, 100% valid synthetic dataset, checkpoint, scores dir, and run_log.json."""
    data_dir = tmp_path / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)

    # 4-node connected graph + optional isolated node 4
    if include_isolated_node:
        x = torch.ones(5, 4)
        nodes = [0, 1, 4]
    else:
        x = torch.ones(4, 4)
        nodes = [0, 1]

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 2], [1, 0, 2, 1, 3, 2, 2, 0]], dtype=torch.long
    )
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": nodes}), encoding="utf-8")

    ckpt_dir = tmp_path / "results" / "checkpoints" / "bashapes"
    ckpt_dir.mkdir(parents=True)
    ckpt_pt = ckpt_dir / "gcn_s1000.pt"
    ckpt_bytes = b"synthetic-checkpoint-binary-for-unit-test"
    ckpt_pt.write_bytes(ckpt_bytes)
    ckpt_sha = _sha256_bytes(ckpt_bytes)
    (ckpt_dir / "gcn_s1000.json").write_text(
        json.dumps({"gnn_seed": 1000, "sha256": ckpt_sha}), encoding="utf-8"
    )

    scores_dir = tmp_path / "results" / "scores" / "bashapes"
    seed_dir = scores_dir / "gcn_s1000"
    seed_dir.mkdir(parents=True)

    def make_csv(method: str, expl_seed: int, score_offset: float) -> pathlib.Path:
        rows = []
        run_id = f"bashapes_gcn_g1000_{method}_e{expl_seed}_test"
        for nid in nodes:
            cand = extract_candidate_edges(edge_index, nid, num_hops=2, num_nodes=x.size(0))
            if cand.num_candidate_edges == 0:
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": "bashapes",
                        "split": "test",
                        "arch": "gcn",
                        "gnn_seed": 1000,
                        "checkpoint_sha256": ckpt_sha,
                        "method": method,
                        "explainer_seed": expl_seed,
                        "seed": expl_seed,
                        "node_id": nid,
                        "target_class": nid % 2,
                        "edge_id": -1,
                        "u": -1,
                        "v": -1,
                        "importance_score": float("nan"),
                        "runtime_sec": 0.001,
                        "status": "empty_candidates",
                    }
                )
            else:
                for eid, (u, v) in zip(cand.edge_ids, cand.candidate_pairs):
                    rows.append(
                        {
                            "run_id": run_id,
                            "dataset": "bashapes",
                            "split": "test",
                            "arch": "gcn",
                            "gnn_seed": 1000,
                            "checkpoint_sha256": ckpt_sha,
                            "method": method,
                            "explainer_seed": expl_seed,
                            "seed": expl_seed,
                            "node_id": nid,
                            "target_class": nid % 2,
                            "edge_id": eid,
                            "u": u,
                            "v": v,
                            "importance_score": 0.1 * (eid + 1) + score_offset,
                            "runtime_sec": 0.012,
                            "status": "success",
                        }
                    )
        out_csv = seed_dir / f"{method}_e{expl_seed}.csv"
        pd.DataFrame(rows, columns=check_scores.REQUIRED_COLUMNS).to_csv(out_csv, index=False)
        return out_csv

    sal0 = make_csv("saliency", 0, 0.0)
    sal1 = make_csv("saliency", 1, 0.0)  # identical to e=0 (deterministic)
    gnn0 = make_csv("gnnexplainer", 0, 0.05)

    gnn_hp = {
        "epochs": 200,
        "lr": 0.01,
        "coeff_size": 0.005,
        "coeff_ent": 0.1,
        "size_loss_mode": "mean",
    }
    (seed_dir / "run_log.json").write_text(
        json.dumps(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1000,
                "split": "test",
                "checkpoint_sha256": ckpt_sha,
                "node_ids": nodes,
                "hyperparams": {"gnnexplainer": gnn_hp},
                "runs_by_id": {
                    "bashapes_gcn_g1000_saliency_e0_test": {
                        "run_id": "bashapes_gcn_g1000_saliency_e0_test",
                        "method": "saliency",
                        "explainer_seed": 0,
                        "hyperparams": {},
                        "csv_file": sal0.name,
                        "csv_sha256": check_scores.sha256_file(sal0),
                    },
                    "bashapes_gcn_g1000_saliency_e1_test": {
                        "run_id": "bashapes_gcn_g1000_saliency_e1_test",
                        "method": "saliency",
                        "explainer_seed": 1,
                        "hyperparams": {},
                        "csv_file": sal1.name,
                        "csv_sha256": check_scores.sha256_file(sal1),
                    },
                    "bashapes_gcn_g1000_gnnexplainer_e0_test": {
                        "run_id": "bashapes_gcn_g1000_gnnexplainer_e0_test",
                        "method": "gnnexplainer",
                        "explainer_seed": 0,
                        "hyperparams": gnn_hp,
                        "csv_file": gnn0.name,
                        "csv_sha256": check_scores.sha256_file(gnn0),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    return {
        "scores_dir": scores_dir,
        "seed_dir": seed_dir,
        "checkpoints_dir": tmp_path / "results" / "checkpoints",
        "data_dir": data_dir,
        "ckpt_sha": ckpt_sha,
        "sal0": sal0,
        "sal1": sal1,
        "gnn0": gnn0,
    }


def test_check_scores_passes_valid_handover(tmp_path):
    fx = _build_valid_fixture(tmp_path, include_isolated_node=False)
    report = check_scores.validate_scores_dir(
        scores_path=fx["scores_dir"],
        checkpoints_dir=fx["checkpoints_dir"],
        data_dir=fx["data_dir"],
    )
    assert report["status"] == "PASS", report["errors"]
    assert report["n_files"] == 3
    assert report["n_errors"] == 0
    assert report["saliency_determinism"]["bashapes/gcn_s1000"]["max_diff"] == 0.0
    assert "bashapes" in report["graph_verified_datasets"]

    # Also verify CLI return code == 0
    rc = check_scores.main(
        [
            str(fx["scores_dir"]),
            "--checkpoints-dir",
            str(fx["checkpoints_dir"]),
            "--data-dir",
            str(fx["data_dir"]),
        ]
    )
    assert rc == 0


def test_check_scores_passes_with_empty_candidates_node(tmp_path):
    fx = _build_valid_fixture(tmp_path, include_isolated_node=True)
    report = check_scores.validate_scores_dir(
        scores_path=fx["scores_dir"],
        checkpoints_dir=fx["checkpoints_dir"],
        data_dir=fx["data_dir"],
    )
    assert report["status"] == "PASS", report["errors"]


def test_check_scores_fails_missing_column(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"]).drop(columns=["runtime_sec"])
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "columns" for e in report["errors"])


def test_check_scores_fails_non_finite_score(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"])
    df.loc[0, "importance_score"] = float("nan")
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "finite_importance_score" for e in report["errors"])


def test_check_scores_fails_u_ge_v(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"])
    u0, v0 = int(df.loc[0, "u"]), int(df.loc[0, "v"])
    df.loc[0, "u"], df.loc[0, "v"] = v0, u0
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "canonical_undirected_edge" for e in report["errors"])


def test_check_scores_fails_edge_alignment_mismatch_across_methods(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"])
    # Swap edge_ids 0 and 1 on node 0
    mask0 = df["node_id"] == 0
    idx = df.index[mask0].tolist()
    df.loc[idx[0], "edge_id"], df.loc[idx[1], "edge_id"] = 1, 0
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "cross_method_edge_alignment" for e in report["errors"])


def test_check_scores_fails_checkpoint_hash_mismatch(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    bad_sha = "0" * 64
    for p in (fx["sal0"], fx["sal1"], fx["gnn0"]):
        df = pd.read_csv(p)
        df["checkpoint_sha256"] = bad_sha
        df.to_csv(p, index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "checkpoint_hash_mismatch" for e in report["errors"])


def test_check_scores_fails_missing_node(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"])
    df = df[df["node_id"] != 1]
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "missing_nodes" for e in report["errors"])


def test_check_scores_fails_duplicate_rows(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"])
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "duplicate_rows" for e in report["errors"])


def test_check_scores_fails_seed_mismatch_and_saliency_nondeterminism(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["sal1"])
    df.loc[0, "importance_score"] = df.loc[0, "importance_score"] + 0.123
    df.loc[0, "seed"] = 99
    df.to_csv(fx["sal1"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    checks = {e["check"] for e in report["errors"]}
    assert "seed_consistency" in checks
    assert "saliency_determinism" in checks


def test_check_scores_fails_failed_status_and_missing_run_log(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    (fx["seed_dir"] / "run_log.json").unlink()
    df = pd.read_csv(fx["gnn0"])
    df.loc[0, "status"] = "failed"
    df.to_csv(fx["gnn0"], index=False)
    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    checks = {e["check"] for e in report["errors"]}
    assert "run_log" in checks
    assert "failed_rows" in checks


def test_check_scores_json_only_checkpoint_and_edge_cases(tmp_path):
    """Verify hash check works with .json only (CI without .pt) and catches inf, negative runtime, bad run_id, target mismatch."""
    fx = _build_valid_fixture(tmp_path)
    # Remove .pt, keep .json (as in GitHub repo without binary checkpoints)
    (fx["checkpoints_dir"] / "bashapes" / "gcn_s1000.pt").unlink()
    report_ok = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report_ok["status"] == "PASS", report_ok["errors"]

    # Inject inf score, negative runtime, mismatched target_class across methods, bad run_id
    df = pd.read_csv(fx["gnn0"])
    df.loc[0, "importance_score"] = float("inf")
    df.loc[0, "runtime_sec"] = -1.0
    df["run_id"] = "wrong_run_id_format"
    df.to_csv(fx["gnn0"], index=False)

    report_bad = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report_bad["status"] == "FAIL"
    checks = {e["check"] for e in report_bad["errors"]}
    assert "finite_importance_score" in checks
    assert "runtime_sec" in checks
    assert "run_id_format" in checks


def test_check_scores_fails_cross_method_target_class_and_run_log_hash_mismatch(tmp_path):
    fx = _build_valid_fixture(tmp_path)
    df = pd.read_csv(fx["gnn0"])
    df.loc[df["node_id"] == 0, "target_class"] = 99
    df.to_csv(fx["gnn0"], index=False)

    log_path = fx["seed_dir"] / "run_log.json"
    log_obj = json.loads(log_path.read_text(encoding="utf-8"))
    log_obj["checkpoint_sha256"] = "f" * 64
    log_path.write_text(json.dumps(log_obj), encoding="utf-8")

    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    checks = {e["check"] for e in report["errors"]}
    assert "cross_method_target_class" in checks
    assert "run_log_hash_mismatch" in checks


def test_check_scores_verifies_graph_candidates_across_multiple_gnn_seeds(tmp_path):
    """Graph candidate verification must check every gnn_seed, not skip nid after seed 1000."""
    fx = _build_valid_fixture(tmp_path, include_isolated_node=False)

    # Create a second gnn_seed 1001 with valid hash/metadata but corrupted edge list (drops 1 edge)
    ckpt_dir = fx["checkpoints_dir"] / "bashapes"
    ckpt_bytes2 = b"synthetic-checkpoint-binary-seed-1001"
    (ckpt_dir / "gcn_s1001.pt").write_bytes(ckpt_bytes2)
    sha2 = _sha256_bytes(ckpt_bytes2)
    (ckpt_dir / "gcn_s1001.json").write_text(
        json.dumps({"gnn_seed": 1001, "sha256": sha2}), encoding="utf-8"
    )

    seed_dir2 = fx["scores_dir"] / "gcn_s1001"
    seed_dir2.mkdir(parents=True)
    df = pd.read_csv(fx["sal0"])
    df["gnn_seed"] = 1001
    df["run_id"] = "bashapes_gcn_g1001_saliency_e0_test"
    df["checkpoint_sha256"] = sha2
    # Drop the last candidate edge for node 0 so edge_ids remain contiguous 0..k-2 but mismatch data.pt
    node0_idx = df.index[df["node_id"] == 0].tolist()
    df_bad = df.drop(index=[node0_idx[-1]]).reset_index(drop=True)
    sal2_csv = seed_dir2 / "saliency_e0.csv"
    df_bad.to_csv(sal2_csv, index=False)
    (seed_dir2 / "run_log.json").write_text(
        json.dumps(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1001,
                "split": "test",
                "checkpoint_sha256": sha2,
                "node_ids": [0, 1],
                "runs_by_id": {
                    "bashapes_gcn_g1001_saliency_e0_test": {
                        "run_id": "bashapes_gcn_g1001_saliency_e0_test",
                        "method": "saliency",
                        "explainer_seed": 0,
                        "hyperparams": {},
                        "csv_file": sal2_csv.name,
                        "csv_sha256": check_scores.sha256_file(sal2_csv),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert report["status"] == "FAIL"
    assert any(e["check"] == "graph_candidate_mismatch" and "gcn_s1001" in e["file"] for e in report["errors"])


def test_check_scores_allow_failed_skips_failed_nodes_and_validates_rest(tmp_path):
    """--allow-failed must pass when one run has a properly formatted failed node while others succeed."""
    fx = _build_valid_fixture(tmp_path, include_isolated_node=False)
    df = pd.read_csv(fx["gnn0"])
    failed_row = df[df["node_id"] == 0].iloc[:1].copy()
    failed_row["edge_id"] = -1
    failed_row["u"] = -1
    failed_row["v"] = -1
    failed_row["importance_score"] = float("nan")
    failed_row["status"] = "failed"
    df_new = pd.concat([failed_row, df[df["node_id"] != 0]], ignore_index=True)
    df_new.to_csv(fx["gnn0"], index=False)
    # Refresh csv_sha256 in run_log.json as run_explainers.py would after recording a failed node
    log_path = fx["seed_dir"] / "run_log.json"
    log_obj = json.loads(log_path.read_text(encoding="utf-8"))
    log_obj["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = check_scores.sha256_file(fx["gnn0"])
    log_path.write_text(json.dumps(log_obj), encoding="utf-8")

    # Default mode must still FAIL
    rep_strict = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"], allow_failed=False
    )
    assert rep_strict["status"] == "FAIL"
    assert any(e["check"] == "failed_rows" for e in rep_strict["errors"])

    # --allow-failed must PASS cleanly (including verify_graph=True) with warning
    rep_allow = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"], allow_failed=True
    )
    assert rep_allow["status"] == "PASS", rep_allow["errors"]
    assert any("Allowed failed node 0" in w for w in rep_allow["warnings"])


def test_check_scores_rejects_whitespace_tampered_csv_booleans_missing_test_nodes_and_bad_placeholders(tmp_path):
    """Catch whitespace identifiers, tampered CSV SHA-256, booleans, bogus empty_candidates, missing test_nodes, and split leakage."""
    # 1. Booleans must fail _is_nonneg_int_series
    ok_bool, bad_lines = check_scores._is_nonneg_int_series(pd.Series([True, False]))
    assert not ok_bool and bad_lines == [2, 3]
    ok_bool_str, _ = check_scores._is_nonneg_int_series(pd.Series(["True", "1"]))
    assert not ok_bool_str

    # 2. Whitespace in method + CSV SHA-256 tampering against run_log.json
    fx = _build_valid_fixture(tmp_path, include_isolated_node=False)

    # Sanity: untouched fixture passes with csv_sha256 recorded for all 3 CSVs
    assert check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )["status"] == "PASS"

    # Tamper saliency_e0.csv with whitespace in method + bogus arch
    df_sal = pd.read_csv(fx["sal0"])
    df_sal["method"] = " saliency "
    df_sal.to_csv(fx["sal0"], index=False)
    rep_tamper = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_tamper["status"] == "FAIL"
    checks_t = {e["check"] for e in rep_tamper["errors"]}
    assert "identifier_whitespace" in checks_t
    assert "csv_sha256_mismatch" in checks_t

    # 3. Missing test_nodes.json must FAIL for split='test' (no blind fallback to run_log.json)
    fx2 = _build_valid_fixture(tmp_path / "case_notest", include_isolated_node=False)
    (fx2["data_dir"] / "bashapes" / "test_nodes.json").unlink()
    rep_notest = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep_notest["status"] == "FAIL"
    assert any(e["check"] == "expected_nodes_lookup" for e in rep_notest["errors"])

    # 4. Bogus empty_candidates row with finite importance_score and non-(-1) u/v must FAIL
    fx3 = _build_valid_fixture(tmp_path / "case_bad_ec", include_isolated_node=True)
    df_ec = pd.read_csv(fx3["gnn0"])
    df_ec.loc[df_ec["status"] == "empty_candidates", "importance_score"] = 0.99
    df_ec.loc[df_ec["status"] == "empty_candidates", "u"] = 0
    df_ec.to_csv(fx3["gnn0"], index=False)
    rep_ec = check_scores.validate_scores_dir(
        fx3["scores_dir"], checkpoints_dir=fx3["checkpoints_dir"], data_dir=fx3["data_dir"], verify_graph=False
    )
    assert rep_ec["status"] == "FAIL"
    assert any(e["check"] == "placeholder_row_invalid" for e in rep_ec["errors"])

    # 5. Split-mask anti-leakage check when data.pt has test_mask / train_mask
    fx4 = _build_valid_fixture(tmp_path / "case_leak", include_isolated_node=False)
    data_pt = fx4["data_dir"] / "bashapes" / "data.pt"
    d_obj = torch.load(data_pt, map_location="cpu", weights_only=False)
    # Mark node 0 as train (not test) -> must fail split_mask_leakage
    d_obj["train_mask"] = torch.tensor([True, False, False, False])
    d_obj["test_mask"] = torch.tensor([False, True, False, False])
    torch.save(d_obj, data_pt)
    rep_leak = check_scores.validate_scores_dir(
        fx4["scores_dir"], checkpoints_dir=fx4["checkpoints_dir"], data_dir=fx4["data_dir"]
    )
    assert rep_leak["status"] == "FAIL"
    assert any(e["check"] == "split_mask_leakage" for e in rep_leak["errors"])


def test_def_b2_check_scores_rejects_hyperparams_mismatch_cross_gnn_seed_and_none_mix(tmp_path):
    """DEF-B2: Catch hyperparams drift across gnn_seeds (sum vs mean) and None vs non-None mixing."""
    fx = _build_valid_fixture(tmp_path, include_isolated_node=False)

    # Add gnn_seed=1001 with size_loss_mode='sum' while g=1000 used 'mean'
    ckpt_dir = fx["checkpoints_dir"] / "bashapes"
    ckpt_bytes2 = b"synthetic-checkpoint-binary-seed-1001-hp"
    (ckpt_dir / "gcn_s1001.pt").write_bytes(ckpt_bytes2)
    sha2 = _sha256_bytes(ckpt_bytes2)
    (ckpt_dir / "gcn_s1001.json").write_text(
        json.dumps({"gnn_seed": 1001, "sha256": sha2}), encoding="utf-8"
    )
    seed_dir2 = fx["scores_dir"] / "gcn_s1001"
    seed_dir2.mkdir(parents=True)
    df2 = pd.read_csv(fx["gnn0"])
    df2["gnn_seed"] = 1001
    df2["run_id"] = "bashapes_gcn_g1001_gnnexplainer_e0_test"
    df2["checkpoint_sha256"] = sha2
    gnn2_csv = seed_dir2 / "gnnexplainer_e0.csv"
    df2.to_csv(gnn2_csv, index=False)

    hp_sum = {
        "epochs": 200,
        "lr": 0.01,
        "coeff_size": 0.005,
        "coeff_ent": 0.1,
        "size_loss_mode": "sum",
    }
    log2_path = seed_dir2 / "run_log.json"
    log2_path.write_text(
        json.dumps(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1001,
                "split": "test",
                "checkpoint_sha256": sha2,
                "node_ids": [0, 1],
                "hyperparams": {"gnnexplainer": hp_sum},
                "runs_by_id": {
                    "bashapes_gcn_g1001_gnnexplainer_e0_test": {
                        "run_id": "bashapes_gcn_g1001_gnnexplainer_e0_test",
                        "method": "gnnexplainer",
                        "explainer_seed": 0,
                        "hyperparams": hp_sum,
                        "csv_file": gnn2_csv.name,
                        "csv_sha256": check_scores.sha256_file(gnn2_csv),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rep_cross = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_cross["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep_cross["errors"])

    # Also catch mixing None and non-None hyperparams across runs
    log2_obj = json.loads(log2_path.read_text(encoding="utf-8"))
    log2_obj["runs_by_id"]["bashapes_gcn_g1001_gnnexplainer_e0_test"]["hyperparams"] = None
    log2_path.write_text(json.dumps(log2_obj), encoding="utf-8")
    rep_none = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_none["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep_none["errors"])

    # Also test fallback to top-level hyperparams when runs_by_id omits 'hyperparams' key
    del log2_obj["runs_by_id"]["bashapes_gcn_g1001_gnnexplainer_e0_test"]["hyperparams"]
    log2_path.write_text(json.dumps(log2_obj), encoding="utf-8")
    rep_fallback = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_fallback["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep_fallback["errors"])


def test_def_b3_check_scores_rejects_boolean_floats_pure_and_mixed_with_nan(tmp_path):
    """DEF-B3a: Reject pure boolean and bool+NaN mixed columns in importance_score and runtime_sec."""
    # Direct helper checks
    assert not check_scores._is_finite_float_series(pd.Series([True, False]))[0]
    assert not check_scores._is_finite_float_series(pd.Series([True, float("nan")], dtype=object))[0]
    assert not check_scores._is_finite_float_series(pd.Series(["true", "0.5"]))[0]
    assert not check_scores._is_finite_float_series(pd.Series(["FALSE", "0.5"]), non_negative=True)[0]

    # Case 1: Pure boolean importance_score and runtime_sec in CSV
    fx1 = _build_valid_fixture(tmp_path / "pure_bool", include_isolated_node=False)
    df1 = pd.read_csv(fx1["gnn0"])
    df1["importance_score"] = True
    df1["runtime_sec"] = False
    df1.to_csv(fx1["gnn0"], index=False)
    rep1 = check_scores.validate_scores_dir(
        fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
    )
    assert rep1["status"] == "FAIL"
    checks1 = {e["check"] for e in rep1["errors"]}
    assert "finite_importance_score" in checks1
    assert "runtime_sec" in checks1

    # Case 2: Mixed True (on success rows) + NaN (on empty_candidates row) -> object dtype with Python bool
    fx2 = _build_valid_fixture(tmp_path / "mixed_bool_nan", include_isolated_node=True)
    df2 = pd.read_csv(fx2["gnn0"])
    df2["importance_score"] = df2["importance_score"].astype(object)
    df2.loc[df2["status"] == "success", "importance_score"] = True
    df2.to_csv(fx2["gnn0"], index=False)
    rep2 = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep2["status"] == "FAIL"
    assert any(e["check"] == "finite_importance_score" for e in rep2["errors"])


def test_def_b3_check_scores_rejects_placeholder_garbage_strings_and_booleans(tmp_path):
    """DEF-B3b: Reject non-empty garbage strings like GARBAGE_NOT_NAN or bool on placeholder importance_score."""
    fx1 = _build_valid_fixture(tmp_path / "ec_garbage", include_isolated_node=True)
    df1 = pd.read_csv(fx1["gnn0"])
    df1["importance_score"] = df1["importance_score"].astype(object)
    df1.loc[df1["status"] == "empty_candidates", "importance_score"] = "GARBAGE_NOT_NAN"
    df1.to_csv(fx1["gnn0"], index=False)
    rep1 = check_scores.validate_scores_dir(
        fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
    )
    assert rep1["status"] == "FAIL"
    assert any(e["check"] == "placeholder_row_invalid" for e in rep1["errors"])

    # Also verify failed row with garbage string is rejected even under allow_failed=True
    fx2 = _build_valid_fixture(tmp_path / "failed_garbage", include_isolated_node=False)
    df2 = pd.read_csv(fx2["gnn0"])
    failed_row = df2[df2["node_id"] == 0].iloc[:1].copy()
    failed_row["edge_id"] = -1
    failed_row["u"] = -1
    failed_row["v"] = -1
    failed_row["status"] = "failed"
    df2_new = pd.concat([failed_row, df2[df2["node_id"] != 0]], ignore_index=True)
    df2_new["importance_score"] = df2_new["importance_score"].astype(object)
    df2_new.loc[df2_new["status"] == "failed", "importance_score"] = "GARBAGE_NOT_NAN"
    df2_new.to_csv(fx2["gnn0"], index=False)
    rep2 = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"], allow_failed=True
    )
    assert rep2["status"] == "FAIL"
    assert any(e["check"] == "placeholder_row_invalid" for e in rep2["errors"])


def test_def_b3_check_scores_rejects_mixed_split_and_run_log_split_mismatch(tmp_path):
    """DEF-B3c: Reject mixed val/test splits across files and CSV split != run_log split."""
    fx = _build_valid_fixture(tmp_path / "mixed_split", include_isolated_node=False)
    (fx["data_dir"] / "bashapes" / "val_nodes.json").write_text(json.dumps({"nodes": [0, 1]}), encoding="utf-8")

    # Add gnn_seed=1001 with split='val' while g=1000 has split='test'
    ckpt_dir = fx["checkpoints_dir"] / "bashapes"
    ckpt_bytes2 = b"synthetic-checkpoint-binary-seed-1001-split"
    (ckpt_dir / "gcn_s1001.pt").write_bytes(ckpt_bytes2)
    sha2 = _sha256_bytes(ckpt_bytes2)
    (ckpt_dir / "gcn_s1001.json").write_text(
        json.dumps({"gnn_seed": 1001, "sha256": sha2}), encoding="utf-8"
    )
    seed_dir2 = fx["scores_dir"] / "gcn_s1001"
    seed_dir2.mkdir(parents=True)
    df2 = pd.read_csv(fx["sal0"])
    df2["gnn_seed"] = 1001
    df2["split"] = "val"
    df2["run_id"] = "bashapes_gcn_g1001_saliency_e0_val"
    df2["checkpoint_sha256"] = sha2
    sal2_csv = seed_dir2 / "saliency_e0.csv"
    df2.to_csv(sal2_csv, index=False)
    (seed_dir2 / "run_log.json").write_text(
        json.dumps(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1001,
                "split": "val",
                "checkpoint_sha256": sha2,
                "node_ids": [0, 1],
                "runs_by_id": {
                    "bashapes_gcn_g1001_saliency_e0_val": {
                        "run_id": "bashapes_gcn_g1001_saliency_e0_val",
                        "method": "saliency",
                        "explainer_seed": 0,
                        "hyperparams": {},
                        "csv_file": sal2_csv.name,
                        "csv_sha256": check_scores.sha256_file(sal2_csv),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rep_mix = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_mix["status"] == "FAIL"
    assert any(e["check"] == "mixed_split" for e in rep_mix["errors"])

    # Also test CSV split='test' vs run_log split='val' in the same folder
    fx2 = _build_valid_fixture(tmp_path / "log_split_mismatch", include_isolated_node=False)
    log_path = fx2["seed_dir"] / "run_log.json"
    log_obj = json.loads(log_path.read_text(encoding="utf-8"))
    log_obj["split"] = "val"
    log_path.write_text(json.dumps(log_obj), encoding="utf-8")
    rep_sp = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep_sp["status"] == "FAIL"
    assert any(e["check"] == "run_log_metadata_mismatch" for e in rep_sp["errors"])


def test_def_b5_check_scores_rejects_empty_run_log_missing_logged_csv_unlogged_csv_and_dir_metadata_mismatch(tmp_path):
    """DEF-B5: Enforce minimum run_log.json schema and 2-way reconciliation with CSVs on disk."""
    # 1. Empty {} run_log.json must FAIL with check='run_log'
    fx_empty = _build_valid_fixture(tmp_path / "empty_log", include_isolated_node=False)
    (fx_empty["seed_dir"] / "run_log.json").write_text("{}", encoding="utf-8")
    rep_empty = check_scores.validate_scores_dir(
        fx_empty["scores_dir"], checkpoints_dir=fx_empty["checkpoints_dir"], data_dir=fx_empty["data_dir"]
    )
    assert rep_empty["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_empty["errors"])

    # 2. Deleted CSV that is still declared in run_log.json['runs_by_id'] -> missing_logged_csv
    fx_del = _build_valid_fixture(tmp_path / "deleted_csv", include_isolated_node=False)
    fx_del["gnn0"].unlink()
    rep_del = check_scores.validate_scores_dir(
        fx_del["scores_dir"], checkpoints_dir=fx_del["checkpoints_dir"], data_dir=fx_del["data_dir"]
    )
    assert rep_del["status"] == "FAIL"
    assert any(e["check"] == "missing_logged_csv" for e in rep_del["errors"])

    # 3. Extra CSV on disk not declared in run_log.json['runs_by_id'] -> unlogged_csv
    fx_extra = _build_valid_fixture(tmp_path / "extra_csv", include_isolated_node=False)
    df_extra = pd.read_csv(fx_extra["gnn0"])
    df_extra["explainer_seed"] = 1
    df_extra["seed"] = 1
    df_extra["run_id"] = "bashapes_gcn_g1000_gnnexplainer_e1_test"
    df_extra.to_csv(fx_extra["seed_dir"] / "gnnexplainer_e1.csv", index=False)
    rep_extra = check_scores.validate_scores_dir(
        fx_extra["scores_dir"], checkpoints_dir=fx_extra["checkpoints_dir"], data_dir=fx_extra["data_dir"]
    )
    assert rep_extra["status"] == "FAIL"
    assert any(e["check"] == "unlogged_csv" for e in rep_extra["errors"])

    # 4. CSV gnn_seed differs from its folder's run_log.json -> run_log_metadata_mismatch
    fx_dir = _build_valid_fixture(tmp_path / "dir_mismatch", include_isolated_node=False)
    log_path = fx_dir["seed_dir"] / "run_log.json"
    log_obj = json.loads(log_path.read_text(encoding="utf-8"))
    log_obj["gnn_seed"] = 1009
    log_path.write_text(json.dumps(log_obj), encoding="utf-8")
    rep_dir = check_scores.validate_scores_dir(
        fx_dir["scores_dir"], checkpoints_dir=fx_dir["checkpoints_dir"], data_dir=fx_dir["data_dir"]
    )
    assert rep_dir["status"] == "FAIL"
    assert any(e["check"] == "run_log_metadata_mismatch" for e in rep_dir["errors"])


def test_def_b6_check_scores_allow_failed_cross_checks_target_class_regardless_of_file_order(tmp_path):
    """DEF-B6: Failed node with wrong target_class must fail cross_method_target_class even if its CSV sorts FIRST alphabetically."""
    # Case A: gnnexplainer_e0.csv (sorts BEFORE saliency_e0.csv!) has failed node 0 with corrupted target_class=99
    fx_a = _build_valid_fixture(tmp_path / "failed_first", include_isolated_node=False)
    df_g = pd.read_csv(fx_a["gnn0"])
    failed_row = df_g[df_g["node_id"] == 0].iloc[:1].copy()
    failed_row["edge_id"] = -1
    failed_row["u"] = -1
    failed_row["v"] = -1
    failed_row["importance_score"] = float("nan")
    failed_row["status"] = "failed"
    failed_row["target_class"] = 99
    df_g_new = pd.concat([failed_row, df_g[df_g["node_id"] != 0]], ignore_index=True)
    df_g_new.to_csv(fx_a["gnn0"], index=False)
    log_a = fx_a["seed_dir"] / "run_log.json"
    obj_a = json.loads(log_a.read_text(encoding="utf-8"))
    obj_a["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = check_scores.sha256_file(fx_a["gnn0"])
    log_a.write_text(json.dumps(obj_a), encoding="utf-8")

    rep_a = check_scores.validate_scores_dir(
        fx_a["scores_dir"], checkpoints_dir=fx_a["checkpoints_dir"], data_dir=fx_a["data_dir"], allow_failed=True
    )
    assert rep_a["status"] == "FAIL"
    assert any(e["check"] == "cross_method_target_class" for e in rep_a["errors"])

    # Case B: saliency_e1.csv (sorts LAST!) has failed node 0 with corrupted target_class=99
    fx_b = _build_valid_fixture(tmp_path / "failed_last", include_isolated_node=False)
    df_s = pd.read_csv(fx_b["sal1"])
    failed_row_b = df_s[df_s["node_id"] == 0].iloc[:1].copy()
    failed_row_b["edge_id"] = -1
    failed_row_b["u"] = -1
    failed_row_b["v"] = -1
    failed_row_b["importance_score"] = float("nan")
    failed_row_b["status"] = "failed"
    failed_row_b["target_class"] = 99
    df_s_new = pd.concat([failed_row_b, df_s[df_s["node_id"] != 0]], ignore_index=True)
    df_s_new.to_csv(fx_b["sal1"], index=False)
    log_b = fx_b["seed_dir"] / "run_log.json"
    obj_b = json.loads(log_b.read_text(encoding="utf-8"))
    obj_b["runs_by_id"]["bashapes_gcn_g1000_saliency_e1_test"]["csv_sha256"] = check_scores.sha256_file(fx_b["sal1"])
    log_b.write_text(json.dumps(obj_b), encoding="utf-8")

    rep_b = check_scores.validate_scores_dir(
        fx_b["scores_dir"], checkpoints_dir=fx_b["checkpoints_dir"], data_dir=fx_b["data_dir"], allow_failed=True
    )
    assert rep_b["status"] == "FAIL"
    assert any(e["check"] == "cross_method_target_class" for e in rep_b["errors"])


def test_check_scores_hardens_runs_list_shape_b_renamed_csv_dup_log_and_na_strings(tmp_path):
    """Verify per-seed runs list, Shape B aggregate run_log, renamed CSV, duplicate seed log hash, bad top hyperparams, and NA/null placeholder strings."""
    import shutil

    # 1. Per-seed run_log.json using 'runs' list: catches hyperparams_mismatch AND missing_logged_csv
    fx1 = _build_valid_fixture(tmp_path / "per_seed_runs_list")
    lp1 = fx1["seed_dir"] / "run_log.json"
    lo1 = json.loads(lp1.read_text(encoding="utf-8"))
    g0 = pd.read_csv(fx1["gnn0"])
    g1 = g0.copy()
    g1["explainer_seed"] = 1
    g1["seed"] = 1
    g1["run_id"] = "bashapes_gcn_g1000_gnnexplainer_e1_test"
    p1 = fx1["seed_dir"] / "gnnexplainer_e1.csv"
    g1.to_csv(p1, index=False)
    hp_mean = lo1["hyperparams"]["gnnexplainer"]
    hp_sum = dict(hp_mean, size_loss_mode="sum")
    runs_list = list(lo1["runs_by_id"].values()) + [
        {
            "run_id": "bashapes_gcn_g1000_gnnexplainer_e1_test",
            "method": "gnnexplainer",
            "explainer_seed": 1,
            "hyperparams": hp_sum,
            "csv_file": p1.name,
            "csv_sha256": check_scores.sha256_file(p1),
        }
    ]
    del lo1["runs_by_id"]
    lo1["runs"] = runs_list
    lp1.write_text(json.dumps(lo1), encoding="utf-8")

    rep1_hp = check_scores.validate_scores_dir(
        fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
    )
    assert rep1_hp["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep1_hp["errors"])

    p1.unlink()
    rep1_del = check_scores.validate_scores_dir(
        fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
    )
    assert rep1_del["status"] == "FAIL"
    assert any(e["check"] == "missing_logged_csv" for e in rep1_del["errors"])

    # 2. Aggregate root run_log.json (Shape B {"runs": [seed_log]}): PASS when valid, FAIL on deleted/unlogged CSV
    fx2 = _build_valid_fixture(tmp_path / "shape_b")
    lp2 = fx2["seed_dir"] / "run_log.json"
    seed_log = json.loads(lp2.read_text(encoding="utf-8"))
    lp2.unlink()
    root_log = fx2["scores_dir"] / "run_log.json"
    root_log.write_text(json.dumps({"runs": [seed_log]}), encoding="utf-8")

    rep2_ok = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep2_ok["status"] == "PASS", rep2_ok["errors"]

    fx2["gnn0"].unlink()
    rep2_del = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep2_del["status"] == "FAIL"
    assert any(e["check"] == "missing_logged_csv" for e in rep2_del["errors"])

    # 2b. Shape B flat single-run list catches unlogged_csv AND hyperparams_mismatch
    fx2b = _build_valid_fixture(tmp_path / "shape_b_flat")
    lp2b = fx2b["seed_dir"] / "run_log.json"
    seed_log_b = json.loads(lp2b.read_text(encoding="utf-8"))
    lp2b.unlink()
    g1_b = pd.read_csv(fx2b["gnn0"])
    g1_b["explainer_seed"] = 1
    g1_b["seed"] = 1
    g1_b["run_id"] = "bashapes_gcn_g1000_gnnexplainer_e1_test"
    p1_b = fx2b["seed_dir"] / "gnnexplainer_e1.csv"
    g1_b.to_csv(p1_b, index=False)
    flat_runs = []
    for rid, rinfo in seed_log_b["runs_by_id"].items():
        flat_runs.append(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1000,
                "split": "test",
                "checkpoint_sha256": fx2b["ckpt_sha"],
                "run_id": rid,
                "method": rinfo["method"],
                "hyperparams": rinfo["hyperparams"],
                "csv_file": f"gcn_s1000/{rinfo['csv_file']}",
                "csv_sha256": rinfo["csv_sha256"],
            }
        )
    # Without p1_b declared -> unlogged_csv
    root_log_b = fx2b["scores_dir"] / "run_log.json"
    root_log_b.write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
    rep2b_un = check_scores.validate_scores_dir(
        fx2b["scores_dir"], checkpoints_dir=fx2b["checkpoints_dir"], data_dir=fx2b["data_dir"]
    )
    assert rep2b_un["status"] == "FAIL"
    assert any(e["check"] == "unlogged_csv" for e in rep2b_un["errors"])

    # Declare p1_b with conflicting size_loss_mode='sum' -> hyperparams_mismatch
    flat_runs.append(
        {
            "dataset": "bashapes",
            "arch": "gcn",
            "gnn_seed": 1000,
            "split": "test",
            "checkpoint_sha256": fx2b["ckpt_sha"],
            "run_id": "bashapes_gcn_g1000_gnnexplainer_e1_test",
            "method": "gnnexplainer",
            "hyperparams": dict(seed_log_b["hyperparams"]["gnnexplainer"], size_loss_mode="sum"),
            "csv_file": f"gcn_s1000/{p1_b.name}",
            "csv_sha256": check_scores.sha256_file(p1_b),
        }
    )
    root_log_b.write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
    rep2b_hp = check_scores.validate_scores_dir(
        fx2b["scores_dir"], checkpoints_dir=fx2b["checkpoints_dir"], data_dir=fx2b["data_dir"]
    )
    assert rep2b_hp["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep2b_hp["errors"])

    # 3. Renamed CSV stealing logged run_id triggers BOTH missing_logged_csv and unlogged_csv
    fx3 = _build_valid_fixture(tmp_path / "renamed_csv")
    shutil.copyfile(fx3["gnn0"], fx3["seed_dir"] / "rogue.csv")
    fx3["gnn0"].unlink()
    rep3 = check_scores.validate_scores_dir(
        fx3["scores_dir"], checkpoints_dir=fx3["checkpoints_dir"], data_dir=fx3["data_dir"]
    )
    assert rep3["status"] == "FAIL"
    checks3 = {e["check"] for e in rep3["errors"]}
    assert "missing_logged_csv" in checks3
    assert "unlogged_csv" in checks3

    # 4. Duplicate (dataset, arch, split, gnn_seed) run_log in sibling folder cannot mask bad checkpoint_sha256
    fx4 = _build_valid_fixture(tmp_path / "dup_seed_log")
    lp4 = fx4["seed_dir"] / "run_log.json"
    lo4 = json.loads(lp4.read_text(encoding="utf-8"))
    lo4_good = json.loads(json.dumps(lo4))
    lo4["checkpoint_sha256"] = "f" * 64
    lp4.write_text(json.dumps(lo4), encoding="utf-8")
    d_dup = fx4["scores_dir"] / "gcn_s1000_zz"
    d_dup.mkdir()
    df_s2 = pd.read_csv(fx4["sal0"])
    df_s2["explainer_seed"] = 2
    df_s2["seed"] = 2
    df_s2["run_id"] = "bashapes_gcn_g1000_saliency_e2_test"
    s2_csv = d_dup / "saliency_e2.csv"
    df_s2.to_csv(s2_csv, index=False)
    lo4_good["runs_by_id"] = {
        "bashapes_gcn_g1000_saliency_e2_test": {
            "run_id": "bashapes_gcn_g1000_saliency_e2_test",
            "method": "saliency",
            "explainer_seed": 2,
            "hyperparams": {},
            "csv_file": s2_csv.name,
            "csv_sha256": check_scores.sha256_file(s2_csv),
        }
    }
    (d_dup / "run_log.json").write_text(json.dumps(lo4_good), encoding="utf-8")
    rep4 = check_scores.validate_scores_dir(
        fx4["scores_dir"], checkpoints_dir=fx4["checkpoints_dir"], data_dir=fx4["data_dir"]
    )
    assert rep4["status"] == "FAIL"
    assert any(e["check"] == "run_log_hash_mismatch" for e in rep4["errors"])

    # 5. Non-dict top-level hyperparams must fail check='run_log'
    fx5 = _build_valid_fixture(tmp_path / "bad_top_hp")
    lp5 = fx5["seed_dir"] / "run_log.json"
    lo5 = json.loads(lp5.read_text(encoding="utf-8"))
    lo5["hyperparams"] = "NOT_A_DICT"
    lp5.write_text(json.dumps(lo5), encoding="utf-8")
    rep5 = check_scores.validate_scores_dir(
        fx5["scores_dir"], checkpoints_dir=fx5["checkpoints_dir"], data_dir=fx5["data_dir"]
    )
    assert rep5["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep5["errors"])

    # 6. Placeholder importance_score with literal "null" or "NA" must fail placeholder_row_invalid
    for bad_token in ("null", "NA"):
        fx6 = _build_valid_fixture(tmp_path / f"na_tok_{bad_token}", include_isolated_node=True)
        df6 = pd.read_csv(fx6["gnn0"])
        df6["importance_score"] = df6["importance_score"].astype(object)
        df6.loc[df6["status"] == "empty_candidates", "importance_score"] = bad_token
        df6.to_csv(fx6["gnn0"], index=False)
        lp6 = fx6["seed_dir"] / "run_log.json"
        lo6 = json.loads(lp6.read_text(encoding="utf-8"))
        lo6["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = check_scores.sha256_file(fx6["gnn0"])
        lp6.write_text(json.dumps(lo6), encoding="utf-8")
        rep6 = check_scores.validate_scores_dir(
            fx6["scores_dir"], checkpoints_dir=fx6["checkpoints_dir"], data_dir=fx6["data_dir"]
        )
        assert rep6["status"] == "FAIL", f"Expected FAIL for placeholder token {bad_token!r}"
        assert any(e["check"] == "placeholder_row_invalid" for e in rep6["errors"])


def test_def_b7_rejects_empty_invalid_and_ambiguous_run_containers_shape_a_and_b(tmp_path):
    """DEF-B7: Block empty/invalid/ambiguous runs_by_id & runs containers in both Shape A and Shape B."""
    # Direct _normalize_runs_dict checks
    direct_errs = []
    add_err = lambda f, c, m, lines=None: direct_errs.append((f, c, m))
    assert check_scores._normalize_runs_dict({"runs_by_id": {}}, "log.json", add_err) is None
    assert check_scores._normalize_runs_dict({"runs": []}, "log.json", add_err) is None
    assert (
        check_scores._normalize_runs_dict(
            {"runs_by_id": {"a": {}}, "runs": [{"run_id": "a"}]}, "log.json", add_err
        )
        is None
    )
    assert len(direct_errs) == 3 and all(c == "run_log" for _, c, _ in direct_errs)

    # Shape A cases: empty runs_by_id, non-dict runs_by_id, empty runs, non-list runs, ambiguous both
    shape_a_mutations = [
        ("empty_rbi", lambda o: (o.update({"runs_by_id": {}}), o.pop("runs", None))),
        ("none_rbi", lambda o: (o.update({"runs_by_id": None}), o.pop("runs", None))),
        ("empty_runs", lambda o: (o.pop("runs_by_id", None), o.update({"runs": []}))),
        ("bad_runs", lambda o: (o.pop("runs_by_id", None), o.update({"runs": "not_a_list"}))),
        ("both_valid", lambda o: o.update({"runs": list(o["runs_by_id"].values())})),
        ("empty_rbi_and_valid_runs", lambda o: o.update({"runs": list(o["runs_by_id"].values()), "runs_by_id": {}})),
    ]
    for label, mutate in shape_a_mutations:
        fx = _build_valid_fixture(tmp_path / f"shape_a_{label}")
        lp = fx["seed_dir"] / "run_log.json"
        obj = json.loads(lp.read_text(encoding="utf-8"))
        mutate(obj)
        lp.write_text(json.dumps(obj), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL", f"Shape A case {label} unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep["errors"]), f"Shape A {label}: {rep['errors']}"

    # Shape B root ambiguous (both runs and runs_by_id at root)
    fx_root = _build_valid_fixture(tmp_path / "shape_b_root_both")
    lp_root = fx_root["seed_dir"] / "run_log.json"
    seed_log_r = json.loads(lp_root.read_text(encoding="utf-8"))
    lp_root.unlink()
    (fx_root["scores_dir"] / "run_log.json").write_text(
        json.dumps({"runs": [seed_log_r], "runs_by_id": {}}), encoding="utf-8"
    )
    rep_root = check_scores.validate_scores_dir(
        fx_root["scores_dir"], checkpoints_dir=fx_root["checkpoints_dir"], data_dir=fx_root["data_dir"]
    )
    assert rep_root["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_root["errors"])

    # Shape B item cases: empty/non-dict runs_by_id, empty/non-list runs, both present
    for label, mutate in shape_a_mutations:
        fx = _build_valid_fixture(tmp_path / f"shape_b_{label}")
        lp = fx["seed_dir"] / "run_log.json"
        item = json.loads(lp.read_text(encoding="utf-8"))
        lp.unlink()
        mutate(item)
        (fx["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": [item]}), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL", f"Shape B case {label} unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep["errors"]), f"Shape B {label}: {rep['errors']}"


def test_def_b8_validates_top_level_hyperparams_without_breaking_shape_b_flat(tmp_path):
    """DEF-B8: Catch top-level hyperparams drift/garbage on Shape A & Shape B nested while keeping Shape B flat PASS."""
    # 1. Shape A: top-level hyperparams['gnnexplainer'] disagrees with runs_by_id[*]['hyperparams']
    fx1 = _build_valid_fixture(tmp_path / "shape_a_top_hp_drift")
    lp1 = fx1["seed_dir"] / "run_log.json"
    lo1 = json.loads(lp1.read_text(encoding="utf-8"))
    lo1["hyperparams"]["gnnexplainer"] = dict(lo1["hyperparams"]["gnnexplainer"], size_loss_mode="sum")
    lp1.write_text(json.dumps(lo1), encoding="utf-8")
    rep1 = check_scores.validate_scores_dir(
        fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
    )
    assert rep1["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep1["errors"])

    # 2. Shape A: non-dict value inside top-level hyperparams or unknown method key -> check='run_log'
    for bad_top, tag in (({"gnnexplainer": "garbage"}, "nondict_val"), ({"unknown_method": {}}, "bad_method")):
        fx_b = _build_valid_fixture(tmp_path / f"shape_a_top_{tag}")
        lp_b = fx_b["seed_dir"] / "run_log.json"
        lo_b = json.loads(lp_b.read_text(encoding="utf-8"))
        lo_b["hyperparams"] = bad_top
        lp_b.write_text(json.dumps(lo_b), encoding="utf-8")
        rep_b = check_scores.validate_scores_dir(
            fx_b["scores_dir"], checkpoints_dir=fx_b["checkpoints_dir"], data_dir=fx_b["data_dir"]
        )
        assert rep_b["status"] == "FAIL"
        assert any(e["check"] == "run_log" for e in rep_b["errors"])

    # 3. Shape B nested: top-level hyperparams conflict with runs_by_id -> hyperparams_mismatch
    fx3 = _build_valid_fixture(tmp_path / "shape_b_nested_top_hp")
    lp3 = fx3["seed_dir"] / "run_log.json"
    seed_log3 = json.loads(lp3.read_text(encoding="utf-8"))
    lp3.unlink()
    seed_log3["hyperparams"]["gnnexplainer"] = dict(
        seed_log3["hyperparams"]["gnnexplainer"], size_loss_mode="sum"
    )
    (fx3["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": [seed_log3]}), encoding="utf-8")
    rep3 = check_scores.validate_scores_dir(
        fx3["scores_dir"], checkpoints_dir=fx3["checkpoints_dir"], data_dir=fx3["data_dir"]
    )
    assert rep3["status"] == "FAIL"
    assert any(e["check"] == "hyperparams_mismatch" for e in rep3["errors"])

    # 4. Shape B flat single-run: content['hyperparams'] is method's param dict -> MUST PASS!
    fx4 = _build_valid_fixture(tmp_path / "shape_b_flat_valid")
    lp4 = fx4["seed_dir"] / "run_log.json"
    seed_log4 = json.loads(lp4.read_text(encoding="utf-8"))
    lp4.unlink()
    flat_runs = []
    for rid, rinfo in seed_log4["runs_by_id"].items():
        flat_runs.append(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1000,
                "split": "test",
                "checkpoint_sha256": fx4["ckpt_sha"],
                "run_id": rid,
                "method": rinfo["method"],
                "hyperparams": rinfo["hyperparams"],
                "csv_file": f"gcn_s1000/{rinfo['csv_file']}",
                "csv_sha256": rinfo["csv_sha256"],
            }
        )
    (fx4["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
    rep4 = check_scores.validate_scores_dir(
        fx4["scores_dir"], checkpoints_dir=fx4["checkpoints_dir"], data_dir=fx4["data_dir"]
    )
    assert rep4["status"] == "PASS", rep4["errors"]


def test_def_b10_no_crash_on_invalid_placeholder_and_numeric_edge_sort(tmp_path):
    """DEF-B10: Do not crash when placeholder row has invalid edge_id; sort edge_id numerically even with string/float formatting."""
    # Case A: invalid string in placeholder edge_id makes column object dtype -> must FAIL cleanly without ValueError
    fx_a = _build_valid_fixture(tmp_path / "bad_placeholder_edge", include_isolated_node=True)
    df_a = pd.read_csv(fx_a["gnn0"])
    df_a["edge_id"] = df_a["edge_id"].astype(object)
    df_a["u"] = df_a["u"].astype(object)
    df_a.loc[df_a["status"] == "empty_candidates", "edge_id"] = "BAD_EDGE"
    df_a.loc[df_a["status"] == "empty_candidates", "u"] = "BAD_U"
    df_a.to_csv(fx_a["gnn0"], index=False)
    lp_a = fx_a["seed_dir"] / "run_log.json"
    lo_a = json.loads(lp_a.read_text(encoding="utf-8"))
    lo_a["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = check_scores.sha256_file(fx_a["gnn0"])
    lp_a.write_text(json.dumps(lo_a), encoding="utf-8")

    rep_a = check_scores.validate_scores_dir(
        fx_a["scores_dir"], checkpoints_dir=fx_a["checkpoints_dir"], data_dir=fx_a["data_dir"]
    )
    assert rep_a["status"] == "FAIL"
    assert any(e["check"] == "placeholder_row_invalid" for e in rep_a["errors"])

    # Case B: >= 12 edges with string/float formatting ("0.0", "10", "11") + empty placeholder -> numeric sort PASS!
    data_dir = tmp_path / "big_sort" / "data"
    ds_dir = data_dir / "bashapes"
    ds_dir.mkdir(parents=True)
    # 13-node star around node 0 -> 12 undirected candidate edges (edge_ids 0..11) + isolated node 13
    src = [0] * 12 + list(range(1, 13))
    dst = list(range(1, 13)) + [0] * 12
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.ones(14, 4)
    nodes = [0, 13]
    torch.save({"x": x, "edge_index": edge_index}, ds_dir / "data.pt")
    (ds_dir / "test_nodes.json").write_text(json.dumps({"nodes": nodes}), encoding="utf-8")

    ckpt_dir = tmp_path / "big_sort" / "checkpoints" / "bashapes"
    ckpt_dir.mkdir(parents=True)
    ckpt_bytes = b"star-graph-ckpt"
    (ckpt_dir / "gcn_s1000.pt").write_bytes(ckpt_bytes)
    ckpt_sha = _sha256_bytes(ckpt_bytes)
    (ckpt_dir / "gcn_s1000.json").write_text(json.dumps({"gnn_seed": 1000, "sha256": ckpt_sha}), encoding="utf-8")

    scores_dir = tmp_path / "big_sort" / "scores" / "bashapes"
    seed_dir = scores_dir / "gcn_s1000"
    seed_dir.mkdir(parents=True)

    cand0 = extract_candidate_edges(edge_index, 0, num_hops=2, num_nodes=14)
    assert cand0.num_candidate_edges == 12
    rows = []
    rid = "bashapes_gcn_g1000_saliency_e0_test"
    # Shuffle order and mix "0.0" with integer strings so Pandas loads object/float dtype
    for idx_i, (eid, (u, v)) in enumerate(reversed(list(zip(cand0.edge_ids, cand0.candidate_pairs)))):
        rows.append(
            {
                "run_id": rid,
                "dataset": "bashapes",
                "split": "test",
                "arch": "gcn",
                "gnn_seed": 1000,
                "checkpoint_sha256": ckpt_sha,
                "method": "saliency",
                "explainer_seed": 0,
                "seed": 0,
                "node_id": 0,
                "target_class": 0,
                "edge_id": f"{eid}.0" if eid == 0 else str(eid),
                "u": f"{u}.0" if idx_i == 0 else str(u),
                "v": f"{v}.0" if idx_i == 0 else str(v),
                "importance_score": 0.1 * (eid + 1),
                "runtime_sec": 0.01,
                "status": "success",
            }
        )
    rows.append(
        {
            "run_id": rid,
            "dataset": "bashapes",
            "split": "test",
            "arch": "gcn",
            "gnn_seed": 1000,
            "checkpoint_sha256": ckpt_sha,
            "method": "saliency",
            "explainer_seed": 0,
            "seed": 0,
            "node_id": 13,
            "target_class": 1,
            "edge_id": "",
            "u": "",
            "v": "",
            "importance_score": "",
            "runtime_sec": 0.001,
            "status": "empty_candidates",
        }
    )
    sal_csv = seed_dir / "saliency_e0.csv"
    pd.DataFrame(rows, columns=check_scores.REQUIRED_COLUMNS).to_csv(sal_csv, index=False)
    (seed_dir / "run_log.json").write_text(
        json.dumps(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1000,
                "split": "test",
                "checkpoint_sha256": ckpt_sha,
                "node_ids": nodes,
                "runs_by_id": {
                    rid: {
                        "run_id": rid,
                        "method": "saliency",
                        "explainer_seed": 0,
                        "hyperparams": {},
                        "csv_file": sal_csv.name,
                        "csv_sha256": check_scores.sha256_file(sal_csv),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rep_b = check_scores.validate_scores_dir(
        scores_dir, checkpoints_dir=tmp_path / "big_sort" / "checkpoints", data_dir=data_dir
    )
    assert rep_b["status"] == "PASS", rep_b["errors"]


def test_def_b11_load_expected_nodes_no_crash_on_malformed_node_ids_and_json(tmp_path):
    """DEF-B11: Malformed run_log node_ids or node JSON files must return clean errors, never TypeError/ValueError."""
    # 1. Direct _load_expected_nodes checks on malformed run_log_entry['node_ids']
    for bad_nids in (None, [], 123, "0,1", [True, False], ["not_int"], [-1]):
        nodes, err = check_scores._load_expected_nodes(
            dataset="bashapes",
            split="val",
            data_dir=tmp_path / "empty_data",
            expected_nodes_file=None,
            max_nodes=None,
            run_log_entry={"node_ids": bad_nids},
        )
        assert nodes is None and isinstance(err, str) and len(err) > 0

    # 2. Direct _load_expected_nodes checks on malformed expected_nodes_file JSON
    for idx, raw_text in enumerate(
        (
            "{not valid json",
            json.dumps({}),
            json.dumps({"nodes": None}),
            json.dumps({"nodes": ["abc"]}),
            json.dumps({"nodes": [True]}),
            json.dumps({"nodes": [-2]}),
            json.dumps({"nodes": []}),
            json.dumps(123),
        )
    ):
        bad_file = tmp_path / f"bad_nodes_{idx}.json"
        bad_file.write_text(raw_text, encoding="utf-8")
        nodes, err = check_scores._load_expected_nodes(
            dataset="bashapes",
            split="test",
            data_dir=tmp_path,
            expected_nodes_file=bad_file,
            max_nodes=None,
            run_log_entry=None,
        )
        assert nodes is None and isinstance(err, str) and len(err) > 0

    # 3. End-to-end validate_scores_dir with bad run_log node_ids -> check='run_log' (no crash)
    for tag, bad_nids in (("null", None), ("bool", [True, False]), ("str", ["a"]), ("empty", [])):
        fx = _build_valid_fixture(tmp_path / f"bad_log_nids_{tag}")
        lp = fx["seed_dir"] / "run_log.json"
        lo = json.loads(lp.read_text(encoding="utf-8"))
        lo["node_ids"] = bad_nids
        lp.write_text(json.dumps(lo), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL"
        assert any(e["check"] == "run_log" for e in rep["errors"])

    # 4. End-to-end validate_scores_dir with malformed test_nodes.json -> check='expected_nodes_lookup' (no crash)
    for tag, payload in (("null_nodes", {"nodes": None}), ("str_nodes", {"nodes": ["bad"]})):
        fx = _build_valid_fixture(tmp_path / f"bad_test_nodes_{tag}")
        (fx["data_dir"] / "bashapes" / "test_nodes.json").write_text(json.dumps(payload), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL"
        assert any(e["check"] == "expected_nodes_lookup" for e in rep["errors"])


def test_def_b12_blocks_int64_overflow_wraparound(tmp_path):
    """DEF-B12: Reject integers above 2**53 - 1 (1e20, 2**63, 1e308) to prevent int64 wraparound."""
    for bad_val in (1e20, "1e20", 9223372036854775808, "9223372036854775808", 1e308, "1e308", 9007199254740992):
        ok, bad_lines = check_scores._is_nonneg_int_series(pd.Series([bad_val]))
        assert not ok and bad_lines == [2], f"Expected rejection for overflow value {bad_val!r}"

    # Boundary 2**53 - 1 must remain valid
    ok_max, bad_max = check_scores._is_nonneg_int_series(pd.Series([9007199254740991]))
    assert ok_max and bad_max == []

    # End-to-end CSV with overflow in edge_id must fail cleanly as edge_integer_field
    fx = _build_valid_fixture(tmp_path / "overflow_csv")
    df = pd.read_csv(fx["gnn0"])
    df["edge_id"] = df["edge_id"].astype(object)
    df.loc[0, "edge_id"] = "1e20"
    df.to_csv(fx["gnn0"], index=False)
    rep = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep["status"] == "FAIL"
    assert any(e["check"] == "edge_integer_field" for e in rep["errors"])


def test_def_b13_harden_csv_sha256_empty_bool_and_skip_checkpoint_hash_still_checks_run_log(tmp_path):
    """DEF-B13 (#1 & #2): Reject falsy/invalid csv_sha256 in Shape A/B and verify run_log checkpoint_sha256 even under --skip-checkpoint-hash."""
    # 1. Invalid/falsy csv_sha256 ("", False, "bad") cannot bypass SHA-256 check in Shape A
    for tag, bad_sha in (("empty", ""), ("bool_false", False), ("bad_hex", "not-a-sha256")):
        fx = _build_valid_fixture(tmp_path / f"bad_csv_sha_a_{tag}")
        lp = fx["seed_dir"] / "run_log.json"
        lo = json.loads(lp.read_text(encoding="utf-8"))
        lo["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = bad_sha
        lp.write_text(json.dumps(lo), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL", f"Expected FAIL for Shape A csv_sha256={bad_sha!r}"
        assert any(e["check"] == "run_log" for e in rep["errors"])

    # Also test Shape B nested and Shape B flat with csv_sha256="" / False
    fx_bn = _build_valid_fixture(tmp_path / "bad_csv_sha_b_nested")
    lp_bn = fx_bn["seed_dir"] / "run_log.json"
    seed_log_bn = json.loads(lp_bn.read_text(encoding="utf-8"))
    lp_bn.unlink()
    seed_log_bn["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = ""
    (fx_bn["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": [seed_log_bn]}), encoding="utf-8")
    rep_bn = check_scores.validate_scores_dir(
        fx_bn["scores_dir"], checkpoints_dir=fx_bn["checkpoints_dir"], data_dir=fx_bn["data_dir"]
    )
    assert rep_bn["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_bn["errors"])

    fx_bf = _build_valid_fixture(tmp_path / "bad_csv_sha_b_flat")
    lp_bf = fx_bf["seed_dir"] / "run_log.json"
    seed_log_bf = json.loads(lp_bf.read_text(encoding="utf-8"))
    lp_bf.unlink()
    flat_runs = []
    for rid, rinfo in seed_log_bf["runs_by_id"].items():
        flat_runs.append(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1000,
                "split": "test",
                "checkpoint_sha256": fx_bf["ckpt_sha"],
                "run_id": rid,
                "method": rinfo["method"],
                "hyperparams": rinfo["hyperparams"],
                "csv_file": f"gcn_s1000/{rinfo['csv_file']}",
                "csv_sha256": False if "gnnexplainer" in rid else rinfo["csv_sha256"],
            }
        )
    (fx_bf["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
    rep_bf = check_scores.validate_scores_dir(
        fx_bf["scores_dir"], checkpoints_dir=fx_bf["checkpoints_dir"], data_dir=fx_bf["data_dir"]
    )
    assert rep_bf["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_bf["errors"])

    # 2. --skip-checkpoint-hash must STILL catch run_log.json checkpoint_sha256 != CSV checkpoint_sha256
    fx_skip = _build_valid_fixture(tmp_path / "skip_ckpt_still_checks_log")
    lp_skip = fx_skip["seed_dir"] / "run_log.json"
    lo_skip = json.loads(lp_skip.read_text(encoding="utf-8"))
    lo_skip["checkpoint_sha256"] = "e" * 64
    lp_skip.write_text(json.dumps(lo_skip), encoding="utf-8")
    rep_skip = check_scores.validate_scores_dir(
        fx_skip["scores_dir"],
        checkpoints_dir=tmp_path / "nonexistent_ckpts",
        data_dir=fx_skip["data_dir"],
        skip_checkpoint_hash=True,
    )
    assert rep_skip["status"] == "FAIL"
    assert any(e["check"] == "run_log_hash_mismatch" for e in rep_skip["errors"])


def test_adversarial_mixed_shape_ab_spoofed_method_flat_meta_and_giant_int_overflow(tmp_path):
    """Block mixed Shape A+B csv_sha256 conflict, spoofed method bypass, Shape B metadata/hash spoof, whitespace keys, and >309-digit OverflowError."""
    # 1. Mixed Shape A + Shape B coexisting with conflicting csv_sha256 in Shape B MUST fail
    fx_mix = _build_valid_fixture(tmp_path / "mixed_ab_sha")
    lp_mix = fx_mix["seed_dir"] / "run_log.json"
    lo_a = json.loads(lp_mix.read_text(encoding="utf-8"))
    lo_b = json.loads(json.dumps(lo_a))
    lo_b["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["csv_sha256"] = "0" * 64
    (fx_mix["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": [lo_b]}), encoding="utf-8")
    rep_mix = check_scores.validate_scores_dir(
        fx_mix["scores_dir"], checkpoints_dir=fx_mix["checkpoints_dir"], data_dir=fx_mix["data_dir"]
    )
    assert rep_mix["status"] == "FAIL"
    assert any(e["check"] == "csv_sha256_mismatch" for e in rep_mix["errors"])

    # 2. Spoofing rinfo['method'] (both unsupported and valid-but-wrong) cannot bypass hyperparams_mismatch
    for spoof_method, expected_check in (("fake_method", "run_log"), ("nearest_first", "run_log_metadata_mismatch")):
        fx_sp = _build_valid_fixture(tmp_path / f"spoof_method_{spoof_method}")
        lp_sp = fx_sp["seed_dir"] / "run_log.json"
        lo_sp = json.loads(lp_sp.read_text(encoding="utf-8"))
        lo_sp["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["method"] = spoof_method
        lo_sp["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]["hyperparams"] = {"epochs": 999}
        lp_sp.write_text(json.dumps(lo_sp), encoding="utf-8")
        rep_sp = check_scores.validate_scores_dir(
            fx_sp["scores_dir"], checkpoints_dir=fx_sp["checkpoints_dir"], data_dir=fx_sp["data_dir"]
        )
        assert rep_sp["status"] == "FAIL"
        checks_sp = {e["check"] for e in rep_sp["errors"]}
        assert expected_check in checks_sp, rep_sp["errors"]
        assert "hyperparams_mismatch" in checks_sp, rep_sp["errors"]

    # 3. Shape B flat entry spoofing gnn_seed + checkpoint_sha256 while matching run_id MUST fail (even with --skip-checkpoint-hash)
    fx_fs = _build_valid_fixture(tmp_path / "shape_b_flat_spoof")
    lp_fs = fx_fs["seed_dir"] / "run_log.json"
    lo_fs = json.loads(lp_fs.read_text(encoding="utf-8"))
    lp_fs.unlink()
    flat_spoof = []
    for rid, rinfo in lo_fs["runs_by_id"].items():
        is_gnn = "gnnexplainer" in rid
        flat_spoof.append(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1001 if is_gnn else 1000,
                "split": "test",
                "checkpoint_sha256": ("f" * 64) if is_gnn else fx_fs["ckpt_sha"],
                "run_id": rid,
                "method": rinfo["method"],
                "hyperparams": rinfo["hyperparams"],
                "csv_file": f"gcn_s1000/{rinfo['csv_file']}",
                "csv_sha256": rinfo["csv_sha256"],
            }
        )
    (fx_fs["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_spoof}), encoding="utf-8")
    for skip_flag in (False, True):
        rep_fs = check_scores.validate_scores_dir(
            fx_fs["scores_dir"],
            checkpoints_dir=fx_fs["checkpoints_dir"],
            data_dir=fx_fs["data_dir"],
            skip_checkpoint_hash=skip_flag,
        )
        assert rep_fs["status"] == "FAIL"
        checks_fs = {e["check"] for e in rep_fs["errors"]}
        assert "run_log_metadata_mismatch" in checks_fs, rep_fs["errors"]
        assert "run_log_hash_mismatch" in checks_fs, rep_fs["errors"]

    # 4. Oversized >309-digit Python int in object Series must return clean False (never OverflowError)
    giant_series = pd.Series([0, 10**400], dtype=object)
    ok_int, bad_int = check_scores._is_nonneg_int_series(giant_series)
    assert not ok_int and bad_int == [3]
    ok_flt, bad_flt = check_scores._is_finite_float_series(giant_series, non_negative=True)
    assert not ok_flt and bad_flt == [3]

    # 5. Whitespace-padded runs_by_id key and trailing-newline run_id/dataset must be rejected
    assert check_scores._extract_method_from_run_id("bashapes_gcn_g1000_saliency_e0_test\n") is None
    fx_ws = _build_valid_fixture(tmp_path / "ws_run_key")
    lp_ws = fx_ws["seed_dir"] / "run_log.json"
    lo_ws = json.loads(lp_ws.read_text(encoding="utf-8"))
    k0 = "bashapes_gcn_g1000_saliency_e0_test"
    v0 = lo_ws["runs_by_id"].pop(k0)
    v0["run_id"] = f" {k0} "
    lo_ws["runs_by_id"][f" {k0} "] = v0
    lp_ws.write_text(json.dumps(lo_ws), encoding="utf-8")
    rep_ws = check_scores.validate_scores_dir(
        fx_ws["scores_dir"], checkpoints_dir=fx_ws["checkpoints_dir"], data_dir=fx_ws["data_dir"]
    )
    assert rep_ws["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_ws["errors"])


def test_def_b14_reconciliation_rejects_traversal_nonflat_noncsv_abs_drive_and_nul_csv_file(tmp_path):
    """DEF-B14: Reject traversal, non-flat, non-CSV, absolute/drive paths, NUL bytes, and dir mismatches in Shape A & B."""
    # 1. Direct helper checks
    assert check_scores._is_flat_csv_filename("saliency_e0.csv")
    for bad_flat in (
        "",
        "../gcn_s1000/saliency_e0.csv",
        "gcn_s1000/saliency_e0.csv",
        "gcn_s1000\\saliency_e0.csv",
        "run_log.json",
        ".csv",
        "/tmp/saliency_e0.csv",
        "C:\\tmp\\saliency_e0.csv",
        "C:/tmp/saliency_e0.csv",
        "C:saliency_e0.csv",
        "gnn\x00.csv",
        ".",
        "..",
    ):
        assert not check_scores._is_flat_csv_filename(bad_flat), f"Expected flat rejection for {bad_flat!r}"

    for ok_rel in ("saliency_e0.csv", "gcn_s1000/saliency_e0.csv", "bashapes/gcn_s1000/saliency_e0.csv"):
        assert check_scores._is_safe_relative_csv_path(ok_rel), f"Expected relative acceptance for {ok_rel!r}"
    for bad_rel in (
        "",
        "../gcn_s1000/saliency_e0.csv",
        "gcn_s1000/../gcn_s1000/saliency_e0.csv",
        "./saliency_e0.csv",
        "gcn_s1000/./saliency_e0.csv",
        "run_log.json",
        "gcn_s1000/run_log.json",
        "/abs/saliency_e0.csv",
        "\\abs\\saliency_e0.csv",
        "C:/tmp/saliency_e0.csv",
        "C:\\tmp\\saliency_e0.csv",
        "C:saliency_e0.csv",
        "\\\\server\\share\\saliency_e0.csv",
        "gnn\x00.csv",
    ):
        assert not check_scores._is_safe_relative_csv_path(bad_rel), f"Expected relative rejection for {bad_rel!r}"

    # 2. End-to-end Shape A: traversal back into same dir, run_log.json, non-flat subdir, abs/drive, NUL byte
    rid_sal = "bashapes_gcn_g1000_saliency_e0_test"
    for tag, bad_cf in (
        ("traversal", "../gcn_s1000/saliency_e0.csv"),
        ("noncsv", "run_log.json"),
        ("subdir", "sub/saliency_e0.csv"),
        ("abs_posix", "/tmp/saliency_e0.csv"),
        ("win_drive", "C:\\tmp\\saliency_e0.csv"),
        ("win_drive_rel", "C:saliency_e0.csv"),
        ("nul_byte", "gnn\x00.csv"),
    ):
        fx = _build_valid_fixture(tmp_path / f"b14_a_{tag}")
        lp = fx["seed_dir"] / "run_log.json"
        lo = json.loads(lp.read_text(encoding="utf-8"))
        lo["runs_by_id"][rid_sal]["csv_file"] = bad_cf
        lp.write_text(json.dumps(lo), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL", f"Shape A case {tag} ({bad_cf!r}) unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep["errors"]), rep["errors"]

    # 3. End-to-end Shape B (flat & nested): traversal, run_log.json, abs/drive, NUL byte, and wrong-dir same-basename
    for tag, bad_cf in (
        ("traversal", "../gcn_s1000/saliency_e0.csv"),
        ("inner_traversal", "gcn_s1000/../gcn_s1000/saliency_e0.csv"),
        ("noncsv", "run_log.json"),
        ("noncsv_sub", "gcn_s1000/run_log.json"),
        ("abs_posix", "/tmp/saliency_e0.csv"),
        ("win_drive", "C:/tmp/saliency_e0.csv"),
        ("win_drive_rel", "C:saliency_e0.csv"),
        ("nul_byte", "gcn_s1000/gnn\x00.csv"),
    ):
        fx = _build_valid_fixture(tmp_path / f"b14_b_{tag}")
        lp = fx["seed_dir"] / "run_log.json"
        seed_log = json.loads(lp.read_text(encoding="utf-8"))
        lp.unlink()
        flat_runs = []
        for rid, rinfo in seed_log["runs_by_id"].items():
            flat_runs.append(
                {
                    "dataset": "bashapes",
                    "arch": "gcn",
                    "gnn_seed": 1000,
                    "split": "test",
                    "checkpoint_sha256": fx["ckpt_sha"],
                    "run_id": rid,
                    "method": rinfo["method"],
                    "hyperparams": rinfo["hyperparams"],
                    "csv_file": bad_cf if rid == rid_sal else f"gcn_s1000/{rinfo['csv_file']}",
                    "csv_sha256": rinfo["csv_sha256"],
                }
            )
        (fx["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL", f"Shape B case {tag} ({bad_cf!r}) unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep["errors"]), rep["errors"]

    # 4. Shape B strict resolved path matching: declared 'other_dir/saliency_e0.csv' vs disk 'gcn_s1000/saliency_e0.csv'
    fx_dir = _build_valid_fixture(tmp_path / "b14_b_wrong_dir")
    lp_dir = fx_dir["seed_dir"] / "run_log.json"
    seed_log_dir = json.loads(lp_dir.read_text(encoding="utf-8"))
    lp_dir.unlink()
    flat_wrong_dir = []
    for rid, rinfo in seed_log_dir["runs_by_id"].items():
        flat_wrong_dir.append(
            {
                "dataset": "bashapes",
                "arch": "gcn",
                "gnn_seed": 1000,
                "split": "test",
                "checkpoint_sha256": fx_dir["ckpt_sha"],
                "run_id": rid,
                "method": rinfo["method"],
                "hyperparams": rinfo["hyperparams"],
                "csv_file": (
                    f"other_dir/{rinfo['csv_file']}" if rid == rid_sal else f"gcn_s1000/{rinfo['csv_file']}"
                ),
                "csv_sha256": rinfo["csv_sha256"],
            }
        )
    (fx_dir["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_wrong_dir}), encoding="utf-8")
    rep_dir = check_scores.validate_scores_dir(
        fx_dir["scores_dir"], checkpoints_dir=fx_dir["checkpoints_dir"], data_dir=fx_dir["data_dir"]
    )
    assert rep_dir["status"] == "FAIL"
    checks_dir = {e["check"] for e in rep_dir["errors"]}
    assert "missing_logged_csv" in checks_dir
    assert "unlogged_csv" in checks_dir


def test_def_b15_root_shape_a_run_log_never_hides_unlogged_subdirectory_csv(tmp_path):
    """DEF-B15: A root Shape A run_log.json must never set root_has_aggregate_log or hide rogue_sub/*.csv."""
    import shutil

    # Case 1: Root scores_dir holds a valid Shape A run_log.json + CSVs directly in root, plus rogue_sub/gnnexplainer_e9.csv
    fx = _build_valid_fixture(tmp_path / "b15_root_shape_a")
    for p in list(fx["seed_dir"].iterdir()):
        shutil.move(str(p), str(fx["scores_dir"] / p.name))
    fx["seed_dir"].rmdir()

    # Sanity: valid flat root Shape A passes before adding rogue_sub
    rep_ok = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_ok["status"] == "PASS", rep_ok["errors"]

    # Add rogue_sub/gnnexplainer_e9.csv with same (dataset, arch, split, gnn_seed) and NO run_log.json in rogue_sub
    rogue_sub = fx["scores_dir"] / "rogue_sub"
    rogue_sub.mkdir()
    df_rogue = pd.read_csv(fx["scores_dir"] / "gnnexplainer_e0.csv")
    df_rogue["explainer_seed"] = 9
    df_rogue["seed"] = 9
    df_rogue["run_id"] = "bashapes_gcn_g1000_gnnexplainer_e9_test"
    rogue_csv = rogue_sub / "gnnexplainer_e9.csv"
    df_rogue.to_csv(rogue_csv, index=False)

    rep_rogue = check_scores.validate_scores_dir(
        fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
    )
    assert rep_rogue["status"] == "FAIL"
    assert any(
        e["check"] == "unlogged_csv" and "rogue_sub" in e["file"] for e in rep_rogue["errors"]
    ), rep_rogue["errors"]

    # Case 2: Standard Shape A in gcn_s1000/ (NO root run_log.json) + rogue_sub/gnnexplainer_e9.csv
    fx2 = _build_valid_fixture(tmp_path / "b15_subdir_rogue")
    rogue_sub2 = fx2["scores_dir"] / "rogue_sub"
    rogue_sub2.mkdir()
    shutil.copyfile(rogue_csv, rogue_sub2 / "gnnexplainer_e9.csv")
    rep_rogue2 = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep_rogue2["status"] == "FAIL"
    assert any(
        e["check"] == "unlogged_csv" and "rogue_sub" in e["file"] for e in rep_rogue2["errors"]
    ), rep_rogue2["errors"]


def test_def_b16_rejects_falsy_and_malformed_csv_file_and_blocks_renamed_csv_bypass(tmp_path):
    """DEF-B16: Falsy/malformed csv_file (False, '', whitespace, non-str, NUL) is rejected in Shape A & B and cannot bypass renamed CSV checks."""
    import shutil

    # Direct helper checks
    assert check_scores._extract_valid_csv_file({}, "log", "r") == (True, None)
    assert check_scores._extract_valid_csv_file({"csv_file": None}, "log", "r") == (True, None)
    assert check_scores._extract_valid_csv_file({"csv_file": "saliency_e0.csv"}, "log", "r") == (
        True,
        "saliency_e0.csv",
    )
    for bad_cf in (False, "", "   ", " gnnexplainer_e0.csv ", 123, ["a.csv"], "gnn\x00.csv"):
        errs_direct = []
        ok, val = check_scores._extract_valid_csv_file(
            {"csv_file": bad_cf}, "log.json", "rid", lambda f, c, m, lines=None: errs_direct.append((f, c, m))
        )
        assert not ok and val is None and len(errs_direct) == 1 and errs_direct[0][1] == "run_log"

    rid_gnn = "bashapes_gcn_g1000_gnnexplainer_e0_test"
    bad_values = [
        ("bool_false", False),
        ("empty_str", ""),
        ("ws_only", "   "),
        ("padded", " gnnexplainer_e0.csv "),
        ("int_val", 123),
        ("nul_byte", "gnn\x00.csv"),
    ]

    for tag, bad_cf in bad_values:
        # Shape A with renamed CSV trying to bypass filename reconciliation via falsy/malformed csv_file
        fx_a = _build_valid_fixture(tmp_path / f"b16_a_{tag}")
        shutil.copyfile(fx_a["gnn0"], fx_a["seed_dir"] / "rogue.csv")
        fx_a["gnn0"].unlink()
        lp_a = fx_a["seed_dir"] / "run_log.json"
        lo_a = json.loads(lp_a.read_text(encoding="utf-8"))
        lo_a["runs_by_id"][rid_gnn]["csv_file"] = bad_cf
        lp_a.write_text(json.dumps(lo_a), encoding="utf-8")
        rep_a = check_scores.validate_scores_dir(
            fx_a["scores_dir"], checkpoints_dir=fx_a["checkpoints_dir"], data_dir=fx_a["data_dir"]
        )
        assert rep_a["status"] == "FAIL", f"Shape A DEF-B16 {tag} unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep_a["errors"]), rep_a["errors"]

        # Shape B nested with renamed CSV + falsy/malformed csv_file
        fx_bn = _build_valid_fixture(tmp_path / f"b16_bn_{tag}")
        shutil.copyfile(fx_bn["gnn0"], fx_bn["seed_dir"] / "rogue.csv")
        fx_bn["gnn0"].unlink()
        lp_bn = fx_bn["seed_dir"] / "run_log.json"
        seed_bn = json.loads(lp_bn.read_text(encoding="utf-8"))
        lp_bn.unlink()
        seed_bn["runs_by_id"][rid_gnn]["csv_file"] = bad_cf
        (fx_bn["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": [seed_bn]}), encoding="utf-8")
        rep_bn = check_scores.validate_scores_dir(
            fx_bn["scores_dir"], checkpoints_dir=fx_bn["checkpoints_dir"], data_dir=fx_bn["data_dir"]
        )
        assert rep_bn["status"] == "FAIL", f"Shape B nested DEF-B16 {tag} unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep_bn["errors"]), rep_bn["errors"]

        # Shape B flat with renamed CSV + falsy/malformed csv_file
        fx_bf = _build_valid_fixture(tmp_path / f"b16_bf_{tag}")
        shutil.copyfile(fx_bf["gnn0"], fx_bf["seed_dir"] / "rogue.csv")
        fx_bf["gnn0"].unlink()
        lp_bf = fx_bf["seed_dir"] / "run_log.json"
        seed_bf = json.loads(lp_bf.read_text(encoding="utf-8"))
        lp_bf.unlink()
        flat_runs = []
        for rid, rinfo in seed_bf["runs_by_id"].items():
            flat_runs.append(
                {
                    "dataset": "bashapes",
                    "arch": "gcn",
                    "gnn_seed": 1000,
                    "split": "test",
                    "checkpoint_sha256": fx_bf["ckpt_sha"],
                    "run_id": rid,
                    "method": rinfo["method"],
                    "hyperparams": rinfo["hyperparams"],
                    "csv_file": bad_cf if rid == rid_gnn else f"gcn_s1000/{rinfo['csv_file']}",
                    "csv_sha256": rinfo["csv_sha256"],
                }
            )
        (fx_bf["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
        rep_bf = check_scores.validate_scores_dir(
            fx_bf["scores_dir"], checkpoints_dir=fx_bf["checkpoints_dir"], data_dir=fx_bf["data_dir"]
        )
        assert rep_bf["status"] == "FAIL", f"Shape B flat DEF-B16 {tag} unexpectedly passed"
        assert any(e["check"] == "run_log" for e in rep_bf["errors"]), rep_bf["errors"]


def test_def_b17_checkpoint_path_nonstring_and_directory_paths_no_crash(tmp_path):
    """DEF-B17: Reject non-string checkpoint_path and directories for checkpoint_path, data.pt, and run_log.json without TypeError/PermissionError."""
    # 1. Direct _resolve_checkpoint_hash checks with non-string explicit path & directory explicit path
    empty_ckpts = tmp_path / "empty_ckpts"
    empty_ckpts.mkdir()
    for bad_explicit in (123, tmp_path, pathlib.Path(".")):
        sha, src, err = check_scores._resolve_checkpoint_hash(
            dataset="bashapes",
            arch="gcn",
            gnn_seed=1000,
            checkpoints_dir=empty_ckpts,
            explicit_ckpt_path=bad_explicit,  # type: ignore[arg-type]
            run_log_map={},
            split="test",
        )
        assert sha is None and src is None and isinstance(err, str) and len(err) > 0

    # 2. Direct _resolve_checkpoint_hash fallback via run_log_map with non-string, '.', directory, and NUL byte
    for bad_cp in (123, False, "", ".", str(tmp_path), "bad\x00.pt"):
        sha, src, err = check_scores._resolve_checkpoint_hash(
            dataset="bashapes",
            arch="gcn",
            gnn_seed=1000,
            checkpoints_dir=empty_ckpts,
            explicit_ckpt_path=None,
            run_log_map={("bashapes", "gcn", "test", 1000): {"checkpoint_path": bad_cp}},
            split="test",
        )
        assert sha is None and src is None and isinstance(err, str) and len(err) > 0

    # 3. End-to-end: --checkpoint-path <dir> returns clean FAIL (no PermissionError)
    fx_dir_ckpt = _build_valid_fixture(tmp_path / "b17_dir_ckpt")
    rep_dir_ckpt = check_scores.validate_scores_dir(
        fx_dir_ckpt["scores_dir"],
        checkpoints_dir=empty_ckpts,
        checkpoint_path=tmp_path,
        data_dir=fx_dir_ckpt["data_dir"],
    )
    assert rep_dir_ckpt["status"] == "FAIL"
    assert any(e["check"] == "checkpoint_hash_lookup" for e in rep_dir_ckpt["errors"])

    # 4. End-to-end: run_log.json checkpoint_path=123 and checkpoint_path="." return clean FAIL (no TypeError/PermissionError)
    for tag, bad_cp, expected_check in (("nonstring", 123, "run_log"), ("dot_dir", ".", "checkpoint_hash_lookup")):
        fx_cp = _build_valid_fixture(tmp_path / f"b17_log_cp_{tag}")
        lp_cp = fx_cp["seed_dir"] / "run_log.json"
        lo_cp = json.loads(lp_cp.read_text(encoding="utf-8"))
        lo_cp["checkpoint_path"] = bad_cp
        lp_cp.write_text(json.dumps(lo_cp), encoding="utf-8")
        rep_cp = check_scores.validate_scores_dir(
            fx_cp["scores_dir"], checkpoints_dir=empty_ckpts, data_dir=fx_cp["data_dir"]
        )
        assert rep_cp["status"] == "FAIL"
        assert any(e["check"] == expected_check for e in rep_cp["errors"]), rep_cp["errors"]

    # 5. End-to-end: directory named data.pt -> clean graph_verification FAIL (no PermissionError)
    fx_dpt = _build_valid_fixture(tmp_path / "b17_dir_datapt")
    dpt = fx_dpt["data_dir"] / "bashapes" / "data.pt"
    dpt.unlink()
    dpt.mkdir()
    rep_dpt = check_scores.validate_scores_dir(
        fx_dpt["scores_dir"], checkpoints_dir=fx_dpt["checkpoints_dir"], data_dir=fx_dpt["data_dir"]
    )
    assert rep_dpt["status"] == "FAIL"
    assert any(e["check"] == "graph_verification" for e in rep_dpt["errors"]), rep_dpt["errors"]

    # 6. End-to-end: directory named run_log.json -> clean run_log FAIL (no PermissionError)
    fx_dlog = _build_valid_fixture(tmp_path / "b17_dir_runlog")
    rlog = fx_dlog["seed_dir"] / "run_log.json"
    rlog.unlink()
    rlog.mkdir()
    rep_dlog = check_scores.validate_scores_dir(
        fx_dlog["scores_dir"], checkpoints_dir=fx_dlog["checkpoints_dir"], data_dir=fx_dlog["data_dir"]
    )
    assert rep_dlog["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_dlog["errors"]), rep_dlog["errors"]


def test_def_b14_to_b17_adversarial_case_mixed_ab_sibling_b_junction_nan_and_pt_dir(tmp_path):
    """Adversarial hardening for DEF-B14..B17: exact Shape B relpath casing/dots, mixed A+incomplete B, sibling B scoping, junction escape, NaN JSON, and dir .pt/.json."""
    import os
    import shutil
    import subprocess

    # 1. Helper rejections for trailing-dot/trailing-space directory segments and control characters
    for bad_rel in ("gcn_s1000./saliency_e0.csv", "gcn_s1000 /saliency_e0.csv", "gcn_s1000/a\nb.csv"):
        assert not check_scores._is_safe_relative_csv_path(bad_rel), bad_rel
    assert not check_scores._is_flat_csv_filename("a\nb.csv")
    assert not check_scores._is_flat_csv_filename(" saliency_e0.csv ")

    # 2. Shape B exact relative path matching (blocks Windows case-only dir spoofing & trailing dot/space)
    rid_sal = "bashapes_gcn_g1000_saliency_e0_test"
    for tag, bad_cf in (
        ("upper_dir", "GCN_S1000/saliency_e0.csv"),
        ("dot_dir", "gcn_s1000./saliency_e0.csv"),
        ("space_dir", "gcn_s1000 /saliency_e0.csv"),
        ("upper_file", "gcn_s1000/SALIENCY_E0.CSV"),
    ):
        fx = _build_valid_fixture(tmp_path / f"adv_case_{tag}")
        lo = json.loads((fx["seed_dir"] / "run_log.json").read_text(encoding="utf-8"))
        (fx["seed_dir"] / "run_log.json").unlink()
        flat_runs = []
        for rid, rinfo in lo["runs_by_id"].items():
            flat_runs.append(
                {
                    "dataset": "bashapes",
                    "arch": "gcn",
                    "gnn_seed": 1000,
                    "split": "test",
                    "checkpoint_sha256": fx["ckpt_sha"],
                    "run_id": rid,
                    "method": rinfo["method"],
                    "hyperparams": rinfo["hyperparams"],
                    "csv_file": bad_cf if rid == rid_sal else f"gcn_s1000/{rinfo['csv_file']}",
                    "csv_sha256": rinfo["csv_sha256"],
                }
            )
        (fx["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_runs}), encoding="utf-8")
        rep = check_scores.validate_scores_dir(
            fx["scores_dir"], checkpoints_dir=fx["checkpoints_dir"], data_dir=fx["data_dir"]
        )
        assert rep["status"] == "FAIL", f"Expected FAIL for Shape B spoof {tag} ({bad_cf!r})"
        assert any(e["check"] in {"run_log", "missing_logged_csv", "unlogged_csv"} for e in rep["errors"])

    # 3. Mixed Shape A (complete) + Shape B root aggregate log (incomplete: omits gnnexplainer_e0) MUST FAIL
    fx_mix = _build_valid_fixture(tmp_path / "adv_mixed_ab_incomplete")
    lo_mix = json.loads((fx_mix["seed_dir"] / "run_log.json").read_text(encoding="utf-8"))
    lo_b_incomplete = json.loads(json.dumps(lo_mix))
    del lo_b_incomplete["runs_by_id"]["bashapes_gcn_g1000_gnnexplainer_e0_test"]
    (fx_mix["scores_dir"] / "run_log.json").write_text(
        json.dumps({"runs": [lo_b_incomplete]}), encoding="utf-8"
    )
    rep_mix = check_scores.validate_scores_dir(
        fx_mix["scores_dir"], checkpoints_dir=fx_mix["checkpoints_dir"], data_dir=fx_mix["data_dir"]
    )
    assert rep_mix["status"] == "FAIL"
    assert any(e["check"] == "unlogged_csv" for e in rep_mix["errors"]), rep_mix["errors"]

    # 4. Two sibling Shape B logs cannot steal CSVs from each other when csv_file is omitted
    fx_sib = _build_valid_fixture(tmp_path / "adv_sibling_shape_b")
    lo_sib = json.loads((fx_sib["seed_dir"] / "run_log.json").read_text(encoding="utf-8"))
    (fx_sib["seed_dir"] / "run_log.json").unlink()
    sub1 = fx_sib["scores_dir"] / "sub1"
    sub2 = fx_sib["scores_dir"] / "sub2"
    sub1.mkdir()
    sub2.mkdir()
    for csv_f in list(fx_sib["seed_dir"].glob("*.csv")):
        shutil.move(str(csv_f), str(sub2 / csv_f.name))
    fx_sib["seed_dir"].rmdir()
    r1 = {
        k: {kk: vv for kk, vv in v.items() if kk != "csv_file"}
        for k, v in lo_sib["runs_by_id"].items()
        if "saliency_e0" in k
    }
    r2 = {
        k: {kk: vv for kk, vv in v.items() if kk != "csv_file"}
        for k, v in lo_sib["runs_by_id"].items()
        if "saliency_e0" not in k
    }
    (sub1 / "run_log.json").write_text(json.dumps({"runs": [dict(lo_sib, runs_by_id=r1)]}), encoding="utf-8")
    (sub2 / "run_log.json").write_text(json.dumps({"runs": [dict(lo_sib, runs_by_id=r2)]}), encoding="utf-8")
    rep_sib = check_scores.validate_scores_dir(
        fx_sib["scores_dir"], checkpoints_dir=fx_sib["checkpoints_dir"], data_dir=fx_sib["data_dir"]
    )
    assert rep_sib["status"] == "FAIL"
    checks_sib = {e["check"] for e in rep_sib["errors"]}
    assert "missing_logged_csv" in checks_sib, rep_sib["errors"]
    assert "unlogged_csv" in checks_sib, rep_sib["errors"]

    # 5. Real OS-level symlink/junction escape outside scores_dir MUST FAIL
    fx_junc = _build_valid_fixture(tmp_path / "adv_junction_escape")
    ext_dir = tmp_path / "adv_external_scores"
    ext_dir.mkdir()
    shutil.move(str(fx_junc["gnn0"]), str(ext_dir / "gnnexplainer_e0.csv"))
    junc_dir = fx_junc["scores_dir"] / "junc_escape"
    link_created = False
    try:
        if os.name == "nt":
            rc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junc_dir), str(ext_dir)],
                capture_output=True,
                text=True,
            )
            link_created = rc.returncode == 0
        else:
            os.symlink(str(ext_dir), str(junc_dir), target_is_directory=True)
            link_created = True
        assert link_created, "Failed to create directory junction/symlink for escape test"
        lo_j = json.loads((fx_junc["seed_dir"] / "run_log.json").read_text(encoding="utf-8"))
        (fx_junc["seed_dir"] / "run_log.json").unlink()
        flat_j = []
        for rid, rinfo in lo_j["runs_by_id"].items():
            flat_j.append(
                {
                    "dataset": "bashapes",
                    "arch": "gcn",
                    "gnn_seed": 1000,
                    "split": "test",
                    "checkpoint_sha256": fx_junc["ckpt_sha"],
                    "run_id": rid,
                    "method": rinfo["method"],
                    "hyperparams": rinfo["hyperparams"],
                    "csv_file": (
                        "junc_escape/gnnexplainer_e0.csv"
                        if "gnnexplainer" in rid
                        else f"gcn_s1000/{rinfo['csv_file']}"
                    ),
                    "csv_sha256": rinfo["csv_sha256"],
                }
            )
        (fx_junc["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_j}), encoding="utf-8")
        rep_j = check_scores.validate_scores_dir(
            fx_junc["scores_dir"], checkpoints_dir=fx_junc["checkpoints_dir"], data_dir=fx_junc["data_dir"]
        )
        assert rep_j["status"] == "FAIL"
        assert any(e["check"] == "missing_logged_csv" for e in rep_j["errors"]), rep_j["errors"]
    finally:
        if link_created:
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "rmdir", str(junc_dir)], capture_output=True)
            elif junc_dir.is_symlink():
                junc_dir.unlink()

    # 6. Non-finite NaN/Inf in run_log.json top-level metadata or checkpoint .json MUST FAIL
    fx_nan = _build_valid_fixture(tmp_path / "adv_nan_runlog")
    lp_nan = fx_nan["seed_dir"] / "run_log.json"
    lo_nan = json.loads(lp_nan.read_text(encoding="utf-8"))
    lo_nan["last_invocation_runtime_sec"] = float("nan")
    lp_nan.write_text(json.dumps(lo_nan, allow_nan=True), encoding="utf-8")
    rep_nan = check_scores.validate_scores_dir(
        fx_nan["scores_dir"], checkpoints_dir=fx_nan["checkpoints_dir"], data_dir=fx_nan["data_dir"]
    )
    assert rep_nan["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_nan["errors"]), rep_nan["errors"]

    fx_cknan = _build_valid_fixture(tmp_path / "adv_nan_ckpt_json")
    ck_json = fx_cknan["checkpoints_dir"] / "bashapes" / "gcn_s1000.json"
    ck_json.write_text(
        json.dumps({"gnn_seed": 1000, "sha256": fx_cknan["ckpt_sha"], "val_acc": float("inf")}, allow_nan=True),
        encoding="utf-8",
    )
    rep_cknan = check_scores.validate_scores_dir(
        fx_cknan["scores_dir"], checkpoints_dir=fx_cknan["checkpoints_dir"], data_dir=fx_cknan["data_dir"]
    )
    assert rep_cknan["status"] == "FAIL"
    assert any(e["check"] == "checkpoint_hash_lookup" for e in rep_cknan["errors"]), rep_cknan["errors"]

    # 7. Directory standing in for checkpoint .pt or .json alongside the valid sibling MUST FAIL
    for ext in (".pt", ".json"):
        fx_dir_ck = _build_valid_fixture(tmp_path / f"adv_dir_ckpt_{ext.lstrip('.')}")
        target = fx_dir_ck["checkpoints_dir"] / "bashapes" / f"gcn_s1000{ext}"
        target.unlink()
        target.mkdir()
        rep_dir_ck = check_scores.validate_scores_dir(
            fx_dir_ck["scores_dir"],
            checkpoints_dir=fx_dir_ck["checkpoints_dir"],
            data_dir=fx_dir_ck["data_dir"],
        )
        assert rep_dir_ck["status"] == "FAIL", f"Expected FAIL when checkpoint {ext} is a directory"
        assert any(e["check"] == "checkpoint_hash_lookup" for e in rep_dir_ck["errors"]), rep_dir_ck["errors"]


def test_def_b19_zero_csv_paths_return_complete_12_key_schema_and_clean_cli(tmp_path, capsys):
    """DEF-B19: Empty dir, missing path, and fake.csv-dir-only must return full 12-key schema and exit 1 cleanly without KeyError."""
    import subprocess

    expected_keys = {
        "status",
        "scores_path",
        "n_files",
        "n_rows",
        "n_run_logs",
        "datasets",
        "verified_hashes",
        "graph_verified_datasets",
        "saliency_determinism",
        "n_errors",
        "errors",
        "warnings",
    }

    empty_dir = tmp_path / "empty_scores"
    empty_dir.mkdir()
    missing_dir = tmp_path / "does_not_exist"
    fake_csv_only_dir = tmp_path / "fake_csv_dir_only"
    (fake_csv_only_dir / "fake.csv").mkdir(parents=True)

    for tag, target in (
        ("empty_dir", empty_dir),
        ("missing_dir", missing_dir),
        ("fake_csv_dir", fake_csv_only_dir),
    ):
        # 1. Direct validate_scores_dir check
        rep = check_scores.validate_scores_dir(target)
        assert set(rep.keys()) == expected_keys, f"{tag}: keys={sorted(rep.keys())}"
        assert rep["status"] == "FAIL"
        assert rep["n_files"] == 0
        assert rep["n_rows"] == 0
        assert rep["n_run_logs"] == 0
        assert rep["datasets"] == []
        assert rep["verified_hashes"] == {}
        assert rep["graph_verified_datasets"] == []
        assert rep["saliency_determinism"] == {}
        assert rep["n_errors"] == len(rep["errors"]) == 1
        assert rep["errors"][0]["check"] == "discovery"

        # 2. In-process CLI main() with --json-out
        json_out = tmp_path / f"report_inproc_{tag}.json"
        rc = check_scores.main([str(target), "--json-out", str(json_out)])
        captured = capsys.readouterr()
        assert rc == 1, f"{tag}: expected rc=1, got {rc}"
        assert "[FAIL]" in captured.out, f"{tag}: stdout={captured.out!r}"
        assert "Traceback" not in captured.err
        loaded = json.loads(json_out.read_text(encoding="utf-8"))
        assert set(loaded.keys()) == expected_keys
        assert loaded["status"] == "FAIL" and loaded["n_run_logs"] == 0 and loaded["n_errors"] == 1

        # 3. Subprocess CLI invocation with --json-out
        sub_json = tmp_path / f"report_subproc_{tag}.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_scores.py"),
                str(target),
                "--json-out",
                str(sub_json),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1, f"{tag}: rc={proc.returncode}, stderr={proc.stderr!r}"
        assert "[FAIL]" in proc.stdout, f"{tag}: stdout={proc.stdout!r}"
        assert "Traceback" not in proc.stderr and "KeyError" not in proc.stderr, proc.stderr
        sub_loaded = json.loads(sub_json.read_text(encoding="utf-8"))
        assert set(sub_loaded.keys()) == expected_keys


def test_def_b20_junction_and_symlink_containment_blocks_shape_a_shape_b_omitted_csvfile_and_alias(tmp_path):
    """DEF-B20: Block external NTFS junctions/symlinks in Shape B (omitted csv_file), Shape A (seed dir junction), and internal aliases."""
    import os
    import shutil
    import subprocess

    def _make_dir_link(link_path: pathlib.Path, target_path: pathlib.Path) -> None:
        if os.name == "nt":
            rc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
                capture_output=True,
                text=True,
            )
            assert rc.returncode == 0, f"mklink /J failed: {rc.stdout} {rc.stderr}"
        else:
            os.symlink(str(target_path), str(link_path), target_is_directory=True)

    def _remove_dir_link(link_path: pathlib.Path) -> None:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "rmdir", str(link_path)], capture_output=True)
        elif link_path.is_symlink():
            link_path.unlink()

    # Reproducer A: Shape B nested with omitted csv_file + external junction
    fx_a = _build_valid_fixture(tmp_path / "b20_repro_a")
    ext_a = tmp_path / "b20_external_a"
    ext_a.mkdir()
    shutil.move(str(fx_a["gnn0"]), str(ext_a / "gnnexplainer_e0.csv"))
    junc_a = fx_a["scores_dir"] / "junc_escape"
    _make_dir_link(junc_a, ext_a)
    try:
        lo_a = json.loads((fx_a["seed_dir"] / "run_log.json").read_text(encoding="utf-8"))
        (fx_a["seed_dir"] / "run_log.json").unlink()
        for rinfo in lo_a["runs_by_id"].values():
            rinfo.pop("csv_file", None)
        (fx_a["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": [lo_a]}), encoding="utf-8")
        rep_a = check_scores.validate_scores_dir(
            fx_a["scores_dir"], checkpoints_dir=fx_a["checkpoints_dir"], data_dir=fx_a["data_dir"]
        )
        assert rep_a["status"] == "FAIL", f"Reproducer A unexpectedly passed: {rep_a}"
        assert any(e["check"] == "path_containment" for e in rep_a["errors"]), rep_a["errors"]

        # Reproducer A2: Shape B flat single-run log with omitted csv_file + external junction
        flat_runs_a = []
        for rid, rinfo in lo_a["runs_by_id"].items():
            flat_runs_a.append(
                {
                    "dataset": "bashapes",
                    "arch": "gcn",
                    "gnn_seed": 1000,
                    "split": "test",
                    "checkpoint_sha256": fx_a["ckpt_sha"],
                    "run_id": rid,
                    "method": rinfo["method"],
                    "hyperparams": rinfo["hyperparams"],
                    "csv_sha256": rinfo["csv_sha256"],
                }
            )
        (fx_a["scores_dir"] / "run_log.json").write_text(json.dumps({"runs": flat_runs_a}), encoding="utf-8")
        rep_a2 = check_scores.validate_scores_dir(
            fx_a["scores_dir"], checkpoints_dir=fx_a["checkpoints_dir"], data_dir=fx_a["data_dir"]
        )
        assert rep_a2["status"] == "FAIL", f"Reproducer A2 unexpectedly passed: {rep_a2}"
        assert any(e["check"] == "path_containment" for e in rep_a2["errors"]), rep_a2["errors"]
    finally:
        _remove_dir_link(junc_a)

    # Reproducer B: Shape A where gcn_s1000 is itself a junction to an external folder
    fx_b = _build_valid_fixture(tmp_path / "b20_repro_b")
    ext_b = tmp_path / "b20_external_b_seed"
    shutil.move(str(fx_b["seed_dir"]), str(ext_b))
    junc_b = fx_b["scores_dir"] / "gcn_s1000"
    _make_dir_link(junc_b, ext_b)
    try:
        rep_b = check_scores.validate_scores_dir(
            fx_b["scores_dir"], checkpoints_dir=fx_b["checkpoints_dir"], data_dir=fx_b["data_dir"]
        )
        assert rep_b["status"] == "FAIL", f"Reproducer B unexpectedly passed: {rep_b}"
        containment_errs = [e for e in rep_b["errors"] if e["check"] == "path_containment"]
        assert len(containment_errs) >= 1, rep_b["errors"]
        if os.name == "nt":
            assert len(containment_errs) >= 4, rep_b["errors"]  # 3 CSVs + 1 run_log.json (+ dir)
    finally:
        _remove_dir_link(junc_b)

    # Reproducer C: Internal junction alias inside scores_dir (rel_resolved != rel_raw)
    fx_c = _build_valid_fixture(tmp_path / "b20_repro_c")
    alias_c = fx_c["scores_dir"] / "alias_seed"
    _make_dir_link(alias_c, fx_c["seed_dir"])
    try:
        rep_c = check_scores.validate_scores_dir(
            fx_c["scores_dir"], checkpoints_dir=fx_c["checkpoints_dir"], data_dir=fx_c["data_dir"]
        )
        assert rep_c["status"] == "FAIL", f"Reproducer C unexpectedly passed: {rep_c}"
        assert any(e["check"] == "path_containment" for e in rep_c["errors"]), rep_c["errors"]
    finally:
        _remove_dir_link(alias_c)

    # Reproducer D: Cyclic junction/symlink inside scores_dir (must never crash with WinError 1921 / OSError)
    fx_d = _build_valid_fixture(tmp_path / "b20_repro_d_cycle")
    loop_d = fx_d["seed_dir"] / "loop_to_root"
    _make_dir_link(loop_d, fx_d["scores_dir"])
    try:
        rep_d = check_scores.validate_scores_dir(
            fx_d["scores_dir"], checkpoints_dir=fx_d["checkpoints_dir"], data_dir=fx_d["data_dir"]
        )
        assert rep_d["status"] == "FAIL", f"Reproducer D (cycle) unexpectedly passed: {rep_d}"
        assert any(e["check"] == "path_containment" for e in rep_d["errors"]), rep_d["errors"]
    finally:
        _remove_dir_link(loop_d)

    # Reproducer E: Dangling junction/symlink inside scores_dir alongside valid CSVs
    fx_e = _build_valid_fixture(tmp_path / "b20_repro_e_dangling")
    dangling_e = fx_e["scores_dir"] / "dangling_link"
    _make_dir_link(dangling_e, tmp_path / "b20_nonexistent_target")
    try:
        rep_e = check_scores.validate_scores_dir(
            fx_e["scores_dir"], checkpoints_dir=fx_e["checkpoints_dir"], data_dir=fx_e["data_dir"]
        )
        assert rep_e["status"] == "FAIL", f"Reproducer E (dangling) unexpectedly passed: {rep_e}"
        assert any(e["check"] == "path_containment" for e in rep_e["errors"]), rep_e["errors"]
    finally:
        _remove_dir_link(dangling_e)


def test_round5_parity_dual_ckpt_sha_nonfile_nodes_and_duplicate_node_ids(tmp_path):
    """Round 5 parity Notes #1, #2, #4: dual ckpt json hash check, non-file val/test_nodes.json, and duplicate node IDs."""
    # Note #1: Checkpoint .json with BOTH sha256 and checkpoint_sha256
    fx1 = _build_valid_fixture(tmp_path / "r5_dual_sha")
    ckpt_pt = fx1["checkpoints_dir"] / "bashapes" / "gcn_s1000.pt"
    ckpt_json = fx1["checkpoints_dir"] / "bashapes" / "gcn_s1000.json"
    ckpt_pt.unlink()
    valid_sha = fx1["ckpt_sha"]

    # Both keys valid and agreeing -> PASS
    ckpt_json.write_text(
        json.dumps({"gnn_seed": 1000, "sha256": valid_sha, "checkpoint_sha256": valid_sha}),
        encoding="utf-8",
    )
    rep_ok = check_scores.validate_scores_dir(
        fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
    )
    assert rep_ok["status"] == "PASS", rep_ok["errors"]

    # Either key malformed or keys disagree -> FAIL
    for tag, meta_payload in (
        ("bad_ckpt_sha", {"sha256": valid_sha, "checkpoint_sha256": "MALFORMED"}),
        ("bad_sha", {"sha256": "", "checkpoint_sha256": valid_sha}),
        ("disagree", {"sha256": valid_sha, "checkpoint_sha256": "0" * 64}),
    ):
        ckpt_json.write_text(json.dumps(meta_payload), encoding="utf-8")
        rep_bad = check_scores.validate_scores_dir(
            fx1["scores_dir"], checkpoints_dir=fx1["checkpoints_dir"], data_dir=fx1["data_dir"]
        )
        assert rep_bad["status"] == "FAIL", f"Dual sha case {tag} unexpectedly passed"
        assert any(e["check"] == "checkpoint_hash_lookup" for e in rep_bad["errors"]), rep_bad["errors"]

    # Note #2: Directory OR dangling junction/broken symlink standing in for val_nodes.json or test_nodes.json must error (no silent fallback)
    import os
    import subprocess

    val_data_dir = tmp_path / "r5_nonfile_val_data"
    (val_data_dir / "bashapes" / "val_nodes.json").mkdir(parents=True)
    nodes_v, err_v = check_scores._load_expected_nodes(
        dataset="bashapes",
        split="val",
        data_dir=val_data_dir,
        expected_nodes_file=None,
        max_nodes=None,
        run_log_entry={"node_ids": [0, 1]},
    )
    assert nodes_v is None and isinstance(err_v, str) and "not a regular file" in err_v

    # Real OS-level dangling NTFS junction / broken symlink at val_nodes.json must NOT fall back to run_log_entry
    dangling_val_dir = tmp_path / "r5_dangling_val_data"
    (dangling_val_dir / "bashapes").mkdir(parents=True)
    dangling_val_path = dangling_val_dir / "bashapes" / "val_nodes.json"
    missing_target = tmp_path / "r5_missing_val_target"
    if os.name == "nt":
        rc_j = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dangling_val_path), str(missing_target)],
            capture_output=True,
            text=True,
        )
        assert rc_j.returncode == 0, f"mklink /J failed: {rc_j.stdout} {rc_j.stderr}"
    else:
        os.symlink(str(missing_target), str(dangling_val_path))
    try:
        nodes_dv, err_dv = check_scores._load_expected_nodes(
            dataset="bashapes",
            split="val",
            data_dir=dangling_val_dir,
            expected_nodes_file=None,
            max_nodes=None,
            run_log_entry={"node_ids": [0, 1]},
        )
        assert nodes_dv is None and isinstance(err_dv, str) and "not a regular file" in err_dv
    finally:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "rmdir", str(dangling_val_path)], capture_output=True)
        elif dangling_val_path.is_symlink():
            dangling_val_path.unlink()

    # Real OS-level dangling NTFS junction / broken symlink at checkpoint .json alongside valid .pt must fail
    fx_dck = _build_valid_fixture(tmp_path / "r5_dangling_ckpt_json")
    dck_json = fx_dck["checkpoints_dir"] / "bashapes" / "gcn_s1000.json"
    dck_json.unlink()
    if os.name == "nt":
        rc_ck = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dck_json), str(tmp_path / "r5_missing_ck_target")],
            capture_output=True,
            text=True,
        )
        assert rc_ck.returncode == 0, f"mklink /J failed: {rc_ck.stdout} {rc_ck.stderr}"
    else:
        os.symlink(str(tmp_path / "r5_missing_ck_target"), str(dck_json))
    try:
        rep_dck = check_scores.validate_scores_dir(
            fx_dck["scores_dir"], checkpoints_dir=fx_dck["checkpoints_dir"], data_dir=fx_dck["data_dir"]
        )
        assert rep_dck["status"] == "FAIL"
        assert any(e["check"] == "checkpoint_hash_lookup" for e in rep_dck["errors"]), rep_dck["errors"]
    finally:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "rmdir", str(dck_json)], capture_output=True)
        elif dck_json.is_symlink():
            dck_json.unlink()

    (val_data_dir / "bashapes" / "test_nodes.json").mkdir(parents=True)
    nodes_t, err_t = check_scores._load_expected_nodes(
        dataset="bashapes",
        split="test",
        data_dir=val_data_dir,
        expected_nodes_file=None,
        max_nodes=None,
        run_log_entry=None,
    )
    assert nodes_t is None and isinstance(err_t, str) and "not a regular file" in err_t

    # End-to-end split='val' with directory val_nodes.json and valid run_log node_ids -> FAIL
    fx2 = _build_valid_fixture(tmp_path / "r5_val_dir_e2e")
    for csv_p in (fx2["sal0"], fx2["sal1"], fx2["gnn0"]):
        df = pd.read_csv(csv_p)
        df["split"] = "val"
        df["run_id"] = df["run_id"].str.replace("_test", "_val", regex=False)
        df.to_csv(csv_p, index=False)
    lp2 = fx2["seed_dir"] / "run_log.json"
    lo2 = json.loads(lp2.read_text(encoding="utf-8"))
    lo2["split"] = "val"
    new_rbi = {}
    for k, v in lo2["runs_by_id"].items():
        nk = k.replace("_test", "_val")
        nv = dict(v, run_id=nk, csv_sha256=check_scores.sha256_file(fx2["seed_dir"] / v["csv_file"]))
        new_rbi[nk] = nv
    lo2["runs_by_id"] = new_rbi
    lp2.write_text(json.dumps(lo2), encoding="utf-8")
    (fx2["data_dir"] / "bashapes" / "val_nodes.json").mkdir()
    rep_val_dir = check_scores.validate_scores_dir(
        fx2["scores_dir"], checkpoints_dir=fx2["checkpoints_dir"], data_dir=fx2["data_dir"]
    )
    assert rep_val_dir["status"] == "FAIL"
    assert any(e["check"] == "expected_nodes_lookup" for e in rep_val_dir["errors"]), rep_val_dir["errors"]

    # Note #4: Duplicate node IDs rejected in _parse_nonneg_int_node_list, test_nodes.json, val_nodes.json, and run_log
    dup_nodes, dup_err = check_scores._parse_nonneg_int_node_list([0, 1, 0], "unit_test")
    assert dup_nodes is None and isinstance(dup_err, str) and "duplicate" in dup_err

    dup_val_dir = tmp_path / "r5_dup_val_nodes"
    (dup_val_dir / "bashapes").mkdir(parents=True)
    (dup_val_dir / "bashapes" / "val_nodes.json").write_text(
        json.dumps({"nodes": [0, 1, 0]}), encoding="utf-8"
    )
    dv_nodes, dv_err = check_scores._load_expected_nodes(
        dataset="bashapes",
        split="val",
        data_dir=dup_val_dir,
        expected_nodes_file=None,
        max_nodes=None,
        run_log_entry={"node_ids": [0, 1]},
    )
    assert dv_nodes is None and isinstance(dv_err, str) and "duplicate" in dv_err

    fx3 = _build_valid_fixture(tmp_path / "r5_dup_test_nodes")
    (fx3["data_dir"] / "bashapes" / "test_nodes.json").write_text(
        json.dumps({"nodes": [0, 1, 0]}), encoding="utf-8"
    )
    rep_dup_test = check_scores.validate_scores_dir(
        fx3["scores_dir"], checkpoints_dir=fx3["checkpoints_dir"], data_dir=fx3["data_dir"]
    )
    assert rep_dup_test["status"] == "FAIL"
    assert any(e["check"] == "expected_nodes_lookup" for e in rep_dup_test["errors"]), rep_dup_test["errors"]

    # Duplicate node_ids in run_log.json rejected as run_log error
    fx4 = _build_valid_fixture(tmp_path / "r5_dup_log_nodes")
    lp4 = fx4["seed_dir"] / "run_log.json"
    lo4 = json.loads(lp4.read_text(encoding="utf-8"))
    lo4["node_ids"] = [0, 1, 0]
    lp4.write_text(json.dumps(lo4), encoding="utf-8")
    rep_dup_log = check_scores.validate_scores_dir(
        fx4["scores_dir"], checkpoints_dir=fx4["checkpoints_dir"], data_dir=fx4["data_dir"]
    )
    assert rep_dup_log["status"] == "FAIL"
    assert any(e["check"] == "run_log" for e in rep_dup_log["errors"]), rep_dup_log["errors"]
