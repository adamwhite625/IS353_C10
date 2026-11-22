"""Unit tests for PN1 (Dang) tasks: F5, F6, F7, sha256/manifest, and T2/T3."""

import json
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sna.analysis.mechanisms_t2_t3 import evaluate_node_t2_t3, recompute_degree_features
from sna.core.config import CONFIGS_DIR, DATA_DIR, load_json_config, load_yaml_config
from sna.core.repro import build_manifest, compute_sha256, save_manifest
from sna.models.checkpoint import load_gcn_checkpoint, save_gcn_checkpoint
from sna.models.gcn import GCN


def test_f6_config_consistency():
    """F6: bashapes.yaml and data/bashapes/config.json must agree on model hyperparameters."""
    yaml_cfg = load_yaml_config(CONFIGS_DIR / "datasets" / "bashapes.yaml")
    json_cfg = load_json_config(DATA_DIR / "bashapes" / "config.json")

    assert yaml_cfg["model"]["hidden_dim"] == json_cfg["model"]["hidden"]
    assert yaml_cfg["model"]["dropout"] == json_cfg["model"]["dropout"]
    assert yaml_cfg["model"]["lr"] == json_cfg["model"]["lr"]
    assert yaml_cfg["model"]["weight_decay"] == json_cfg["model"]["weight_decay"]
    assert yaml_cfg["model"]["epochs"] == json_cfg["model"]["epochs"]


def test_sha256_and_checkpoint_roundtrip(tmp_path: pathlib.Path):
    """F5 & FB-1: save_gcn_checkpoint writes .pt/.json and load_gcn_checkpoint enforces sha256."""
    model = GCN(in_dim=4, hidden_dim=8, out_dim=2, dropout=0.0)
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1)

    ckpt_path = tmp_path / "gcn_s1000.pt"
    pt_path, json_path, sha = save_gcn_checkpoint(
        ckpt_path=ckpt_path,
        state_dict=model.state_dict(),
        in_dim=4,
        hidden_dim=8,
        n_classes=2,
        dropout=0.0,
        pred=pred,
        metrics={"test_acc": 1.0},
    )
    assert pt_path.exists() and json_path.exists()
    assert sha == compute_sha256(pt_path)

    loaded_model, loaded_pred, loaded_metrics = load_gcn_checkpoint(
        ckpt_path=pt_path,
        expected_sha256=sha,
    )
    assert torch.equal(loaded_pred, pred)
    assert loaded_metrics["test_acc"] == 1.0
    with torch.no_grad():
        recomputed = loaded_model(x, edge_index).argmax(dim=-1)
    assert torch.equal(recomputed, pred)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_gcn_checkpoint(ckpt_path=pt_path, expected_sha256="0" * 64)


def test_manifest_generation(tmp_path: pathlib.Path):
    """FB-1: build_manifest and save_manifest produce valid JSON with sha256 fields."""
    sample_file = tmp_path / "dummy.txt"
    sample_file.write_text("hello", encoding="utf-8")

    manifest = build_manifest(
        command="python scripts/evaluate.py",
        device="cpu",
        data_files={"dummy": sample_file},
        nodes=[0, 1, 2],
        protocol={"k": 5},
    )
    out = save_manifest(manifest, tmp_path / "manifest.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["data_sha256"]["dummy"] == compute_sha256(sample_file)
    assert loaded["num_nodes"] == 3
    assert loaded["protocol"] == {"k": 5}


def test_t2_t3_mechanisms_synthetic():
    """FB-3: T2 and T3 compute finite Fid_motif, Fid_1hop, and recomputed degree features."""
    torch.manual_seed(0)
    x = torch.zeros(4, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 1], [1, 0, 2, 1, 3, 2, 1, 3]],
        dtype=torch.long,
    )
    x = recompute_degree_features(x, edge_index)
    # Node 1 has degree 3 (neighbors 0, 2, 3)
    assert float(x[1, 0]) == 3.0
    assert abs(float(x[1, 1]) - (3.0 ** 0.5)) < 1e-5

    # Motif edges: (1,2), (2,3), (1,3)
    edge_mask = torch.tensor([False, False, True, True, True, True, True, True], dtype=torch.bool)
    data = {"x": x, "edge_index": edge_index, "edge_mask": edge_mask}
    model = GCN(in_dim=10, hidden_dim=8, out_dim=4, dropout=0.0)
    res = evaluate_node_t2_t3(model=model, data=data, node_id=1, target_class=0)

    assert res["num_motif_edges_2hop"] == 3
    assert res["num_1hop_edges"] == 3
    assert isinstance(res["fid_motif"], float)
    assert isinstance(res["fid_motif_recomp_x"], float)

