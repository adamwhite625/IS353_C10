"""GATE 0 Step 1: Verify SHA-256 of midterm data.pt & gcn_ckpt.pt and 100% pred match."""

from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sna.core.config import DATA_DIR
from sna.core.repro import compute_sha256
from sna.models.checkpoint import load_gcn_checkpoint


def verify_dataset(dataset: str) -> dict:
    ds_dir = DATA_DIR / dataset
    data_path = ds_dir / "data.pt"
    ckpt_path = ds_dir / "gcn_ckpt.pt"

    data_sha = compute_sha256(data_path)
    ckpt_sha = compute_sha256(ckpt_path)

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    model, ck_pred, metrics = load_gcn_checkpoint(dataset_name=dataset)
    model = model.cpu()

    with torch.no_grad():
        recomputed_pred = model(data["x"].cpu(), data["edge_index"].cpu()).argmax(dim=-1)

    pred_match = bool(torch.equal(recomputed_pred, ck_pred.cpu()))
    if not pred_match:
        raise AssertionError(f"[{dataset}] Recomputed pred does NOT match ck['pred']!")

    test_acc = float((recomputed_pred[data["test_mask"].cpu()] == data["y"].cpu()[data["test_mask"].cpu()]).float().mean())
    result = {
        "dataset": dataset,
        "data_sha256": data_sha,
        "ckpt_sha256": ckpt_sha,
        "pred_match_100pct": pred_match,
        "num_nodes": int(data["x"].shape[0]),
        "recomputed_test_acc": round(test_acc, 6),
        "ckpt_metrics": metrics,
    }
    print(f"[{dataset}] data_sha256={data_sha}")
    print(f"[{dataset}] ckpt_sha256={ckpt_sha}")
    print(f"[{dataset}] pred_match_100pct={pred_match} | test_acc={test_acc:.4f}")
    return result


def main() -> None:
    results = [verify_dataset(ds) for ds in ("bashapes", "cora")]
    out_path = ROOT / "results" / "gate0_step1_hashes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved GATE 0 Step 1 verification to {out_path}")


if __name__ == "__main__":
    main()

