"""Quản lý nạp và lưu Checkpoint cho các mô hình GNN."""

import json
import pathlib
from typing import Any, Dict, Optional, Tuple
import torch

from sna.core.config import DATA_DIR
from sna.core.repro import compute_sha256
from .gcn import GCN


def load_gcn_checkpoint(
    dataset_name: Optional[str] = None,
    root: Optional[pathlib.Path] = None,
    ckpt_path: Optional[pathlib.Path] = None,
    expected_sha256: Optional[str] = None,
) -> Tuple[GCN, torch.Tensor, Dict]:
    """Nạp GCN model đã khóa eval() và pred từ checkpoint (BG02, BG03).

    Hỗ trợ truyền `ckpt_path` trực tiếp và kiểm tra `expected_sha256` (FB-1).
    """
    if ckpt_path is not None:
        resolved_path = pathlib.Path(ckpt_path)
    elif dataset_name is not None:
        root_path = root or DATA_DIR
        resolved_path = root_path / dataset_name.lower().strip() / "gcn_ckpt.pt"
    else:
        raise ValueError("Phải cung cấp dataset_name hoặc ckpt_path.")

    if not resolved_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {resolved_path}")

    if expected_sha256 is not None:
        actual_sha = compute_sha256(resolved_path)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {resolved_path}: "
                f"expected {expected_sha256}, got {actual_sha}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(resolved_path, map_location=device, weights_only=False)
    model = GCN(
        in_dim=ck["in_dim"],
        hidden_dim=ck["hidden"],
        out_dim=ck["n_classes"],
        dropout=ck.get("dropout", 0.0),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, ck["pred"], ck.get("metrics", {})


def save_gcn_checkpoint(
    ckpt_path: pathlib.Path,
    state_dict: Dict[str, Any],
    in_dim: int,
    hidden_dim: int,
    n_classes: int,
    dropout: float,
    pred: torch.Tensor,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pathlib.Path, pathlib.Path, str]:
    """Lưu checkpoint .pt và file metadata .json kèm sha256."""
    pt_path = pathlib.Path(ckpt_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "in_dim": int(in_dim),
        "hidden": int(hidden_dim),
        "n_classes": int(n_classes),
        "dropout": float(dropout),
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "pred": pred.detach().cpu(),
        "metrics": metrics,
    }
    torch.save(payload, pt_path)
    pt_sha256 = compute_sha256(pt_path)

    json_path = pt_path.with_suffix(".json")
    json_payload: Dict[str, Any] = {
        "checkpoint_path": str(pt_path),
        "checkpoint_sha256": pt_sha256,
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "n_classes": int(n_classes),
        "dropout": float(dropout),
        "metrics": metrics,
    }
    if metadata:
        json_payload.update(metadata)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    return pt_path, json_path, pt_sha256
