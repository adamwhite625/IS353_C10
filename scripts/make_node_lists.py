"""F7: Generate test_nodes.json for BA-Shapes (all 140) and Cora (100 stratified, seed 0)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sna.core.config import DATA_DIR
from sna.core.repro import compute_sha256


def generate_test_nodes(dataset: str, seed: int = 0, cora_sample_size: int = 100) -> pathlib.Path:
    """Create data/{dataset}/test_nodes.json and assert zero overlap with train and val."""
    ds_dir = DATA_DIR / dataset
    data_path = ds_dir / "data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.pt: {data_path}")

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    y = data["y"].cpu().numpy()
    train_mask = data["train_mask"].cpu().numpy().astype(bool)
    val_mask = data["val_mask"].cpu().numpy().astype(bool)
    test_mask = data["test_mask"].cpu().numpy().astype(bool)

    train_nodes = set(np.where(train_mask)[0].tolist())
    val_nodes_mask = set(np.where(val_mask)[0].tolist())

    val_json_path = ds_dir / "val_nodes.json"
    val_nodes_json = set()
    if val_json_path.exists():
        with open(val_json_path, "r", encoding="utf-8") as f:
            val_nodes_json = set(json.load(f).get("nodes", []))

    all_test_indices = np.where(test_mask)[0]

    if dataset == "bashapes":
        selected_nodes: List[int] = sorted(all_test_indices.tolist())
    elif dataset == "cora":
        if len(all_test_indices) < cora_sample_size:
            raise ValueError(f"Cora has only {len(all_test_indices)} test nodes (< {cora_sample_size}).")
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=cora_sample_size, random_state=seed)
        sample_idx, _ = next(splitter.split(all_test_indices, y[all_test_indices]))
        selected_nodes = sorted(all_test_indices[sample_idx].tolist())
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    selected_set = set(selected_nodes)
    assert len(selected_set) == len(selected_nodes), "Duplicate nodes found in test_nodes!"
    assert selected_set.isdisjoint(train_nodes), "test_nodes overlaps with train_mask!"
    assert selected_set.isdisjoint(val_nodes_mask), "test_nodes overlaps with val_mask!"
    assert selected_set.issubset(set(all_test_indices.tolist())), "test_nodes contains non-test nodes!"

    overlap_midterm_val = sorted(selected_set & val_nodes_json)
    if overlap_midterm_val:
        print(
            f"[{dataset}] NOTE: {len(overlap_midterm_val)} test_mask nodes were present in legacy val_nodes.json: "
            f"{overlap_midterm_val}"
        )

    unique_cls, counts = np.unique(y[selected_nodes], return_counts=True)
    class_dist: Dict[str, int] = {str(int(c)): int(n) for c, n in zip(unique_cls, counts)}

    out_path = ds_dir / "test_nodes.json"
    payload = {
        "split": "test",
        "dataset": dataset,
        "seed": seed,
        "num_nodes": len(selected_nodes),
        "data_sha256": compute_sha256(data_path),
        "class_distribution": class_dist,
        "nodes": selected_nodes,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[{dataset}] Generated {out_path} ({len(selected_nodes)} nodes) | class_distribution={class_dist}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stratified test_nodes.json lists.")
    parser.add_argument("--datasets", nargs="+", default=["bashapes", "cora"], choices=["bashapes", "cora"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for ds in args.datasets:
        generate_test_nodes(ds, seed=args.seed)


if __name__ == "__main__":
    main()

