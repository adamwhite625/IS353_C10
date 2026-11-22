"""Schema v2 của CSV điểm cạnh (KE_HOACH_CUOI_KY_FINAL.md, Mục 6.4).

Danh sách cột phải giống hệt ``SCHEMA_V2_COLUMNS`` trong ``scripts/run_explainers.py``
(có test đối chiếu trong ``tests/test_pgexplainer_f1_f2_f9.py``).
"""

from typing import List

SCHEMA_V2_COLUMNS: List[str] = [
    "run_id",
    "dataset",
    "split",
    "arch",
    "gnn_seed",
    "checkpoint_sha256",
    "method",
    "explainer_seed",
    "seed",
    "node_id",
    "target_class",
    "edge_id",
    "u",
    "v",
    "importance_score",
    "runtime_sec",
    "status",
]

# Tên chuẩn của phương pháp ghi vào cột ``method``.
CANONICAL_METHOD_NAMES = {
    "pge": "pgexplainer",
    "random_edge": "random",
}


def canonical_method(name: str) -> str:
    key = name.lower().strip()
    return CANONICAL_METHOD_NAMES.get(key, key)


def make_run_id(dataset: str, arch: str, gnn_seed: int, method: str, explainer_seed: int, split: str) -> str:
    """``{ds}_{arch}_g{g}_{method}_e{e}_{split}`` như ``run_explainers.py``."""
    return f"{dataset}_{arch}_g{gnn_seed}_{canonical_method(method)}_e{explainer_seed}_{split}"
