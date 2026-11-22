"""Unified Multi-Seed Explainer Runner (PN2: Thien + Khoi).

Runs Saliency, GNNExplainer, and/or PGExplainer across a grid of GNN seeds and
explainer seeds on CPU (default, deterministic), emitting Schema v2 edge score
CSVs and run_log.json per Sections 4, 6.2, 6.3, 6.4, and 7.2 of
KE_HOACH_CUOI_KY_FINAL.md.

Usage:
    python scripts/run_explainers.py \
        --dataset bashapes --split test \
        --gnn-seeds 1000 1001 1002 1003 1004 \
        --explainer-seeds 0 1 2 3 4 \
        --methods saliency gnnexplainer pgexplainer
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sna.data.candidate_edges import extract_candidate_edges
from sna.explainers.gnnexplainer import GNNExplainer
from sna.explainers.pgexplainer import PGExplainer, PGExplainerModel
from sna.explainers.saliency import SaliencyExplainer
from sna.models.gcn import GCN

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

SUPPORTED_METHODS = ("saliency", "gnnexplainer", "pgexplainer")
ALLOWED_LOG_METHODS = {"saliency", "gnnexplainer", "pgexplainer", "random", "nearest_first"}
ALLOWED_STATUSES = {"success", "empty_candidates", "failed"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_FULL_RE = re.compile(
    r"^([a-z0-9_]+)_(gcn|gat)_g(\d+)_(saliency|gnnexplainer|pgexplainer|random|nearest_first)_e(\d+)_(test|val)$"
)
RUN_ID_METHOD_RE = re.compile(
    r"^[a-z0-9_]+_(?:gcn|gat)_g\d+_(saliency|gnnexplainer|pgexplainer|random|nearest_first)_e\d+_(?:test|val)$"
)
MAX_SAFE_INT: int = (1 << 53) - 1  # 9007199254740991


def _parse_canonical_run_id(run_id: str) -> Optional[Tuple[str, str, int, str, int, str]]:
    m = RUN_ID_FULL_RE.fullmatch(run_id)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5)), m.group(6)


def _extract_method_from_run_id(run_id: str) -> Optional[str]:
    m = RUN_ID_METHOD_RE.fullmatch(run_id)
    return m.group(1) if m else None


def _reject_json_nonfinite(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {token!r} is not allowed.")


def _loads_strict_json(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_nonfinite)


def _is_flat_csv_filename(name: Any) -> bool:
    if not isinstance(name, str) or not name or name != name.strip() or "\x00" in name:
        return False
    if any(ord(ch) < 32 for ch in name):
        return False
    if "/" in name or "\\" in name or ".." in name or ":" in name or name in {".", ".."}:
        return False
    if pathlib.PureWindowsPath(name).drive:
        return False
    try:
        p_posix = pathlib.PurePosixPath(name)
        return p_posix.name == name and p_posix.suffix.lower() == ".csv" and bool(p_posix.stem)
    except ValueError:
        return False


def _validate_finite_json(obj: Any, label: str) -> None:
    """Ensure obj is JSON-serializable with strictly finite numbers (rejecting NaN/Inf)."""
    try:
        json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Corrupted {label}: contains non-finite (NaN/Inf) or non-serializable value: {exc}"
        ) from exc


def _is_bool_like_mask(series: pd.Series) -> pd.Series:
    """Identify boolean dtype, Python/NumPy bool scalars, and 'true'/'false' strings."""
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(True, index=series.index, dtype=bool)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.map(
            lambda v: isinstance(v, (bool, np.bool_))
            or (isinstance(v, str) and v.strip().lower() in {"true", "false"})
        ).astype(bool)
    return pd.Series(False, index=series.index, dtype=bool)


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric while turning non-parseable and >float64-overflow values into NaN."""
    try:
        return pd.to_numeric(series, errors="coerce")
    except OverflowError:
        vals: List[float] = []
        for v in series.tolist():
            if isinstance(v, (bool, np.bool_)) or v is None:
                vals.append(float("nan"))
                continue
            try:
                vals.append(float(v))
            except (OverflowError, TypeError, ValueError):
                vals.append(float("nan"))
        return pd.Series(vals, index=series.index, dtype=float)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_info() -> Tuple[Optional[str], Optional[bool]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _path_entry_exists(path: pathlib.Path) -> bool:
    """Return True if path exists as any filesystem entry (including broken symlinks and dangling NTFS junctions)."""
    return os.path.lexists(path) or path.exists() or path.is_symlink()


def load_checkpoint_strict(
    ckpt_path: pathlib.Path,
    arch: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[torch.Tensor], str]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found or not a regular file: {ckpt_path}")

    ckpt_sha = sha256_file(ckpt_path)

    # Verify sibling .json metadata hash if present
    meta_path = ckpt_path.with_suffix(".json")
    if _path_entry_exists(meta_path):
        if not meta_path.is_file():
            raise ValueError(f"Checkpoint metadata path is not a regular file: {meta_path}")
        try:
            meta = _loads_strict_json(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Corrupted checkpoint metadata {meta_path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(
                f"Corrupted checkpoint metadata {meta_path}: expected JSON object, got {type(meta).__name__}."
            )
        for sha_key in ("sha256", "checkpoint_sha256"):
            if sha_key in meta and meta[sha_key] is not None:
                meta_sha = meta[sha_key]
                if not isinstance(meta_sha, str) or not SHA256_RE.fullmatch(meta_sha):
                    raise ValueError(
                        f"Invalid {sha_key} in checkpoint metadata {meta_path}: {meta_sha!r}."
                    )
                if meta_sha != ckpt_sha:
                    raise ValueError(
                        f"Checkpoint SHA-256 mismatch between {ckpt_path} ({ckpt_sha}) "
                        f"and {meta_path} ({sha_key}={meta_sha})."
                    )

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(
            f"Corrupted checkpoint {ckpt_path}: expected dict, got {type(ck).__name__}."
        )
    if arch != "gcn":
        raise ValueError(f"Unsupported architecture '{arch}'. Only 'gcn' is enabled before GATE 2.")

    in_dim = int(ck["in_dim"])
    hidden_raw = ck.get("hidden", ck.get("hidden_dim"))
    out_raw = ck.get("n_classes", ck.get("out_dim", ck.get("num_classes")))
    if hidden_raw is None or out_raw is None:
        raise KeyError(f"Checkpoint {ckpt_path} missing hidden/out dimension metadata.")
    hidden_dim = int(hidden_raw)
    out_dim = int(out_raw)
    dropout = float(ck.get("dropout", 0.0))

    model = GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    saved_pred = ck.get("pred", None)
    if saved_pred is not None:
        saved_pred = torch.as_tensor(saved_pred).cpu()

    return model, saved_pred, ckpt_sha


def load_node_list(
    dataset: str,
    split: str,
    data_dir: pathlib.Path,
    nodes_file: Optional[pathlib.Path],
    max_nodes: Optional[int],
    num_nodes: Optional[int] = None,
) -> Tuple[List[int], pathlib.Path, str]:
    path = nodes_file or (data_dir / dataset / f"{split}_nodes.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"Node list file not found or not a regular file: {path}. "
            f"For split='test', run scripts/make_node_lists.py (H1 from PN1) first."
        )
    if max_nodes is not None:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
            raise ValueError(f"--max-nodes must be a positive integer, got {max_nodes!r}.")
    if num_nodes is not None:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
            raise ValueError(f"num_nodes must be a positive integer, got {num_nodes!r}.")
    try:
        raw = _loads_strict_json(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Failed to parse node list JSON {path}: {exc}") from exc
    if isinstance(raw, dict):
        if "nodes" not in raw:
            raise ValueError(f"Node list file {path} is missing required 'nodes' key.")
        raw_list = raw["nodes"]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raise ValueError(
            f"Node list file {path} must contain a JSON list or object with 'nodes', got {type(raw).__name__}."
        )
    if not isinstance(raw_list, list):
        raise ValueError(
            f"Node list in {path} must be a JSON list of non-negative integers, got {type(raw_list).__name__}."
        )
    nodes: List[int] = []
    for idx, item in enumerate(raw_list):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"Invalid node_id at index {idx} in {path}: {item!r} "
                f"(expected non-negative integer, not {type(item).__name__})."
            )
        if item < 0 or item > MAX_SAFE_INT:
            raise ValueError(
                f"Invalid node_id at index {idx} in {path}: {item} "
                f"(must be between 0 and {MAX_SAFE_INT})."
            )
        if num_nodes is not None and item >= num_nodes:
            raise ValueError(
                f"Out-of-bounds node_id at index {idx} in {path}: {item} >= num_nodes ({num_nodes})."
            )
        nodes.append(int(item))
    if len(nodes) == 0:
        raise ValueError(f"Node list is empty in {path}.")
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"Duplicate node IDs found in {path}.")
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    return nodes, path, sha256_file(path)


def _atomic_write_csv(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, out_path)


def _atomic_write_json(payload: Dict[str, Any], out_path: pathlib.Path) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, out_path)


def _method_hyperparams(
    method: str,
    gnn_cfg: Dict[str, Any],
    pg_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if method == "gnnexplainer":
        return dict(gnn_cfg)
    if method == "pgexplainer":
        return dict(pg_cfg)
    return {}


def _hyperparams_equal(hp_a: Any, hp_b: Dict[str, Any]) -> bool:
    """Strict JSON-canonical equality avoiding Python's True == 1 pitfall and NaN/Inf."""
    if not isinstance(hp_a, dict) or not isinstance(hp_b, dict):
        return False
    try:
        canon_a = json.dumps(hp_a, sort_keys=True, separators=(",", ":"), allow_nan=False)
        canon_b = json.dumps(hp_b, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return False
    return canon_a == canon_b


def _validate_resumable_csv(
    *,
    csv_path: pathlib.Path,
    expected_run_id: str,
    dataset: str,
    split: str,
    arch: str,
    gnn_seed: int,
    method: str,
    explainer_seed: int,
    ckpt_sha: str,
    node_ids: List[int],
    expected_hp: Dict[str, Any],
    prior_run_entry: Optional[Dict[str, Any]],
    old_top_hyperparams: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Validate that an existing CSV is safe to skip under --resume.

    Raises ValueError on any checkpoint, split, node-list, run_id, or hyperparameter drift.
    Returns (df_prev, verified_hyperparams_or_None).
    """
    try:
        df_prev = pd.read_csv(
            csv_path,
            dtype={
                "run_id": str,
                "dataset": str,
                "split": str,
                "arch": str,
                "checkpoint_sha256": str,
                "method": str,
                "status": str,
            },
            keep_default_na=False,
            na_values=["", "nan", "NaN", "NAN"],
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot --resume from unreadable/corrupted CSV {csv_path}: {exc}. "
            f"Re-run without --resume or remove the corrupted CSV."
        ) from exc
    missing_cols = [c for c in SCHEMA_V2_COLUMNS if c not in df_prev.columns]
    if missing_cols or df_prev.empty:
        raise ValueError(
            f"Cannot --resume from invalid or empty CSV {csv_path} (missing_cols={missing_cols}). "
            f"Re-run without --resume or remove the corrupted CSV."
        )

    if df_prev["status"].isna().any() or not df_prev["status"].isin(ALLOWED_STATUSES).all():
        raise ValueError(
            f"Cannot --resume {csv_path}: column 'status' contains missing or unsupported values. "
            f"Use a fresh --out-dir or omit --resume to overwrite."
        )

    success_prev = df_prev[df_prev["status"] == "success"]
    if not success_prev.empty:
        sc_bool = _is_bool_like_mask(success_prev["importance_score"])
        sc_clean = success_prev["importance_score"].astype(object).where(~sc_bool, np.nan)
        sc_num = _safe_to_numeric(sc_clean)
        if sc_bool.any() or sc_num.isna().any() or not np.isfinite(sc_num).all():
            raise ValueError(
                f"Cannot --resume {csv_path}: 'importance_score' contains non-finite, boolean, or missing values on success rows. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )

    checks = [
        ("run_id", expected_run_id),
        ("dataset", dataset),
        ("split", split),
        ("arch", arch),
        ("checkpoint_sha256", ckpt_sha),
        ("method", method),
    ]
    for col, expected_val in checks:
        if df_prev[col].isna().any():
            raise ValueError(
                f"Cannot --resume {csv_path}: column '{col}' contains missing/NaN values. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )
        vals = df_prev[col].astype(str).unique().tolist()
        if vals != [expected_val]:
            raise ValueError(
                f"Cannot --resume {csv_path}: existing '{col}'={vals} != current '{expected_val}'. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )

    for col, expected_int in (
        ("gnn_seed", gnn_seed),
        ("explainer_seed", explainer_seed),
        ("seed", explainer_seed),
    ):
        bool_mask = _is_bool_like_mask(df_prev[col])
        if bool_mask.any():
            raise ValueError(
                f"Cannot --resume {csv_path}: column '{col}' contains boolean values. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )
        col_num = _safe_to_numeric(df_prev[col])
        if (
            col_num.isna().any()
            or not np.isfinite(col_num).all()
            or (col_num < 0).any()
            or (col_num > MAX_SAFE_INT).any()
            or (col_num != np.floor(col_num)).any()
        ):
            raise ValueError(
                f"Cannot --resume {csv_path}: column '{col}' contains invalid/missing values. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )
        vals_int = col_num.astype(np.int64).unique().tolist()
        if vals_int != [expected_int]:
            raise ValueError(
                f"Cannot --resume {csv_path}: existing '{col}'={vals_int} != current {expected_int}. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )

    nid_bool = _is_bool_like_mask(df_prev["node_id"])
    if nid_bool.any():
        raise ValueError(
            f"Cannot --resume {csv_path}: column 'node_id' contains boolean values. "
            f"Use a fresh --out-dir or omit --resume to overwrite."
        )
    nid_num = _safe_to_numeric(df_prev["node_id"])
    if (
        nid_num.isna().any()
        or not np.isfinite(nid_num).all()
        or (nid_num < 0).any()
        or (nid_num > MAX_SAFE_INT).any()
        or (nid_num != np.floor(nid_num)).any()
    ):
        raise ValueError(
            f"Cannot --resume {csv_path}: column 'node_id' contains invalid/missing values. "
            f"Use a fresh --out-dir or omit --resume to overwrite."
        )
    df_prev["node_id"] = nid_num.astype(np.int64)
    prev_nodes = set(df_prev["node_id"].tolist())
    if prev_nodes != set(node_ids):
        raise ValueError(
            f"Cannot --resume {csv_path}: existing node set (n={len(prev_nodes)}) != current node list (n={len(node_ids)}). "
            f"Use a fresh --out-dir or omit --resume to overwrite."
        )

    if prior_run_entry is not None and "method" in prior_run_entry:
        if prior_run_entry["method"] != method:
            raise ValueError(
                f"Cannot --resume {csv_path}: run_log entry declares method={prior_run_entry['method']!r} "
                f"!= expected '{method}'."
            )

    if prior_run_entry is not None and "csv_file" in prior_run_entry and prior_run_entry["csv_file"] is not None:
        if prior_run_entry["csv_file"] != csv_path.name:
            raise ValueError(
                f"Cannot --resume {csv_path}: run_log entry declares csv_file={prior_run_entry['csv_file']!r} "
                f"!= expected '{csv_path.name}'."
            )

    recorded_hp: Optional[Dict[str, Any]] = None
    if prior_run_entry is not None and "hyperparams" in prior_run_entry:
        raw_hp = prior_run_entry["hyperparams"]
        if raw_hp is None:
            if method in ("gnnexplainer", "pgexplainer"):
                raise ValueError(
                    f"Cannot --resume {csv_path}: existing {method} entry has hyperparams=None, "
                    f"cannot verify compatibility with current hyperparams {expected_hp}."
                )
            recorded_hp = None
        elif isinstance(raw_hp, dict):
            _validate_finite_json(raw_hp, f"{method} hyperparams for '{expected_run_id}'")
            recorded_hp = dict(raw_hp)
        else:
            raise ValueError(
                f"Cannot --resume {csv_path}: existing {method} hyperparams has invalid type {type(raw_hp).__name__}."
            )
    elif old_top_hyperparams is not None and method in old_top_hyperparams:
        raw_top = old_top_hyperparams[method]
        if isinstance(raw_top, dict):
            _validate_finite_json(raw_top, f"top-level {method} hyperparams")
            recorded_hp = dict(raw_top)
        elif raw_top is not None:
            raise ValueError(
                f"Cannot --resume {csv_path}: top-level {method} hyperparams has invalid type {type(raw_top).__name__}."
            )

    if method == "saliency" and recorded_hp is None:
        recorded_hp = {}

    if method in ("gnnexplainer", "pgexplainer") and recorded_hp is None:
        raise ValueError(
            f"Cannot --resume {csv_path}: missing verified {method} hyperparams in run_log.json "
            f"(cannot verify compatibility with current hyperparams {expected_hp}). "
            f"Re-run without --resume to overwrite cleanly."
        )

    if recorded_hp is not None and not _hyperparams_equal(recorded_hp, expected_hp):
        raise ValueError(
            f"Cannot --resume {csv_path}: existing {method} hyperparams {recorded_hp} "
            f"!= current hyperparams {expected_hp}. Use a fresh --out-dir or omit --resume to overwrite."
        )

    if (
        prior_run_entry is not None
        and "csv_sha256" in prior_run_entry
        and prior_run_entry["csv_sha256"] is not None
    ):
        recorded_sha = prior_run_entry["csv_sha256"]
        if not isinstance(recorded_sha, str) or not SHA256_RE.fullmatch(recorded_sha):
            raise ValueError(
                f"Cannot --resume {csv_path}: invalid csv_sha256={recorded_sha!r} in run_log.json. "
                f"Use a fresh --out-dir or omit --resume to overwrite."
            )
        actual_sha = sha256_file(csv_path)
        if actual_sha != recorded_sha:
            raise ValueError(
                f"Cannot --resume {csv_path}: CSV SHA-256 ({actual_sha}) != run_log.json ({recorded_sha}). "
                f"Possible tampering or file corruption; omit --resume to regenerate."
            )

    return df_prev, recorded_hp


def _select_seeds_for_method(
    method: str,
    gnn_seed: int,
    explainer_seeds: List[int],
    protocol_saliency: bool,
    primary_gnn_seed: int,
) -> List[int]:
    if not protocol_saliency or method != "saliency":
        return list(explainer_seeds)
    # Section 4 & 7.2: Saliency is deterministic; run e=0,1 on g=1000 (primary), e=0 on others.
    if gnn_seed == primary_gnn_seed:
        wanted = [s for s in (0, 1) if s in explainer_seeds]
        return wanted if wanted else explainer_seeds[:2]
    return [0] if 0 in explainer_seeds else [explainer_seeds[0]]


def run_single_configuration(
    *,
    dataset: str,
    split: str,
    arch: str,
    gnn_seed: int,
    explainer_seed: int,
    method: str,
    model: torch.nn.Module,
    data: Dict[str, Any],
    preds: torch.Tensor,
    node_ids: List[int],
    candidates_by_node: Dict[int, Any],
    checkpoint_sha256: str,
    device: torch.device,
    gnn_cfg: Dict[str, Any],
    pg_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    run_id = f"{dataset}_{arch}_g{gnn_seed}_{method}_e{explainer_seed}_{split}"
    num_nodes_graph = int(data["x"].size(0))
    if not node_ids:
        raise ValueError("node_ids must be a non-empty list of valid node indices.")
    seen_nids = set()
    for nid in node_ids:
        if isinstance(nid, bool) or not isinstance(nid, int) or nid < 0 or nid >= num_nodes_graph:
            raise ValueError(
                f"Invalid node_id {nid!r} for graph with num_nodes={num_nodes_graph}."
            )
        if nid in seen_nids:
            raise ValueError(f"Duplicate node_id {nid} in node_ids.")
        seen_nids.add(nid)
    t_run_start = time.perf_counter()

    pg_train_sec: Optional[float] = None
    explainer: Any
    if method == "saliency":
        explainer = SaliencyExplainer()
    elif method == "gnnexplainer":
        explainer = GNNExplainer()
    elif method == "pgexplainer":
        # Defensive F1 ordering: seed BEFORE creating PGExplainerModel MLP
        set_global_seed(explainer_seed)
        pg_model = PGExplainerModel(model=model, device=device)
        t_pg0 = time.perf_counter()
        pg_model.train_on_graph(
            data=data,
            epochs=int(pg_cfg["epochs"]),
            lr=float(pg_cfg["lr"]),
            temp_init=float(pg_cfg["temp_init"]),
            temp_min=float(pg_cfg["temp_min"]),
            seed=explainer_seed,
        )
        pg_train_sec = time.perf_counter() - t_pg0
        explainer = PGExplainer(trained_model=pg_model)
    else:
        raise ValueError(f"Unsupported method: {method}")

    rows: List[Dict[str, Any]] = []
    node_runtimes: List[float] = []
    n_success = 0
    n_empty = 0
    n_failed = 0
    failed_details: List[Dict[str, Any]] = []

    for node_id in node_ids:
        target_class = int(preds[node_id].item())
        cand = candidates_by_node[node_id]

        if method == "gnnexplainer":
            set_global_seed(explainer_seed)
            res = explainer.explain(
                model=model,
                data=data,
                node_id=node_id,
                target_class=target_class,
                candidate_edges=cand,
                config=gnn_cfg,
                seed=explainer_seed,
            )
        elif method == "saliency":
            res = explainer.explain(
                model=model,
                data=data,
                node_id=node_id,
                target_class=target_class,
                candidate_edges=cand,
                config={},
                seed=explainer_seed,
            )
        else:
            # Handle empty candidates consistently for PGExplainer too
            if cand.num_candidate_edges == 0:
                res = explainer.explain(
                    model=model,
                    data=data,
                    node_id=node_id,
                    target_class=target_class,
                    candidate_edges=cand,
                    config=pg_cfg,
                    seed=explainer_seed,
                )
                if res.status == "success":
                    res.status = "empty_candidates"
            else:
                res = explainer.explain(
                    model=model,
                    data=data,
                    node_id=node_id,
                    target_class=target_class,
                    candidate_edges=cand,
                    config=pg_cfg,
                    seed=explainer_seed,
                )

        if isinstance(res.runtime_sec, (bool, np.bool_)):
            rt = 0.0
        else:
            try:
                rt = float(res.runtime_sec)
            except (TypeError, ValueError):
                rt = 0.0
        if not math.isfinite(rt) or rt < 0.0:
            rt = 0.0
        node_runtimes.append(rt)

        if res.status == "success":
            lengths_ok = (
                len(res.edge_ids) == len(cand.candidate_pairs) == len(res.scores)
                and len(res.scores) > 0
                and list(res.edge_ids) == list(cand.edge_ids)
            )
            scores_float: List[float] = []
            scores_ok = False
            if lengths_ok:
                try:
                    scores_float = [float(s) for s in res.scores]
                    scores_ok = all(
                        not isinstance(s, (bool, np.bool_, str)) and math.isfinite(sf)
                        for s, sf in zip(res.scores, scores_float)
                    )
                except (TypeError, ValueError):
                    scores_ok = False

            if not lengths_ok or not scores_ok:
                reason = (
                    f"Edge ID / length mismatch: edge_ids={len(res.edge_ids)}, "
                    f"candidate_pairs={len(cand.candidate_pairs)}, scores={len(res.scores)}"
                    if not lengths_ok
                    else "Non-finite or boolean importance_score returned with status='success'."
                )
                n_failed += 1
                failed_details.append({"node_id": node_id, "error": reason})
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "split": split,
                        "arch": arch,
                        "gnn_seed": gnn_seed,
                        "checkpoint_sha256": checkpoint_sha256,
                        "method": method,
                        "explainer_seed": explainer_seed,
                        "seed": explainer_seed,
                        "node_id": node_id,
                        "target_class": target_class,
                        "edge_id": -1,
                        "u": -1,
                        "v": -1,
                        "importance_score": float("nan"),
                        "runtime_sec": rt,
                        "status": "failed",
                    }
                )
            else:
                n_success += 1
                for eid, (u, v), score in zip(res.edge_ids, cand.candidate_pairs, scores_float):
                    rows.append(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "split": split,
                            "arch": arch,
                            "gnn_seed": gnn_seed,
                            "checkpoint_sha256": checkpoint_sha256,
                            "method": method,
                            "explainer_seed": explainer_seed,
                            "seed": explainer_seed,
                            "node_id": node_id,
                            "target_class": target_class,
                            "edge_id": int(eid),
                            "u": int(u),
                            "v": int(v),
                            "importance_score": score,
                            "runtime_sec": rt,
                            "status": "success",
                        }
                    )
        elif res.status == "empty_candidates":
            n_empty += 1
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split,
                    "arch": arch,
                    "gnn_seed": gnn_seed,
                    "checkpoint_sha256": checkpoint_sha256,
                    "method": method,
                    "explainer_seed": explainer_seed,
                    "seed": explainer_seed,
                    "node_id": node_id,
                    "target_class": target_class,
                    "edge_id": -1,
                    "u": -1,
                    "v": -1,
                    "importance_score": float("nan"),
                    "runtime_sec": rt,
                    "status": "empty_candidates",
                }
            )
        else:
            n_failed += 1
            failed_details.append({"node_id": node_id, "error": res.error_message})
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split,
                    "arch": arch,
                    "gnn_seed": gnn_seed,
                    "checkpoint_sha256": checkpoint_sha256,
                    "method": method,
                    "explainer_seed": explainer_seed,
                    "seed": explainer_seed,
                    "node_id": node_id,
                    "target_class": target_class,
                    "edge_id": -1,
                    "u": -1,
                    "v": -1,
                    "importance_score": float("nan"),
                    "runtime_sec": rt,
                    "status": "failed",
                }
            )

    df = pd.DataFrame(rows, columns=SCHEMA_V2_COLUMNS)
    total_sec = time.perf_counter() - t_run_start
    rt_arr = np.asarray(node_runtimes, dtype=float)
    stats: Dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        "explainer_seed": explainer_seed,
        "hyperparams": _method_hyperparams(method, gnn_cfg, pg_cfg),
        "n_nodes": len(node_ids),
        "n_rows": len(df),
        "n_success_nodes": n_success,
        "n_empty_candidates_nodes": n_empty,
        "n_failed_nodes": n_failed,
        "failed_details": failed_details,
        "pg_train_sec": round(pg_train_sec, 6) if pg_train_sec is not None else None,
        "total_runtime_sec": round(total_sec, 6),
        "mean_node_runtime_sec": round(float(np.mean(rt_arr)), 6),
        "median_node_runtime_sec": round(float(np.median(rt_arr)), 6),
        "min_node_runtime_sec": round(float(np.min(rt_arr)), 6),
        "max_node_runtime_sec": round(float(np.max(rt_arr)), 6),
    }
    return df, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["bashapes", "cora"])
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--arch", default="gcn", choices=["gcn"])
    parser.add_argument("--gnn-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--explainer-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(SUPPORTED_METHODS),
        default=list(SUPPORTED_METHODS),
    )
    parser.add_argument("--data-dir", type=pathlib.Path, default=ROOT / "data")
    parser.add_argument("--checkpoints-dir", type=pathlib.Path, default=ROOT / "results" / "checkpoints")
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        default=None,
        help="Optional explicit .pt path (valid only with a single --gnn-seeds value).",
    )
    parser.add_argument("--nodes-file", type=pathlib.Path, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "results" / "scores")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--protocol-saliency", action="store_true")
    parser.add_argument(
        "--primary-gnn-seed",
        type=int,
        default=1000,
        help="GNN seed treated as primary under --protocol-saliency (runs Saliency e=0,1; others run e=0).",
    )
    parser.add_argument("--resume", action="store_true")

    # GNNExplainer hyperparameters (Section 4 & F10)
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--gnn-coeff-size", type=float, default=0.005)
    parser.add_argument("--gnn-coeff-ent", type=float, default=0.1)
    parser.add_argument("--gnn-size-loss-mode", choices=["mean", "sum"], default="mean")

    # PGExplainer hyperparameters
    parser.add_argument("--pg-epochs", type=int, default=40)
    parser.add_argument("--pg-lr", type=float, default=0.005)
    parser.add_argument("--pg-temp-init", type=float, default=5.0)
    parser.add_argument("--pg-temp-min", type=float, default=0.5)

    args = parser.parse_args(argv)

    # Deduplicate CLI lists while preserving order and validate numeric bounds
    args.gnn_seeds = list(dict.fromkeys(args.gnn_seeds))
    args.explainer_seeds = list(dict.fromkeys(args.explainer_seeds))
    args.methods = list(dict.fromkeys(args.methods))

    if any(s < 0 or s > MAX_SAFE_INT for s in args.gnn_seeds) or any(
        s < 0 or s > MAX_SAFE_INT for s in args.explainer_seeds
    ):
        parser.error(f"--gnn-seeds and --explainer-seeds must be integers in [0, {MAX_SAFE_INT}].")
    if args.primary_gnn_seed < 0 or args.primary_gnn_seed > MAX_SAFE_INT:
        parser.error(f"--primary-gnn-seed must be an integer in [0, {MAX_SAFE_INT}].")
    if (
        args.gnn_epochs < 0
        or not math.isfinite(args.gnn_lr)
        or args.gnn_lr <= 0
        or not math.isfinite(args.gnn_coeff_size)
        or args.gnn_coeff_size < 0
        or not math.isfinite(args.gnn_coeff_ent)
        or args.gnn_coeff_ent < 0
    ):
        parser.error(
            "Invalid GNNExplainer hyperparameters (epochs >= 0, finite lr > 0, finite coeffs >= 0)."
        )
    if (
        args.pg_epochs < 1
        or not math.isfinite(args.pg_lr)
        or args.pg_lr <= 0
        or not math.isfinite(args.pg_temp_init)
        or args.pg_temp_init <= 0
        or not math.isfinite(args.pg_temp_min)
        or args.pg_temp_min <= 0
        or args.pg_temp_min > args.pg_temp_init
    ):
        parser.error(
            "Invalid PGExplainer hyperparameters (epochs >= 1, finite lr > 0, finite 0 < pg_temp_min <= pg_temp_init)."
        )

    if args.checkpoint_path is not None and len(args.gnn_seeds) != 1:
        parser.error("--checkpoint-path can only be used with a single --gnn-seeds value.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested via --device cuda, but CUDA is not available.")
    device = torch.device(args.device)

    data_pt = args.data_dir / args.dataset / "data.pt"
    if not data_pt.is_file():
        raise FileNotFoundError(f"Dataset file not found or not a regular file: {data_pt}")
    data_sha = sha256_file(data_pt)
    data = torch.load(data_pt, map_location=device, weights_only=False)
    num_nodes_graph = int(data["x"].size(0))

    node_ids, nodes_path, nodes_sha = load_node_list(
        dataset=args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        nodes_file=args.nodes_file,
        max_nodes=args.max_nodes,
        num_nodes=num_nodes_graph,
    )

    # Precompute 2-hop candidate edges once so every (g, method, e) shares exact edge_ids
    candidates_by_node = {
        nid: extract_candidate_edges(data["edge_index"], nid, num_hops=2, num_nodes=num_nodes_graph)
        for nid in node_ids
    }

    git_commit, git_dirty = get_git_info()
    primary_gnn_seed = args.primary_gnn_seed
    gnn_cfg = {
        "epochs": args.gnn_epochs,
        "lr": args.gnn_lr,
        "coeff_size": args.gnn_coeff_size,
        "coeff_ent": args.gnn_coeff_ent,
        "size_loss_mode": args.gnn_size_loss_mode,
    }
    pg_cfg = {
        "epochs": args.pg_epochs,
        "lr": args.pg_lr,
        "temp_init": args.pg_temp_init,
        "temp_min": args.pg_temp_min,
    }

    for gnn_seed in args.gnn_seeds:
        ckpt_path = args.checkpoint_path or (
            args.checkpoints_dir / args.dataset / f"{args.arch}_s{gnn_seed}.pt"
        )
        model, saved_pred, ckpt_sha = load_checkpoint_strict(ckpt_path, args.arch, device)
        with torch.no_grad():
            preds = model(data["x"].to(device), data["edge_index"].to(device)).argmax(dim=-1).cpu()
        if saved_pred is not None:
            if saved_pred.is_floating_point() or saved_pred.is_complex() or saved_pred.dtype == torch.bool:
                raise ValueError(
                    f"Saved ck['pred'] in {ckpt_path} has non-integer dtype {saved_pred.dtype}."
                )
            if not torch.equal(preds, saved_pred.to(dtype=torch.long)):
                raise ValueError(
                    f"Recomputed predictions from {ckpt_path} do not match saved ck['pred']!"
                )

        seed_out_dir = args.out_dir / args.dataset / f"{args.arch}_s{gnn_seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        run_log_path = seed_out_dir / "run_log.json"

        # Preserve prior runs and untouched method hyperparams when merging Khoi + Thien runs
        existing_runs: Dict[str, Any] = {}
        old_top_hyperparams: Optional[Dict[str, Any]] = None
        merged_hyperparams: Dict[str, Any] = {}
        same_context = False
        if _path_entry_exists(run_log_path):
            if not run_log_path.is_file():
                raise ValueError(f"Corrupted {run_log_path}: expected regular file, got non-file.")
            try:
                old_log = json.loads(run_log_path.read_text(encoding="utf-8"))
            except Exception as exc:
                if args.resume:
                    raise ValueError(f"Cannot --resume: corrupted {run_log_path}: {exc}") from exc
                old_log = None

            if isinstance(old_log, dict):
                if args.resume:
                    _validate_finite_json(old_log, str(run_log_path))
                old_gnn_seed = old_log.get("gnn_seed")
                old_nids = old_log.get("node_ids")
                same_context = (
                    old_log.get("dataset") == args.dataset
                    and old_log.get("arch") == args.arch
                    and isinstance(old_gnn_seed, int)
                    and not isinstance(old_gnn_seed, bool)
                    and old_gnn_seed == gnn_seed
                    and old_log.get("checkpoint_sha256") == ckpt_sha
                    and old_log.get("split") == args.split
                    and old_log.get("node_list_sha256") == nodes_sha
                    and isinstance(old_nids, list)
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in old_nids)
                    and old_nids == node_ids
                )
                if same_context:
                    _validate_finite_json(old_log, str(run_log_path))
                    raw_runs = old_log.get("runs_by_id", {})
                    if not isinstance(raw_runs, dict) or any(
                        not isinstance(k, str) or not k.strip() or k != k.strip() or not isinstance(v, dict)
                        for k, v in raw_runs.items()
                    ):
                        raise ValueError(
                            f"Corrupted runs_by_id in {run_log_path}: expected mapping of run_id -> object."
                        )
                    for rk, rv in raw_runs.items():
                        inner_rid = rv.get("run_id")
                        if inner_rid is not None and (not isinstance(inner_rid, str) or inner_rid != rk):
                            raise ValueError(
                                f"Corrupted runs_by_id in {run_log_path}: key '{rk}' disagrees with "
                                f"inner run_id={inner_rid!r}."
                            )
                        parsed_rid = _parse_canonical_run_id(rk)
                        if parsed_rid is None:
                            raise ValueError(
                                f"Corrupted runs_by_id in {run_log_path}: run_id '{rk}' "
                                f"does not match canonical run_id format."
                            )
                        r_ds, r_arch, r_g, rid_method, _r_e, r_sp = parsed_rid
                        if (r_ds, r_arch, r_g, r_sp) != (args.dataset, args.arch, gnn_seed, args.split):
                            raise ValueError(
                                f"Corrupted runs_by_id in {run_log_path}: run_id '{rk}' context "
                                f"({r_ds}, {r_arch}, g={r_g}, {r_sp}) does not match current run "
                                f"({args.dataset}, {args.arch}, g={gnn_seed}, {args.split})."
                            )
                        if "method" in rv:
                            declared_method = rv["method"]
                            if not isinstance(declared_method, str) or declared_method not in ALLOWED_LOG_METHODS:
                                raise ValueError(
                                    f"Corrupted runs_by_id in {run_log_path}: run '{rk}' has unsupported "
                                    f"method={declared_method!r}."
                                )
                            if declared_method != rid_method:
                                raise ValueError(
                                    f"Corrupted runs_by_id in {run_log_path}: run '{rk}' declares "
                                    f"mismatched method={declared_method!r} != canonical '{rid_method}'."
                                )
                        if "csv_file" in rv and rv["csv_file"] is not None:
                            if not _is_flat_csv_filename(rv["csv_file"]):
                                raise ValueError(
                                    f"Corrupted runs_by_id in {run_log_path}: run '{rk}' has invalid "
                                    f"csv_file={rv['csv_file']!r} (expected flat '*.csv' filename)."
                                )
                        if "n_failed_nodes" in rv:
                            nf = rv["n_failed_nodes"]
                            if not isinstance(nf, int) or isinstance(nf, bool) or nf < 0:
                                raise ValueError(
                                    f"Corrupted runs_by_id in {run_log_path}: run '{rk}' has invalid "
                                    f"n_failed_nodes={nf!r} (expected non-negative integer)."
                                )
                        if "hyperparams" in rv and rv["hyperparams"] is not None:
                            if not isinstance(rv["hyperparams"], dict):
                                raise ValueError(
                                    f"Corrupted runs_by_id in {run_log_path}: run '{rk}' has invalid "
                                    f"hyperparams={rv['hyperparams']!r} (expected JSON object or null)."
                                )
                        _validate_finite_json(rv, f"runs_by_id['{rk}'] in {run_log_path}")
                    existing_runs = dict(raw_runs)
                    if "hyperparams" in old_log and old_log["hyperparams"] is not None:
                        raw_top_hp = old_log["hyperparams"]
                        if not isinstance(raw_top_hp, dict):
                            raise ValueError(
                                f"Corrupted hyperparams in {run_log_path}: expected JSON object or null, "
                                f"got {type(raw_top_hp).__name__}."
                            )
                        for m_k, m_hp in raw_top_hp.items():
                            if not isinstance(m_k, str) or m_k not in ALLOWED_LOG_METHODS:
                                raise ValueError(
                                    f"Corrupted hyperparams in {run_log_path}: unsupported method key {m_k!r}."
                                )
                            if m_hp is not None and not isinstance(m_hp, dict):
                                raise ValueError(
                                    f"Corrupted hyperparams in {run_log_path}: hyperparams['{m_k}'] "
                                    f"must be a JSON object or null, got {type(m_hp).__name__}."
                                )
                            if m_hp is not None:
                                _validate_finite_json(m_hp, f"hyperparams['{m_k}'] in {run_log_path}")
                        old_top_hyperparams = dict(raw_top_hp)
                        merged_hyperparams.update(old_top_hyperparams)
                elif args.resume:
                    old_n_nodes = len(old_nids) if isinstance(old_nids, list) else repr(old_nids)
                    raise ValueError(
                        f"Cannot --resume in {seed_out_dir}: existing run_log.json context "
                        f"(split={old_log.get('split')}, ckpt={str(old_log.get('checkpoint_sha256'))[:12]}, "
                        f"n_nodes={old_n_nodes}) differs from current run "
                        f"(split={args.split}, ckpt={ckpt_sha[:12]}, n_nodes={len(node_ids)}). "
                        f"Use a fresh --out-dir or omit --resume to overwrite."
                    )
            elif args.resume:
                raise ValueError(
                    f"Cannot --resume: corrupted {run_log_path}: expected JSON object, "
                    f"got {type(old_log).__name__}."
                )

        # Prevent mixing hyperparams across retained runs of the same method (with or without --resume)
        if same_context and existing_runs:
            for method in args.methods:
                method_hp = _method_hyperparams(method, gnn_cfg, pg_cfg)
                method_seeds = _select_seeds_for_method(
                    method=method,
                    gnn_seed=gnn_seed,
                    explainer_seeds=args.explainer_seeds,
                    protocol_saliency=args.protocol_saliency,
                    primary_gnn_seed=primary_gnn_seed,
                )
                overwritten_ids = set()
                for s in method_seeds:
                    rid = f"{args.dataset}_{args.arch}_g{gnn_seed}_{method}_e{s}_{args.split}"
                    cpath = seed_out_dir / f"{method}_e{s}.csv"
                    if not (args.resume and cpath.is_file() and cpath.stat().st_size > 0):
                        overwritten_ids.add(rid)
                for rid, rinfo in existing_runs.items():
                    if rid in overwritten_ids:
                        continue
                    rid_method = _extract_method_from_run_id(rid)
                    if rid_method is None:
                        raise ValueError(
                            f"Corrupted run entry in {seed_out_dir}: retained run_id '{rid}' "
                            f"does not match canonical run_id format."
                        )
                    if "method" in rinfo:
                        declared_method = rinfo["method"]
                        if not isinstance(declared_method, str) or declared_method not in ALLOWED_LOG_METHODS:
                            raise ValueError(
                                f"Corrupted run entry '{rid}' in {run_log_path}: "
                                f"unsupported method={declared_method!r}."
                            )
                        if declared_method != rid_method:
                            raise ValueError(
                                f"Corrupted run entry '{rid}' in {run_log_path}: "
                                f"method={declared_method!r} disagrees with canonical run_id method '{rid_method}'."
                            )
                    if rid_method != method:
                        continue
                    old_hp = rinfo.get("hyperparams")
                    if method in ("gnnexplainer", "pgexplainer"):
                        if old_hp is None or not _hyperparams_equal(old_hp, method_hp):
                            raise ValueError(
                                f"Cannot mix {method} hyperparams in {seed_out_dir}: "
                                f"retained run '{rid}' has hyperparams={old_hp} != current {method_hp}. "
                                f"Overwrite all {method} seeds or use a fresh --out-dir."
                            )
                    else:
                        if old_hp is not None and not _hyperparams_equal(old_hp, method_hp):
                            raise ValueError(
                                f"Cannot mix {method} hyperparams in {seed_out_dir}: "
                                f"retained run '{rid}' has hyperparams={old_hp} != current {method_hp}."
                            )

        # Only set/update top-level hyperparams for methods actually requested in this invocation
        if "gnnexplainer" in args.methods:
            merged_hyperparams["gnnexplainer"] = dict(gnn_cfg)
        if "pgexplainer" in args.methods:
            merged_hyperparams["pgexplainer"] = dict(pg_cfg)

        try:
            ckpt_rel = ckpt_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            ckpt_rel = ckpt_path.resolve().as_posix()
        try:
            nodes_rel = nodes_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            nodes_rel = nodes_path.resolve().as_posix()

        def flush_run_log() -> None:
            run_log = {
                "schema_version": "v2",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "torch_geometric": torch_geometric.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "device": args.device,
                "dataset": args.dataset,
                "split": args.split,
                "arch": args.arch,
                "gnn_seed": gnn_seed,
                "data_sha256": data_sha,
                "checkpoint_path": ckpt_rel,
                "checkpoint_sha256": ckpt_sha,
                "node_list_path": nodes_rel,
                "node_list_sha256": nodes_sha,
                "n_nodes": len(node_ids),
                "node_ids": node_ids,
                "hyperparams": merged_hyperparams,
                "last_invocation_runtime_sec": round(time.perf_counter() - t_seed_start, 6),
                "total_failed_nodes": sum(int(r.get("n_failed_nodes", 0)) for r in existing_runs.values()),
                "runs_by_id": existing_runs,
            }
            _atomic_write_json(run_log, run_log_path)

        t_seed_start = time.perf_counter()
        for method in args.methods:
            method_hp = _method_hyperparams(method, gnn_cfg, pg_cfg)
            method_seeds = _select_seeds_for_method(
                method=method,
                gnn_seed=gnn_seed,
                explainer_seeds=args.explainer_seeds,
                protocol_saliency=args.protocol_saliency,
                primary_gnn_seed=primary_gnn_seed,
            )
            for expl_seed in method_seeds:
                csv_path = seed_out_dir / f"{method}_e{expl_seed}.csv"
                run_id = f"{args.dataset}_{args.arch}_g{gnn_seed}_{method}_e{expl_seed}_{args.split}"
                if _path_entry_exists(csv_path) and not csv_path.is_file():
                    raise ValueError(f"Output CSV path exists and is not a regular file: {csv_path}")
                if args.resume and csv_path.is_file() and csv_path.stat().st_size > 0:
                    df_prev, verified_hp = _validate_resumable_csv(
                        csv_path=csv_path,
                        expected_run_id=run_id,
                        dataset=args.dataset,
                        split=args.split,
                        arch=args.arch,
                        gnn_seed=gnn_seed,
                        method=method,
                        explainer_seed=expl_seed,
                        ckpt_sha=ckpt_sha,
                        node_ids=node_ids,
                        expected_hp=method_hp,
                        prior_run_entry=existing_runs.get(run_id),
                        old_top_hyperparams=old_top_hyperparams,
                    )
                    print(f"[SKIP --resume] {csv_path}")
                    if run_id not in existing_runs:
                        existing_runs[run_id] = {
                            "run_id": run_id,
                            "method": method,
                            "explainer_seed": expl_seed,
                            "hyperparams": verified_hp,
                            "n_nodes": int(df_prev["node_id"].nunique()),
                            "n_rows": int(len(df_prev)),
                            "n_success_nodes": int(
                                df_prev.loc[df_prev["status"] == "success", "node_id"].nunique()
                            ),
                            "n_empty_candidates_nodes": int(
                                df_prev.loc[df_prev["status"] == "empty_candidates", "node_id"].nunique()
                            ),
                            "n_failed_nodes": int(
                                df_prev.loc[df_prev["status"] == "failed", "node_id"].nunique()
                            ),
                            "csv_file": csv_path.name,
                            "csv_sha256": sha256_file(csv_path),
                            "resumed_from_existing_csv": True,
                        }
                        flush_run_log()
                    continue

                print(
                    f"[RUN] dataset={args.dataset} split={args.split} g={gnn_seed} "
                    f"method={method} e={expl_seed} n_nodes={len(node_ids)} device={args.device}"
                )
                df, stats = run_single_configuration(
                    dataset=args.dataset,
                    split=args.split,
                    arch=args.arch,
                    gnn_seed=gnn_seed,
                    explainer_seed=expl_seed,
                    method=method,
                    model=model,
                    data=data,
                    preds=preds,
                    node_ids=node_ids,
                    candidates_by_node=candidates_by_node,
                    checkpoint_sha256=ckpt_sha,
                    device=device,
                    gnn_cfg=gnn_cfg,
                    pg_cfg=pg_cfg,
                )
                _atomic_write_csv(df, csv_path)
                stats["csv_file"] = csv_path.name
                stats["csv_sha256"] = sha256_file(csv_path)
                existing_runs[run_id] = stats
                flush_run_log()
                print(
                    f"  -> Saved {csv_path.name} ({stats['n_rows']} rows, "
                    f"total={stats['total_runtime_sec']:.2f}s, "
                    f"mean={stats['mean_node_runtime_sec']:.4f}s/node, "
                    f"failed={stats['n_failed_nodes']})"
                )

        flush_run_log()
        print(f"[LOG] Updated {run_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
