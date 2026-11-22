"""Handover H2 Score CSV Validator (Khoi - PN2).

Validates Schema v2 edge score CSV files and run_log.json according to
Sections 6.4 and 6.6 of KE_HOACH_CUOI_KY_FINAL.md before handover from PN2 to PN3.

Usage:
    python scripts/check_scores.py results/scores/bashapes
    python scripts/check_scores.py results/scores/bashapes --checkpoints-dir results/checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

REQUIRED_COLUMNS: List[str] = [
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

ALLOWED_STATUSES: Set[str] = {"success", "failed", "empty_candidates"}
ALLOWED_SPLITS: Set[str] = {"test", "val"}
ALLOWED_ARCHS: Set[str] = {"gcn", "gat"}
ALLOWED_METHODS: Set[str] = {
    "saliency",
    "gnnexplainer",
    "pgexplainer",
    "random",
    "nearest_first",
}
DATASET_RE = re.compile(r"^[a-z0-9_]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_METHOD_RE = re.compile(
    r"^[a-z0-9_]+_(?:gcn|gat)_g\d+_(saliency|gnnexplainer|pgexplainer|random|nearest_first)_e\d+_(?:test|val)$"
)
MAX_SAFE_INT: int = (1 << 53) - 1  # 9007199254740991: exact float64 & int64 safe upper bound


def _extract_method_from_run_id(run_id: str) -> Optional[str]:
    m = RUN_ID_METHOD_RE.fullmatch(run_id)
    return m.group(1) if m else None


def _extract_valid_csv_sha256(
    rinfo: Dict[str, Any],
    rid: str,
    lf_str: str,
    add_error: Any,
) -> Optional[str]:
    if "csv_sha256" not in rinfo or rinfo["csv_sha256"] is None:
        return None
    val = rinfo["csv_sha256"]
    if not isinstance(val, str) or not SHA256_RE.fullmatch(val):
        add_error(
            lf_str,
            "run_log",
            f"Run '{rid}' has invalid csv_sha256={val!r} (expected 64 lowercase hex characters).",
        )
        return None
    return val


def _extract_valid_csv_file(
    rinfo: Dict[str, Any],
    src_label: str,
    rid_label: str,
    add_error: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate optional 'csv_file' field on a run_log entry.

    Returns (is_valid, csv_file_or_None). Rejects falsy/non-string/whitespace-padded/NUL values.
    """
    if not isinstance(rinfo, dict):
        return True, None
    if "csv_file" not in rinfo or rinfo["csv_file"] is None:
        return True, None
    val = rinfo["csv_file"]
    if not isinstance(val, str) or not val.strip() or val != val.strip() or "\x00" in val:
        if add_error is not None:
            add_error(
                src_label,
                "run_log",
                f"Run '{rid_label}' has invalid csv_file={val!r} "
                f"(expected non-empty trimmed string without NUL bytes).",
            )
        return False, None
    return True, val


def _reject_json_nonfinite(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {token!r} is not allowed.")


def _loads_strict_json(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_nonfinite)


def _is_flat_csv_filename(name: str) -> bool:
    """Return True iff name is a plain flat .csv filename with no separators, '..', ':', or drive."""
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
        if p_posix.name != name or p_posix.suffix.lower() != ".csv" or not p_posix.stem:
            return False
        return pathlib.Path(name).name == name
    except ValueError:
        return False


def _is_safe_relative_csv_path(raw_path: str) -> bool:
    """Return True iff raw_path is a safe relative .csv path without NUL, drive, root, '.', or '..'."""
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip() or "\x00" in raw_path:
        return False
    if any(ord(ch) < 32 for ch in raw_path):
        return False
    if raw_path.startswith(("/", "\\")) or ":" in raw_path or raw_path in {".", ".."}:
        return False
    p_posix = pathlib.PurePosixPath(raw_path)
    p_win = pathlib.PureWindowsPath(raw_path)
    if p_posix.is_absolute() or p_win.is_absolute() or bool(p_win.drive) or bool(p_win.root):
        return False
    if ".." in p_posix.parts or ".." in p_win.parts:
        return False
    raw_segments = re.split(r"[/\\]", raw_path)
    if any(
        seg in {"", ".", ".."} or seg != seg.strip() or seg.endswith(".")
        for seg in raw_segments
    ):
        return False
    return _is_flat_csv_filename(raw_segments[-1])


def _canonical_hp_or_error(
    hp_val: Any,
    lf_str: str,
    label: str,
    add_error: Any,
) -> Tuple[bool, Optional[str]]:
    if hp_val is None:
        return True, None
    if not isinstance(hp_val, dict):
        add_error(lf_str, "run_log", f"{label} has invalid non-object hyperparams: {hp_val!r}.")
        return False, None
    try:
        return True, json.dumps(hp_val, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        add_error(lf_str, "run_log", f"{label} has non-serializable/non-finite hyperparams: {exc}.")
        return False, None


def _normalize_runs_dict(
    entry: Dict[str, Any],
    lf_str: str,
    add_error: Any,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Normalize 'runs_by_id' dict or 'runs' list into a {run_id: run_dict} mapping."""
    has_rbi = "runs_by_id" in entry
    has_runs = "runs" in entry
    if has_rbi and has_runs:
        add_error(
            lf_str,
            "run_log",
            "Ambiguous run_log.json entry contains both 'runs_by_id' and 'runs'.",
        )
        return None

    if has_rbi:
        runs_by_id = entry["runs_by_id"]
        if not isinstance(runs_by_id, dict) or len(runs_by_id) == 0:
            add_error(
                lf_str,
                "run_log",
                f"'runs_by_id' must be a non-empty JSON object, got {type(runs_by_id).__name__}.",
            )
            return None
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in runs_by_id.items():
            if not isinstance(k, str) or not k.strip() or k != k.strip() or not isinstance(v, dict):
                add_error(lf_str, "run_log", f"Invalid runs_by_id entry for key {k!r}.")
                continue
            inner_rid = v.get("run_id")
            if inner_rid is not None and (not isinstance(inner_rid, str) or inner_rid != k):
                add_error(
                    lf_str,
                    "run_log",
                    f"runs_by_id key '{k}' disagrees with inner run_id {inner_rid!r}.",
                )
            out[k] = v
        if not out:
            add_error(lf_str, "run_log", "'runs_by_id' contains no valid run entries.")
            return None
        return out

    if has_runs:
        runs_list = entry["runs"]
        if not isinstance(runs_list, list) or len(runs_list) == 0:
            add_error(
                lf_str,
                "run_log",
                f"'runs' must be a non-empty JSON list, got {type(runs_list).__name__}.",
            )
            return None
        out = {}
        for idx, item in enumerate(runs_list):
            if not isinstance(item, dict):
                add_error(lf_str, "run_log", f"runs[{idx}] must be a JSON object.")
                continue
            rid = item.get("run_id")
            if not isinstance(rid, str) or not rid.strip() or rid != rid.strip():
                add_error(lf_str, "run_log", f"runs[{idx}] is missing a valid non-empty 'run_id'.")
                continue
            if rid in out:
                add_error(lf_str, "run_log", f"Duplicate run_id '{rid}' inside run_log.json 'runs'.")
                continue
            out[rid] = item
        if not out:
            add_error(lf_str, "run_log", "'runs' contains no valid run entries.")
            return None
        return out

    return None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _is_nonneg_int_series(series: pd.Series) -> Tuple[bool, List[int]]:
    """Return (all_valid, bad_csv_line_numbers)."""
    bool_mask = _is_bool_like_mask(series)
    if bool_mask.all():
        return False, [int(idx) + 2 for idx in series.index]
    clean = series.astype(object).where(~bool_mask, np.nan)
    num = _safe_to_numeric(clean)
    bad_mask = (
        bool_mask
        | num.isna()
        | ~np.isfinite(num)
        | (num < 0)
        | (num > MAX_SAFE_INT)
        | (num != np.floor(num))
    )
    if not bad_mask.any():
        return True, []
    bad_lines = [int(idx) + 2 for idx in series.index[bad_mask]]  # header is line 1
    return False, bad_lines


def _is_finite_float_series(
    series: pd.Series,
    *,
    non_negative: bool = False,
) -> Tuple[bool, List[int]]:
    """Return (all_valid, bad_csv_line_numbers) for finite float values, rejecting booleans."""
    bool_mask = _is_bool_like_mask(series)
    if bool_mask.all():
        return False, [int(idx) + 2 for idx in series.index]
    clean = series.astype(object).where(~bool_mask, np.nan)
    num = _safe_to_numeric(clean)
    bad_mask = bool_mask | num.isna() | ~np.isfinite(num)
    if non_negative:
        bad_mask = bad_mask | (num < 0)
    if not bad_mask.any():
        return True, []
    return False, [int(idx) + 2 for idx in series.index[bad_mask]]


def _is_valid_nonneg_int_scalar(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_run_log_entry(
    entry: Any,
    lf_str: str,
    add_error: Any,
    *,
    require_runs_container: bool = False,
) -> Optional[Tuple[str, str, str, int, str]]:
    """Validate minimum schema of a run_log entry and return (dataset, arch, split, gnn_seed, ckpt_sha)."""
    if not isinstance(entry, dict) or not entry:
        add_error(lf_str, "run_log", "run_log.json entry must be a non-empty JSON object.")
        return None

    missing = [k for k in ("dataset", "gnn_seed", "split", "checkpoint_sha256") if k not in entry]
    if missing:
        add_error(
            lf_str,
            "run_log",
            f"run_log.json missing required fields: {missing}.",
        )
        return None

    ds_raw = entry["dataset"]
    arch_raw = entry.get("arch", "gcn")
    sp_raw = entry["split"]
    g_raw = entry["gnn_seed"]
    sha_raw = entry["checkpoint_sha256"]

    errs: List[str] = []
    if not isinstance(ds_raw, str) or ds_raw != ds_raw.strip() or not DATASET_RE.fullmatch(ds_raw):
        errs.append(f"invalid dataset={ds_raw!r}")
    if not isinstance(arch_raw, str) or arch_raw != arch_raw.strip() or arch_raw not in ALLOWED_ARCHS:
        errs.append(f"invalid arch={arch_raw!r}")
    if not isinstance(sp_raw, str) or sp_raw not in ALLOWED_SPLITS:
        errs.append(f"invalid split={sp_raw!r}")
    if not _is_valid_nonneg_int_scalar(g_raw):
        errs.append(f"invalid gnn_seed={g_raw!r}")
    if not isinstance(sha_raw, str) or not SHA256_RE.fullmatch(sha_raw):
        errs.append(f"invalid checkpoint_sha256={sha_raw!r}")

    if "hyperparams" in entry and entry["hyperparams"] is not None and not isinstance(entry["hyperparams"], dict):
        errs.append(f"invalid hyperparams={type(entry['hyperparams']).__name__} (expected object or null)")

    if "node_ids" in entry:
        raw_nids = entry["node_ids"]
        if (
            not isinstance(raw_nids, list)
            or len(raw_nids) == 0
            or not all(_is_valid_nonneg_int_scalar(x) and x <= MAX_SAFE_INT for x in raw_nids)
            or len(set(raw_nids)) != len(raw_nids)
        ):
            errs.append("invalid 'node_ids' (expected non-empty list of unique non-negative integers)")

    if "checkpoint_path" in entry and entry["checkpoint_path"] is not None:
        cp_raw = entry["checkpoint_path"]
        if not isinstance(cp_raw, str) or not cp_raw.strip() or cp_raw != cp_raw.strip() or "\x00" in cp_raw:
            errs.append(f"invalid checkpoint_path={cp_raw!r}")

    ok_entry_cf, _ = _extract_valid_csv_file(entry, lf_str, str(entry.get("run_id", "<entry>")))
    if not ok_entry_cf:
        errs.append(f"invalid csv_file={entry.get('csv_file')!r}")

    has_rbi = "runs_by_id" in entry
    has_runs = "runs" in entry
    if has_rbi and has_runs:
        errs.append("ambiguous run container: cannot specify both 'runs_by_id' and 'runs'")
    else:
        if has_rbi:
            runs_by_id = entry["runs_by_id"]
            valid_rbi = (
                isinstance(runs_by_id, dict)
                and len(runs_by_id) > 0
                and all(
                    isinstance(k, str)
                    and bool(k.strip())
                    and k == k.strip()
                    and isinstance(v, dict)
                    and (v.get("run_id") is None or (isinstance(v.get("run_id"), str) and v.get("run_id") == k))
                    for k, v in runs_by_id.items()
                )
            )
            if not valid_rbi:
                errs.append("invalid or empty 'runs_by_id' (expected non-empty mapping of run_id -> object)")
            else:
                for k, v in runs_by_id.items():
                    ok_cf, _ = _extract_valid_csv_file(v, lf_str, k)
                    if not ok_cf:
                        errs.append(f"run '{k}' has invalid csv_file={v.get('csv_file')!r}")
        if has_runs:
            runs_list = entry["runs"]
            valid_rl = (
                isinstance(runs_list, list)
                and len(runs_list) > 0
                and all(
                    isinstance(item, dict)
                    and isinstance(item.get("run_id"), str)
                    and bool(item.get("run_id", "").strip())
                    and item.get("run_id") == item.get("run_id", "").strip()
                    for item in runs_list
                )
            )
            if not valid_rl:
                errs.append("invalid or empty 'runs' (expected non-empty list of objects with non-empty 'run_id')")
            else:
                for item in runs_list:
                    rid_item = str(item.get("run_id"))
                    ok_cf, _ = _extract_valid_csv_file(item, lf_str, rid_item)
                    if not ok_cf:
                        errs.append(f"run '{rid_item}' has invalid csv_file={item.get('csv_file')!r}")
        if require_runs_container and not has_rbi and not has_runs:
            errs.append("missing non-empty valid 'runs_by_id' object or 'runs' list")

    if errs:
        add_error(lf_str, "run_log", f"Invalid run_log.json structure: {', '.join(errs)}.")
        return None

    return str(ds_raw), str(arch_raw), str(sp_raw), int(g_raw), str(sha_raw)


def _validate_placeholder_rows(
    sub_df: pd.DataFrame,
    status_name: str,
    fref: str,
    add_error: Any,
) -> None:
    """Ensure empty_candidates / failed placeholder rows carry empty/NaN scores and (-1 or NaN) edge fields."""
    dup = sub_df.duplicated(subset=["node_id"], keep=False)
    if dup.any():
        add_error(
            fref,
            "duplicate_rows",
            f"Found duplicate {status_name} rows for the same node_id.",
            lines=[int(i) + 2 for i in sub_df.index[dup]],
        )
    raw_sc = sub_df["importance_score"]
    bad_sc = raw_sc.notna()
    if bad_sc.any():
        add_error(
            fref,
            "placeholder_row_invalid",
            f"Rows with status='{status_name}' must have empty/NaN importance_score.",
            lines=[int(i) + 2 for i in sub_df.index[bad_sc]],
        )
    for col in ("edge_id", "u", "v"):
        raw = sub_df[col]
        bool_mask = _is_bool_like_mask(raw)
        clean = raw.astype(object).where(~bool_mask, np.nan)
        num = _safe_to_numeric(clean)
        non_na_str = raw.notna() & num.isna()
        bad_edge = bool_mask | non_na_str | (~num.isna() & (num != -1))
        if bad_edge.any():
            add_error(
                fref,
                "placeholder_row_invalid",
                f"Rows with status='{status_name}' must have '{col}' equal to -1 or empty/NaN.",
                lines=[int(i) + 2 for i in sub_df.index[bad_edge]],
            )


def _lookup_run_log(
    run_log_map: Dict[Tuple[str, str, Optional[str], int], Dict[str, Any]],
    dataset: str,
    arch: str,
    split: Optional[str],
    gnn_seed: int,
) -> Optional[Dict[str, Any]]:
    if split is not None and (dataset, arch, split, gnn_seed) in run_log_map:
        return run_log_map[(dataset, arch, split, gnn_seed)]
    return None


def _path_entry_exists(path: pathlib.Path) -> bool:
    """Return True if path exists as any filesystem entry (including broken symlinks and dangling NTFS junctions)."""
    return os.path.lexists(path) or path.exists() or path.is_symlink()


def _scan_scores_tree(
    scores_path: pathlib.Path,
) -> Tuple[List[pathlib.Path], List[pathlib.Path], List[pathlib.Path], List[Tuple[str, str]]]:
    """Cycle-safe recursive scan of a scores directory.

    Returns (csv_files, raw_log_paths, dir_and_link_paths, scan_errors).
    Descends into regular directories and non-cyclic NTFS junctions while pruning cycles
    and skipping dangling links so WinError 1921 / OSError never crashes discovery.
    Records subdirectories and symlink/dangling-junction entries so path_containment catches
    POSIX directory symlinks, cyclic junctions, and dangling junctions uniformly.
    """
    csv_files: List[pathlib.Path] = []
    raw_log_paths: List[pathlib.Path] = []
    dir_and_link_paths: List[pathlib.Path] = []
    scan_errors: List[Tuple[str, str]] = []

    try:
        root_res = scores_path.resolve()
    except (OSError, ValueError) as exc:
        scan_errors.append((str(scores_path), f"Failed to resolve scores directory {scores_path}: {exc}"))
        return csv_files, raw_log_paths, dir_and_link_paths, scan_errors

    def _walk(cur_dir: pathlib.Path, ancestors: Set[pathlib.Path], depth: int) -> None:
        if depth > 64:
            scan_errors.append((str(cur_dir), f"Directory traversal exceeded max depth (64) at {cur_dir}."))
            return
        try:
            with os.scandir(cur_dir) as it:
                entries = sorted(it, key=lambda e: e.name)
        except (OSError, ValueError) as exc:
            scan_errors.append((str(cur_dir), f"Failed to scan directory {cur_dir}: {exc}"))
            return

        for entry in entries:
            child = cur_dir / entry.name
            try:
                is_sym = entry.is_symlink()
            except (OSError, ValueError):
                is_sym = False
            try:
                child_exists = child.exists()
            except (OSError, ValueError):
                child_exists = False
            try:
                is_dir_nofollow = entry.is_dir(follow_symlinks=False)
            except (OSError, ValueError):
                is_dir_nofollow = False

            if is_sym or not child_exists:
                dir_and_link_paths.append(child)

            if child.match("run_log.json"):
                raw_log_paths.append(child)

            if child.match("*.csv") and child_exists:
                try:
                    if child.is_file():
                        csv_files.append(child)
                except (OSError, ValueError) as exc:
                    scan_errors.append((str(child), f"Failed to inspect CSV path {child}: {exc}"))

            if is_dir_nofollow:
                if not is_sym and child_exists:
                    dir_and_link_paths.append(child)
                if not child_exists:
                    continue
                try:
                    child_res = child.resolve()
                except (OSError, ValueError) as exc:
                    scan_errors.append((str(child), f"Failed to resolve subdirectory {child}: {exc}"))
                    continue
                if child_res in ancestors:
                    continue
                _walk(child, ancestors | {child_res}, depth + 1)

    _walk(scores_path, {root_res}, 0)
    csv_files.sort()
    raw_log_paths.sort()
    dir_and_link_paths.sort()
    return csv_files, raw_log_paths, dir_and_link_paths, scan_errors


def _discover_csv_files(scores_path: pathlib.Path) -> List[pathlib.Path]:
    try:
        if scores_path.is_file():
            return [scores_path]
        if not scores_path.exists():
            return []
        csv_files, _, _, _ = _scan_scores_tree(scores_path)
        return csv_files
    except (OSError, ValueError):
        return []


def _resolve_checkpoint_hash(
    dataset: str,
    arch: str,
    gnn_seed: int,
    checkpoints_dir: pathlib.Path,
    explicit_ckpt_path: Optional[pathlib.Path],
    run_log_map: Dict[Tuple[str, str, Optional[str], int], Dict[str, Any]],
    split: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (expected_sha256, source_description, error_message)."""
    if explicit_ckpt_path is not None:
        if not isinstance(explicit_ckpt_path, (str, pathlib.Path)):
            return None, None, f"Invalid explicit checkpoint path: {explicit_ckpt_path!r}"
        try:
            explicit_p = pathlib.Path(explicit_ckpt_path)
            if not explicit_p.is_file():
                return None, None, f"Explicit checkpoint not found or not a file: {explicit_p}"
            return sha256_file(explicit_p), str(explicit_p), None
        except (OSError, ValueError, TypeError) as exc:
            return None, None, f"Failed to read explicit checkpoint {explicit_ckpt_path!r}: {exc}"

    base = checkpoints_dir / dataset / f"{arch}_s{gnn_seed}"
    pt_path = base.with_suffix(".pt")
    json_path = base.with_suffix(".json")

    pt_hash: Optional[str] = None
    try:
        if _path_entry_exists(pt_path):
            if not pt_path.is_file():
                return None, None, f"Checkpoint path is not a regular file: {pt_path}"
            pt_hash = sha256_file(pt_path)
    except (OSError, ValueError) as exc:
        return None, None, f"Failed to read checkpoint {pt_path}: {exc}"

    json_hash: Optional[str] = None
    try:
        json_present = _path_entry_exists(json_path)
        json_is_file = json_path.is_file()
    except (OSError, ValueError) as exc:
        return None, None, f"Failed to inspect checkpoint metadata {json_path}: {exc}"
    if json_present and not json_is_file:
        return None, None, f"Checkpoint metadata path is not a regular file: {json_path}"
    if json_is_file:
        try:
            meta = _loads_strict_json(json_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                return None, None, f"Checkpoint metadata {json_path} must be a JSON object."
            found_shas: Dict[str, str] = {}
            for sha_key in ("sha256", "checkpoint_sha256"):
                if sha_key in meta and meta[sha_key] is not None:
                    raw_sha = meta[sha_key]
                    if not isinstance(raw_sha, str) or not SHA256_RE.fullmatch(raw_sha):
                        return (
                            None,
                            None,
                            f"Invalid {sha_key} in checkpoint metadata {json_path}: {raw_sha!r}",
                        )
                    found_shas[sha_key] = raw_sha
            if (
                "sha256" in found_shas
                and "checkpoint_sha256" in found_shas
                and found_shas["sha256"] != found_shas["checkpoint_sha256"]
            ):
                return (
                    None,
                    None,
                    f"Conflicting sha256 ({found_shas['sha256']}) and checkpoint_sha256 "
                    f"({found_shas['checkpoint_sha256']}) in checkpoint metadata {json_path}.",
                )
            if found_shas:
                json_hash = found_shas.get("sha256") or found_shas.get("checkpoint_sha256")
        except Exception as exc:
            return None, None, f"Failed to parse checkpoint metadata {json_path}: {exc}"

    if pt_hash is not None and json_hash is not None:
        if pt_hash != json_hash:
            return (
                None,
                None,
                f"Checkpoint metadata mismatch: {pt_path} sha256={pt_hash} != {json_path} sha256={json_hash}",
            )
        return pt_hash, f"{pt_path} & {json_path}", None
    if pt_hash is not None:
        return pt_hash, str(pt_path), None
    if json_hash is not None:
        return json_hash, str(json_path), None

    # Fallback: check if run_log.json points to an existing checkpoint .pt file
    log_entry = _lookup_run_log(run_log_map, dataset, arch, split, gnn_seed)
    if log_entry and "checkpoint_path" in log_entry and log_entry["checkpoint_path"] is not None:
        cp = log_entry["checkpoint_path"]
        if not isinstance(cp, str) or not cp.strip() or cp != cp.strip() or "\x00" in cp:
            return None, None, f"Invalid checkpoint_path in run_log.json: {cp!r}"
        try:
            cand_path = pathlib.Path(cp)
            if not cand_path.is_absolute():
                cand_path = ROOT / cand_path
            if cand_path.is_file():
                return sha256_file(cand_path), f"{cand_path} (via run_log.json)", None
            if cand_path.is_dir():
                return None, None, f"Invalid checkpoint_path in run_log.json (expected file, got directory): {cand_path}"
        except (OSError, ValueError) as exc:
            return None, None, f"Invalid checkpoint_path in run_log.json ({cp!r}): {exc}"

    return (
        None,
        None,
        f"Cannot locate checkpoint .pt or .json for ({dataset}, {arch}, g={gnn_seed}) under {checkpoints_dir}",
    )


def _parse_nonneg_int_node_list(
    raw_list: Any,
    source_desc: str,
) -> Tuple[Optional[List[int]], Optional[str]]:
    if not isinstance(raw_list, list) or len(raw_list) == 0:
        return None, f"{source_desc} must be a non-empty list of non-negative integers."
    nodes: List[int] = []
    seen: Set[int] = set()
    for item in raw_list:
        if not _is_valid_nonneg_int_scalar(item):
            return (
                None,
                f"{source_desc} contains invalid node_id={item!r} (expected non-negative integer, not bool).",
            )
        val = int(item)
        if val > MAX_SAFE_INT:
            return (
                None,
                f"{source_desc} contains out-of-range node_id={val} (must be <= {MAX_SAFE_INT}).",
            )
        if val in seen:
            return (
                None,
                f"{source_desc} contains duplicate node_id={val}.",
            )
        seen.add(val)
        nodes.append(val)
    return nodes, None


def _load_nodes_from_json_file(
    path: pathlib.Path,
    *,
    missing_prefix: str = "Expected nodes file not found",
) -> Tuple[Optional[List[int]], Optional[str]]:
    try:
        path_present = _path_entry_exists(path)
        path_is_file = path.is_file()
    except (OSError, ValueError) as exc:
        return None, f"Invalid expected node list path {path}: {exc}"
    if path_present and not path_is_file:
        return None, f"Expected node list path is not a regular file: {path}"
    if not path_is_file:
        return None, f"{missing_prefix}: {path}"
    try:
        raw = _loads_strict_json(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return None, f"Failed to parse expected node list file {path}: {exc}"
    if isinstance(raw, dict):
        if "nodes" not in raw:
            return None, f"Expected node list file {path} is missing 'nodes' key."
        raw_list = raw["nodes"]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        return None, f"Expected node list file {path} must contain a JSON list or object with 'nodes'."
    return _parse_nonneg_int_node_list(raw_list, f"Nodes list in {path}")


def _load_expected_nodes(
    dataset: str,
    split: str,
    data_dir: pathlib.Path,
    expected_nodes_file: Optional[pathlib.Path],
    max_nodes: Optional[int],
    run_log_entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[List[int]], Optional[str]]:
    if max_nodes is not None and (not _is_valid_nonneg_int_scalar(max_nodes) or max_nodes <= 0):
        return None, f"max_nodes must be a positive integer, got {max_nodes!r}."

    if expected_nodes_file is not None:
        nodes, err = _load_nodes_from_json_file(
            expected_nodes_file, missing_prefix="Expected nodes file not found"
        )
        if err is not None or nodes is None:
            return None, err
        if max_nodes is not None:
            nodes = nodes[:max_nodes]
        return nodes, None

    if split == "test":
        test_file = data_dir / dataset / "test_nodes.json"
        nodes, err = _load_nodes_from_json_file(
            test_file, missing_prefix="Missing expected node list file"
        )
        if err is not None or nodes is None:
            return None, err
        if max_nodes is not None:
            nodes = nodes[:max_nodes]
        return nodes, None

    # split == "val"
    val_file = data_dir / dataset / "val_nodes.json"
    try:
        val_present = _path_entry_exists(val_file)
        val_is_file = val_file.is_file()
    except (OSError, ValueError) as exc:
        return None, f"Invalid expected node list path {val_file}: {exc}"
    if val_present and not val_is_file:
        return None, f"Expected node list path is not a regular file: {val_file}"
    if val_is_file:
        all_val, err = _load_nodes_from_json_file(
            val_file, missing_prefix="Missing expected node list file"
        )
        if err is not None or all_val is None:
            return None, err
        if max_nodes is not None:
            return all_val[:max_nodes], None
        if isinstance(run_log_entry, dict) and "node_ids" in run_log_entry:
            log_nodes, log_err = _parse_nonneg_int_node_list(
                run_log_entry["node_ids"], "run_log.json 'node_ids'"
            )
            if log_err is not None or log_nodes is None:
                return None, log_err
            extra = set(log_nodes) - set(all_val)
            if extra:
                return None, f"run_log.json contains val nodes not in {val_file}: {sorted(extra)}"
            return log_nodes, None
        return all_val[:20], None

    if isinstance(run_log_entry, dict) and "node_ids" in run_log_entry:
        log_nodes, log_err = _parse_nonneg_int_node_list(
            run_log_entry["node_ids"], "run_log.json 'node_ids'"
        )
        if log_err is not None or log_nodes is None:
            return None, log_err
        return log_nodes, None
    return None, f"Missing expected node list file: {val_file}"


def validate_scores_dir(
    scores_path: pathlib.Path,
    checkpoints_dir: Optional[pathlib.Path] = None,
    checkpoint_path: Optional[pathlib.Path] = None,
    data_dir: Optional[pathlib.Path] = None,
    expected_nodes_file: Optional[pathlib.Path] = None,
    max_nodes: Optional[int] = None,
    require_run_log: bool = True,
    verify_graph: bool = True,
    allow_failed: bool = False,
    skip_checkpoint_hash: bool = False,
) -> Dict[str, Any]:
    """Validate score CSV files under scores_path and return structured report."""
    checkpoints_dir = checkpoints_dir or (ROOT / "results" / "checkpoints")
    data_dir = data_dir or (ROOT / "data")

    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def add_error(file_ref: str, check: str, message: str, lines: Optional[Sequence[int]] = None) -> None:
        entry: Dict[str, Any] = {"file": file_ref, "check": check, "message": message}
        if lines:
            entry["lines"] = list(lines[:20])
            if len(lines) > 20:
                entry["total_bad_lines"] = len(lines)
        errors.append(entry)

    scores_is_dir = False
    try:
        scores_is_dir = scores_path.is_dir()
    except (OSError, ValueError):
        scores_is_dir = False

    scores_root_resolved: Optional[pathlib.Path] = None
    raw_log_paths: List[pathlib.Path] = []
    dir_and_link_paths: List[pathlib.Path] = []
    scan_errors: List[Tuple[str, str]] = []
    if scores_is_dir:
        try:
            scores_root_resolved = scores_path.resolve()
        except (OSError, ValueError) as exc:
            add_error(
                str(scores_path),
                "path_containment",
                f"Failed to resolve scores directory {scores_path}: {exc}",
            )
        csv_files, raw_log_paths, dir_and_link_paths, scan_errors = _scan_scores_tree(scores_path)
    else:
        csv_files = _discover_csv_files(scores_path)

    checked_containment: Set[pathlib.Path] = set()

    def _verify_path_containment(p: pathlib.Path, kind: str) -> None:
        if scores_root_resolved is None or p in checked_containment:
            return
        checked_containment.add(p)
        try:
            if not p.exists():
                add_error(
                    str(p),
                    "path_containment",
                    f"Discovered broken symlink or dangling junction ({kind}) under {scores_path}: {p}",
                )
                return
            rel_raw = p.relative_to(scores_path).parts
            p_resolved = p.resolve()
            rel_resolved = p_resolved.relative_to(scores_root_resolved).parts
        except (OSError, ValueError) as exc:
            add_error(
                str(p),
                "path_containment",
                f"Discovered {kind} escapes scores directory {scores_path}: {exc}",
            )
            return
        if rel_resolved != rel_raw:
            add_error(
                str(p),
                "path_containment",
                f"Discovered {kind} canonical relative path {rel_resolved} does not match "
                f"{rel_raw} under {scores_path} (possible symlink or directory junction).",
            )

    if scores_is_dir:
        for err_path, err_msg in scan_errors:
            add_error(err_path, "path_containment", err_msg)

    if not csv_files:
        if scores_is_dir:
            for dp in dir_and_link_paths:
                _verify_path_containment(dp, "directory/link")
        add_error(str(scores_path), "discovery", "No score CSV files found.")
        return {
            "status": "FAIL",
            "scores_path": str(scores_path),
            "n_files": 0,
            "n_rows": 0,
            "n_run_logs": 0,
            "datasets": [],
            "verified_hashes": {},
            "graph_verified_datasets": [],
            "saliency_determinism": {},
            "n_errors": len(errors),
            "errors": errors,
            "warnings": warnings,
        }

    # Discover and parse run_log.json files
    run_log_map: Dict[Tuple[str, str, Optional[str], int], Dict[str, Any]] = {}
    run_log_hashes_by_key: Dict[Tuple[str, str, str, int], List[Tuple[str, str]]] = {}
    dir_run_logs: Dict[pathlib.Path, Tuple[pathlib.Path, Dict[str, Any], Tuple[str, str, str, int, str]]] = {}
    all_logged_runs_by_dir: Dict[pathlib.Path, Tuple[pathlib.Path, Dict[str, Dict[str, Any]]]] = {}
    root_logged_runs: Dict[str, Tuple[pathlib.Path, Dict[str, Any], Tuple[str, str, str, int, str]]] = {}
    agg_logged_runs_by_dir: Dict[
        pathlib.Path,
        Tuple[pathlib.Path, Dict[str, Tuple[Dict[str, Any], Tuple[str, str, str, int, str]]]],
    ] = {}
    # (lf, content_dict, parsed_meta, normalized_runs_dict_or_None, is_flat_single_run)
    run_log_sources: List[
        Tuple[pathlib.Path, Dict[str, Any], Tuple[str, str, str, int, str], Optional[Dict[str, Dict[str, Any]]], bool]
    ] = []
    expected_csv_sha: Dict[Tuple[pathlib.Path, str], List[Tuple[str, str]]] = {}
    expected_csv_sha_global: Dict[str, List[Tuple[str, str]]] = {}
    run_log_files: List[pathlib.Path] = []
    invalid_log_dirs: Set[pathlib.Path] = set()
    aggregate_log_dirs: Set[pathlib.Path] = set()
    root_has_aggregate_log = False
    if scores_is_dir:
        for csv_p in csv_files:
            _verify_path_containment(csv_p, "CSV")

        for lf in raw_log_paths:
            try:
                lf_is_file = lf.is_file()
            except (OSError, ValueError) as exc:
                add_error(str(lf), "run_log", f"Failed to inspect run_log.json path: {exc}")
                invalid_log_dirs.add(lf.parent)
                continue
            if not lf_is_file:
                add_error(str(lf), "run_log", f"run_log.json path is not a regular file: {lf}")
                invalid_log_dirs.add(lf.parent)
                continue
            _verify_path_containment(lf, "run_log.json")
            run_log_files.append(lf)

        for dp in dir_and_link_paths:
            _verify_path_containment(dp, "directory/link")
        if require_run_log and not run_log_files and not invalid_log_dirs:
            add_error(str(scores_path), "run_log", "Missing run_log.json in scores directory.")
        for lf in run_log_files:
            try:
                content = _loads_strict_json(lf.read_text(encoding="utf-8"))
                if not isinstance(content, dict) or not content:
                    add_error(str(lf), "run_log", "run_log.json must be a non-empty JSON object.")
                    invalid_log_dirs.add(lf.parent)
                    continue
                if "runs" in content and "dataset" not in content and "gnn_seed" not in content:
                    if "runs_by_id" in content:
                        add_error(
                            str(lf),
                            "run_log",
                            "run_log.json cannot specify both 'runs' and 'runs_by_id'.",
                        )
                        invalid_log_dirs.add(lf.parent)
                        continue
                    runs_list = content.get("runs")
                    if not isinstance(runs_list, list) or len(runs_list) == 0:
                        add_error(str(lf), "run_log", "run_log.json 'runs' must be a non-empty list.")
                        invalid_log_dirs.add(lf.parent)
                        continue
                    parsed_any_agg = False
                    agg_runs_for_lf: Dict[str, Tuple[Dict[str, Any], Tuple[str, str, str, int, str]]] = {}
                    for item in runs_list:
                        parsed = _validate_run_log_entry(
                            item, str(lf), add_error, require_runs_container=False
                        )
                        if parsed is None:
                            continue
                        ds_l, arch_l, sp_l, g_l, sha_l = parsed
                        run_log_map[(ds_l, arch_l, sp_l, g_l)] = item
                        run_log_hashes_by_key.setdefault((ds_l, arch_l, sp_l, g_l), []).append((str(lf), sha_l))
                        if "runs_by_id" in item or "runs" in item:
                            norm_runs = _normalize_runs_dict(item, str(lf), add_error)
                            if norm_runs is None:
                                add_error(
                                    str(lf),
                                    "run_log",
                                    "Aggregate run_log.json entry failed to normalize non-empty 'runs_by_id'/'runs'.",
                                )
                                continue
                            parsed_any_agg = True
                            run_log_sources.append((lf, item, parsed, norm_runs, False))
                            for rid, rinfo in norm_runs.items():
                                if rid in root_logged_runs or rid in agg_runs_for_lf:
                                    add_error(str(lf), "run_log", f"Duplicate run_id '{rid}' in aggregate run_log.json.")
                                root_logged_runs[rid] = (lf, rinfo, parsed)
                                agg_runs_for_lf[rid] = (rinfo, parsed)
                                csv_sha = _extract_valid_csv_sha256(rinfo, rid, str(lf), add_error)
                                if csv_sha is not None:
                                    expected_csv_sha_global.setdefault(rid, []).append((csv_sha, str(lf)))
                        elif isinstance(item.get("run_id"), str) and item["run_id"].strip() and item["run_id"] == item["run_id"].strip():
                            rid = str(item["run_id"])
                            norm_single = {rid: item}
                            parsed_any_agg = True
                            run_log_sources.append((lf, item, parsed, norm_single, True))
                            if rid in root_logged_runs or rid in agg_runs_for_lf:
                                add_error(str(lf), "run_log", f"Duplicate run_id '{rid}' in aggregate run_log.json.")
                            root_logged_runs[rid] = (lf, item, parsed)
                            agg_runs_for_lf[rid] = (item, parsed)
                            csv_sha = _extract_valid_csv_sha256(item, rid, str(lf), add_error)
                            if csv_sha is not None:
                                expected_csv_sha_global.setdefault(rid, []).append((csv_sha, str(lf)))
                        else:
                            add_error(
                                str(lf),
                                "run_log",
                                "Aggregate run_log.json entry must contain 'runs_by_id', 'runs', or valid 'run_id'.",
                            )
                    if parsed_any_agg:
                        aggregate_log_dirs.add(lf.parent)
                        agg_logged_runs_by_dir[lf.parent] = (lf, agg_runs_for_lf)
                        if lf.parent == scores_path:
                            root_has_aggregate_log = True
                    else:
                        invalid_log_dirs.add(lf.parent)
                else:
                    parsed = _validate_run_log_entry(
                        content, str(lf), add_error, require_runs_container=True
                    )
                    if parsed is None:
                        invalid_log_dirs.add(lf.parent)
                        continue
                    ds_l, arch_l, sp_l, g_l, sha_l = parsed
                    run_log_map[(ds_l, arch_l, sp_l, g_l)] = content
                    run_log_hashes_by_key.setdefault((ds_l, arch_l, sp_l, g_l), []).append((str(lf), sha_l))
                    dir_run_logs[lf.parent] = (lf, content, parsed)
                    norm_runs = _normalize_runs_dict(content, str(lf), add_error)
                    if norm_runs is None:
                        add_error(
                            str(lf),
                            "run_log",
                            "run_log.json failed to normalize non-empty 'runs_by_id'/'runs'.",
                        )
                        invalid_log_dirs.add(lf.parent)
                        continue
                    run_log_sources.append((lf, content, parsed, norm_runs, False))
                    all_logged_runs_by_dir[lf.parent] = (lf, norm_runs)
                    for rid, rinfo in norm_runs.items():
                        csv_sha = _extract_valid_csv_sha256(rinfo, rid, str(lf), add_error)
                        if csv_sha is not None:
                            expected_csv_sha.setdefault((lf.parent, rid), []).append((csv_sha, str(lf)))
            except Exception as exc:
                add_error(str(lf), "run_log", f"Invalid JSON in run_log.json: {exc}")
                invalid_log_dirs.add(lf.parent)

    # Track cross-file invariants
    # group_target_class[(dataset, arch, split, gnn_seed, node_id)] = {"file": str, "target_class": int}
    group_target_class: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    # group_ref[(dataset, arch, split, gnn_seed, node_id)] = {
    #   "file": str, "status": str, "edges": List[(eid, u, v)]
    # }
    group_ref: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    group_hashes: Dict[Tuple[str, str, str, int], Dict[str, str]] = {}
    group_splits: Dict[Tuple[str, str], Set[str]] = {}
    all_splits_seen: Set[str] = set()
    seen_run_ids: Dict[str, str] = {}
    csv_run_id_by_path: Dict[pathlib.Path, str] = {}
    csv_method_by_run_id: Dict[str, str] = {}
    dir_seen_run_ids: Dict[pathlib.Path, Dict[str, pathlib.Path]] = {}
    file_node_sets: List[Dict[str, Any]] = []
    # For Saliency determinism check: saliency_scores[(dataset, arch, split, gnn_seed)][explainer_seed] = Dict[node_id, np.ndarray]
    saliency_scores: Dict[Tuple[str, str, str, int], Dict[int, Dict[int, np.ndarray]]] = {}
    # For optional graph verification
    datasets_seen: Set[str] = set()
    total_rows = 0

    for csv_path in csv_files:
        fref = str(csv_path)
        try:
            df = pd.read_csv(
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
            add_error(fref, "read_csv", f"Failed to read CSV: {exc}")
            continue

        total_rows += len(df)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            add_error(fref, "columns", f"Missing required columns: {missing_cols}")
            continue

        if df.empty:
            add_error(fref, "empty_file", "CSV file has 0 data rows.")
            continue

        # Validate status column
        bad_status_mask = ~df["status"].isin(ALLOWED_STATUSES)
        if bad_status_mask.any():
            bad_lines = [int(i) + 2 for i in df.index[bad_status_mask]]
            add_error(
                fref,
                "status",
                f"Invalid status values: {sorted(df.loc[bad_status_mask, 'status'].astype(str).unique())}",
                lines=bad_lines,
            )
            continue

        failed_mask = df["status"] == "failed"
        if failed_mask.any() and not allow_failed:
            bad_lines = [int(i) + 2 for i in df.index[failed_mask]]
            add_error(
                fref,
                "failed_rows",
                f"Found {int(failed_mask.sum())} rows with status='failed'.",
                lines=bad_lines,
            )

        # Single-run metadata columns must have no NaN and exactly 1 unique value per file
        meta_cols = [
            "run_id",
            "dataset",
            "split",
            "arch",
            "gnn_seed",
            "checkpoint_sha256",
            "method",
            "explainer_seed",
            "seed",
        ]
        meta_missing = False
        for col in meta_cols:
            na_mask = df[col].isna()
            if na_mask.any():
                add_error(
                    fref,
                    "missing_values",
                    f"Column '{col}' has NaN values.",
                    lines=[int(i) + 2 for i in df.index[na_mask]],
                )
                meta_missing = True
        if meta_missing:
            continue

        # Check integer metadata columns
        int_meta_ok = True
        for col in ("gnn_seed", "explainer_seed", "seed", "node_id", "target_class"):
            ok, bad_lines = _is_nonneg_int_series(df[col])
            if not ok:
                add_error(fref, "integer_field", f"Column '{col}' contains non-integer or negative values.", lines=bad_lines)
                int_meta_ok = False
        if not int_meta_ok:
            continue

        for col in ("gnn_seed", "explainer_seed", "seed", "node_id", "target_class"):
            df[col] = pd.to_numeric(df[col]).astype(np.int64)

        # Check single unique value for file-level identifiers
        multi_val = False
        for col in ("run_id", "dataset", "split", "arch", "gnn_seed", "checkpoint_sha256", "method", "explainer_seed"):
            if df[col].nunique(dropna=False) != 1:
                add_error(
                    fref,
                    "single_run_per_file",
                    f"Column '{col}' has multiple values in one CSV: {df[col].unique().tolist()[:5]}",
                )
                multi_val = True
        if multi_val:
            continue

        # Check seed == explainer_seed
        mismatch_seed = df["seed"] != df["explainer_seed"]
        if mismatch_seed.any():
            add_error(
                fref,
                "seed_consistency",
                "Column 'seed' does not equal 'explainer_seed'.",
                lines=[int(i) + 2 for i in df.index[mismatch_seed]],
            )

        run_id = str(df["run_id"].iloc[0])
        dataset = str(df["dataset"].iloc[0])
        split = str(df["split"].iloc[0])
        arch = str(df["arch"].iloc[0])
        gnn_seed = int(df["gnn_seed"].iloc[0])
        ckpt_sha = str(df["checkpoint_sha256"].iloc[0])
        method = str(df["method"].iloc[0])
        expl_seed = int(df["explainer_seed"].iloc[0])

        for col_name, raw_val in (
            ("run_id", run_id),
            ("dataset", dataset),
            ("split", split),
            ("arch", arch),
            ("checkpoint_sha256", ckpt_sha),
            ("method", method),
        ):
            if raw_val != raw_val.strip():
                add_error(
                    fref,
                    "identifier_whitespace",
                    f"Column '{col_name}' contains leading/trailing whitespace: {raw_val!r}.",
                )

        datasets_seen.add(dataset)

        if not DATASET_RE.fullmatch(dataset):
            add_error(fref, "dataset", f"Invalid dataset identifier '{dataset}'.")
        if split not in ALLOWED_SPLITS:
            add_error(fref, "split", f"Invalid split '{split}'. Expected one of {sorted(ALLOWED_SPLITS)}.")
        if arch not in ALLOWED_ARCHS:
            add_error(fref, "arch", f"Unsupported arch '{arch}'. Expected one of {sorted(ALLOWED_ARCHS)}.")
        if method not in ALLOWED_METHODS:
            add_error(fref, "method", f"Unsupported method '{method}'. Expected one of {sorted(ALLOWED_METHODS)}.")
        if not SHA256_RE.fullmatch(ckpt_sha):
            add_error(fref, "checkpoint_sha256", f"Invalid SHA-256 format: '{ckpt_sha}'.")

        expected_run_id = f"{dataset}_{arch}_g{gnn_seed}_{method}_e{expl_seed}_{split}"
        if run_id != expected_run_id:
            add_error(
                fref,
                "run_id_format",
                f"run_id='{run_id}' does not match schema format '{expected_run_id}'.",
            )

        csv_run_id_by_path[csv_path] = run_id
        csv_method_by_run_id[run_id] = method
        dir_seen_run_ids.setdefault(csv_path.parent, {})[run_id] = csv_path
        group_splits.setdefault((dataset, arch), set()).add(split)
        all_splits_seen.add(split)

        # Match CSV metadata against its directory's run_log.json
        if csv_path.parent in dir_run_logs:
            lf_dir, _log_content, (log_ds, log_arch, log_sp, log_g, log_sha) = dir_run_logs[csv_path.parent]
            if (dataset, arch, split, gnn_seed) != (log_ds, log_arch, log_sp, log_g):
                add_error(
                    fref,
                    "run_log_metadata_mismatch",
                    f"CSV metadata (dataset={dataset}, arch={arch}, split={split}, gnn_seed={gnn_seed}) "
                    f"does not match {lf_dir} (dataset={log_ds}, arch={log_arch}, split={log_sp}, gnn_seed={log_g}).",
                )
        elif (
            scores_path.is_dir()
            and require_run_log
            and (
                root_has_aggregate_log
                or any(agg_dir == csv_path.parent or agg_dir in csv_path.parents for agg_dir in aggregate_log_dirs)
            )
            and not (csv_path.parent / "run_log.json").is_file()
        ):
            if _lookup_run_log(run_log_map, dataset, arch, split, gnn_seed) is None:
                add_error(
                    fref,
                    "run_log_metadata_mismatch",
                    f"No matching run_log entry found for (dataset={dataset}, arch={arch}, split={split}, gnn_seed={gnn_seed}).",
                )

        # Also cross-check CSV metadata & checkpoint_sha256 against Shape B root_logged_runs[run_id]
        if run_id in root_logged_runs:
            lf_r, _rinfo_r, (ds_r, arch_r, sp_r, g_r, sha_r) = root_logged_runs[run_id]
            if (dataset, arch, split, gnn_seed) != (ds_r, arch_r, sp_r, g_r):
                add_error(
                    fref,
                    "run_log_metadata_mismatch",
                    f"CSV metadata (dataset={dataset}, arch={arch}, split={split}, gnn_seed={gnn_seed}) "
                    f"does not match {lf_r} entry for '{run_id}' "
                    f"(dataset={ds_r}, arch={arch_r}, split={sp_r}, gnn_seed={g_r}).",
                )
            if sha_r.strip() != ckpt_sha:
                add_error(
                    fref,
                    "run_log_hash_mismatch",
                    f"{lf_r} entry for '{run_id}' checkpoint_sha256={sha_r} != CSV sha256={ckpt_sha}.",
                )

        sha_entries = list(
            dict.fromkeys(
                expected_csv_sha.get((csv_path.parent, run_id), [])
                + expected_csv_sha_global.get(run_id, [])
            )
        )
        if sha_entries:
            actual_csv_hash = sha256_file(csv_path)
            for exp_csv_hash, log_src in sha_entries:
                if actual_csv_hash != exp_csv_hash:
                    add_error(
                        fref,
                        "csv_sha256_mismatch",
                        f"CSV SHA-256={actual_csv_hash} does not match {log_src} ({exp_csv_hash}). "
                        f"Possible manual edit or file corruption.",
                    )

        if run_id in seen_run_ids:
            add_error(
                fref,
                "duplicate_run_id",
                f"Duplicate run_id '{run_id}' already seen in {seen_run_ids[run_id]}.",
            )
        else:
            seen_run_ids[run_id] = fref

        # Validate runtime_sec finite & >= 0 (rejecting booleans)
        ok_rt, bad_rt_lines = _is_finite_float_series(df["runtime_sec"], non_negative=True)
        if not ok_rt:
            add_error(
                fref,
                "runtime_sec",
                "Column 'runtime_sec' contains missing, boolean, non-finite, or negative values.",
                lines=bad_rt_lines,
            )

        # Record checkpoint hash per (dataset, arch, split, gnn_seed)
        g_key = (dataset, arch, split, gnn_seed)
        group_hashes.setdefault(g_key, {})[fref] = ckpt_sha

        # Validate success rows vs empty_candidates / failed placeholder rows
        success_df = df[df["status"] == "success"]
        empty_df = df[df["status"] == "empty_candidates"]
        failed_df = df[df["status"] == "failed"]

        if not success_df.empty:
            for col in ("edge_id", "u", "v"):
                ok, bad_lines = _is_nonneg_int_series(success_df[col])
                if not ok:
                    add_error(
                        fref,
                        "edge_integer_field",
                        f"Success rows have invalid integer in '{col}'.",
                        lines=bad_lines,
                    )
            # Check u < v
            u_num = _safe_to_numeric(
                success_df["u"].astype(object).where(~_is_bool_like_mask(success_df["u"]), np.nan)
            )
            v_num = _safe_to_numeric(
                success_df["v"].astype(object).where(~_is_bool_like_mask(success_df["v"]), np.nan)
            )
            bad_uv = u_num.isna() | v_num.isna() | (u_num >= v_num)
            if bad_uv.any():
                add_error(
                    fref,
                    "canonical_undirected_edge",
                    "Found rows violating canonical u < v constraint.",
                    lines=[int(i) + 2 for i in success_df.index[bad_uv]],
                )

            # Check importance_score finite (rejecting booleans including bool+NaN mixed object columns)
            ok_sc, bad_sc_lines = _is_finite_float_series(
                success_df["importance_score"], non_negative=False
            )
            if not ok_sc:
                add_error(
                    fref,
                    "finite_importance_score",
                    "Found non-finite, boolean, or missing importance_score in success rows.",
                    lines=bad_sc_lines,
                )

            # Check duplicates within file
            dup_eid = success_df.duplicated(subset=["node_id", "edge_id"], keep=False)
            if dup_eid.any():
                add_error(
                    fref,
                    "duplicate_rows",
                    "Found duplicate (node_id, edge_id) rows.",
                    lines=[int(i) + 2 for i in success_df.index[dup_eid]],
                )
            dup_uv = success_df.duplicated(subset=["node_id", "u", "v"], keep=False)
            if dup_uv.any():
                add_error(
                    fref,
                    "duplicate_rows",
                    "Found duplicate (node_id, u, v) rows.",
                    lines=[int(i) + 2 for i in success_df.index[dup_uv]],
                )

        if not empty_df.empty:
            _validate_placeholder_rows(empty_df, "empty_candidates", fref, add_error)
            overlap_nodes = set(empty_df["node_id"]) & set(success_df["node_id"])
            if overlap_nodes:
                add_error(
                    fref,
                    "mixed_node_status",
                    f"Nodes have both 'success' and 'empty_candidates' rows: {sorted(overlap_nodes)}",
                )

        if not failed_df.empty:
            _validate_placeholder_rows(failed_df, "failed", fref, add_error)
            overlap_failed = set(failed_df["node_id"]) & (set(success_df["node_id"]) | set(empty_df["node_id"]))
            if overlap_failed:
                add_error(
                    fref,
                    "mixed_node_status",
                    f"Nodes mix 'failed' with other statuses: {sorted(overlap_failed)}",
                )

        # Per-node consistency checks and cross-method edge alignment
        nodes_in_file = sorted(df["node_id"].unique().tolist())
        file_node_sets.append(
            {
                "file": fref,
                "dataset": dataset,
                "split": split,
                "arch": arch,
                "gnn_seed": gnn_seed,
                "method": method,
                "explainer_seed": expl_seed,
                "nodes": nodes_in_file,
            }
        )

        # Skip per-node edge extraction if critical numeric checks failed on this file
        file_has_edge_err = any(
            e["file"] == fref
            and e["check"]
            in {
                "edge_integer_field",
                "canonical_undirected_edge",
                "finite_importance_score",
                "duplicate_rows",
                "placeholder_row_invalid",
            }
            for e in errors
        )
        if file_has_edge_err:
            continue

        for node_id, part in df.groupby("node_id", sort=True):
            nid = int(node_id)
            if part["target_class"].nunique() != 1:
                add_error(
                    fref,
                    "target_class_consistency",
                    f"Node {nid} has multiple target_class values within file.",
                    lines=[int(i) + 2 for i in part.index],
                )
                continue
            if part["status"].nunique() != 1:
                add_error(
                    fref,
                    "status_consistency",
                    f"Node {nid} has multiple status values within file.",
                    lines=[int(i) + 2 for i in part.index],
                )
                continue

            node_status = str(part["status"].iloc[0])
            target_class = int(part["target_class"].iloc[0])
            ref_key = (dataset, arch, split, gnn_seed, nid)

            # Always cross-check target_class independent of node_status and file alphabetical order
            if ref_key not in group_target_class:
                group_target_class[ref_key] = {"file": fref, "target_class": target_class}
            elif target_class != group_target_class[ref_key]["target_class"]:
                add_error(
                    fref,
                    "cross_method_target_class",
                    f"Node {nid} target_class={target_class} disagrees with "
                    f"{group_target_class[ref_key]['file']} ({group_target_class[ref_key]['target_class']}).",
                    lines=[int(i) + 2 for i in part.index],
                )

            if node_status == "failed":
                if allow_failed:
                    warnings.append(f"Allowed failed node {nid} in {fref} under --allow-failed.")
                continue

            if node_status == "success":
                part_num = part.copy()
                part_num["_edge_id_int"] = pd.to_numeric(part_num["edge_id"], errors="raise").astype(np.int64)
                part_num["_u_int"] = pd.to_numeric(part_num["u"], errors="raise").astype(np.int64)
                part_num["_v_int"] = pd.to_numeric(part_num["v"], errors="raise").astype(np.int64)
                part_sorted = part_num.sort_values("_edge_id_int", kind="mergesort")
                edges_sig = list(
                    zip(
                        part_sorted["_edge_id_int"].tolist(),
                        part_sorted["_u_int"].tolist(),
                        part_sorted["_v_int"].tolist(),
                    )
                )
                if len({eid_i for eid_i, _, _ in edges_sig}) != len(edges_sig):
                    add_error(
                        fref,
                        "duplicate_rows",
                        f"Node {nid} contains duplicate edge_id values after integer normalization.",
                        lines=[int(i) + 2 for i in part_sorted.index],
                    )
                if len({(u_i, v_i) for _, u_i, v_i in edges_sig}) != len(edges_sig):
                    add_error(
                        fref,
                        "duplicate_rows",
                        f"Node {nid} contains duplicate (u, v) edges after integer normalization.",
                        lines=[int(i) + 2 for i in part_sorted.index],
                    )
                expected_eids = list(range(len(edges_sig)))
                if [e[0] for e in edges_sig] != expected_eids:
                    add_error(
                        fref,
                        "edge_id_contiguous",
                        f"Node {nid} edge_id is not contiguous 0..{len(edges_sig)-1}.",
                        lines=[int(i) + 2 for i in part_sorted.index],
                    )
                if method == "saliency":
                    saliency_scores.setdefault(g_key, {}).setdefault(expl_seed, {})[nid] = (
                        pd.to_numeric(part_sorted["importance_score"]).to_numpy(dtype=float)
                    )
            else:
                edges_sig = []

            if ref_key not in group_ref:
                group_ref[ref_key] = {
                    "file": fref,
                    "status": node_status,
                    "edges": edges_sig,
                }
            else:
                ref = group_ref[ref_key]
                if node_status != ref["status"]:
                    add_error(
                        fref,
                        "cross_method_status",
                        f"Node {nid} status='{node_status}' disagrees with {ref['file']} ('{ref['status']}').",
                        lines=[int(i) + 2 for i in part.index],
                    )
                elif edges_sig != ref["edges"]:
                    add_error(
                        fref,
                        "cross_method_edge_alignment",
                        f"Node {nid} candidate (edge_id, u, v) list (len={len(edges_sig)}) differs from {ref['file']} (len={len(ref['edges'])}).",
                        lines=[int(i) + 2 for i in part.index],
                    )

    # Lock single split per (dataset, arch) and across the checked directory
    mixed_group_found = False
    for (ds, arch), splits in sorted(group_splits.items()):
        if len(splits) > 1:
            mixed_group_found = True
            add_error(
                f"{ds}/{arch}",
                "mixed_split",
                f"Mixed split values detected for ({ds}, {arch}): {sorted(splits)}.",
            )
    if not mixed_group_found and len(all_splits_seen) > 1:
        add_error(
            str(scores_path),
            "mixed_split",
            f"Mixed split values detected across directory: {sorted(all_splits_seen)}.",
        )

    # Two-way reconciliation between run_log.json (runs_by_id / runs) and CSVs on disk
    if scores_path.is_dir():
        resolved_csv_by_resolved: Dict[pathlib.Path, pathlib.Path] = {}
        resolved_by_csv: Dict[pathlib.Path, pathlib.Path] = {}
        for p in csv_files:
            try:
                rp = p.resolve()
                resolved_csv_by_resolved[rp] = p
                resolved_by_csv[p] = rp
            except (OSError, ValueError):
                pass

        for log_dir, (lf, logged_runs) in sorted(all_logged_runs_by_dir.items()):
            try:
                log_dir_resolved = log_dir.resolve()
            except (OSError, ValueError) as exc:
                add_error(str(lf), "run_log", f"Failed to resolve log directory {log_dir}: {exc}")
                continue
            disk_csvs_in_dir = [p for p in csv_files if p.parent == log_dir]
            seen_in_dir = dir_seen_run_ids.get(log_dir, {})
            expected_resolved_by_rid: Dict[str, Optional[pathlib.Path]] = {}
            has_declared_csv_file: Set[str] = set()
            valid_declared_resolved_paths: Set[pathlib.Path] = set()

            for rid, rinfo in sorted(logged_runs.items()):
                rid_str = str(rid)
                if isinstance(rinfo, dict) and "csv_file" in rinfo and rinfo["csv_file"] is not None:
                    has_declared_csv_file.add(rid_str)
                    ok_cf, csv_file_name = _extract_valid_csv_file(rinfo, str(lf), rid_str, add_error)
                    if not ok_cf or csv_file_name is None:
                        expected_resolved_by_rid[rid_str] = None
                        continue
                    if not _is_flat_csv_filename(csv_file_name):
                        add_error(
                            str(lf),
                            "run_log",
                            f"Logged run_id '{rid_str}' has invalid non-flat or non-CSV csv_file={csv_file_name!r} "
                            f"(expected flat '*.csv' filename inside {log_dir}).",
                        )
                        expected_resolved_by_rid[rid_str] = None
                        continue
                    cand_csv = log_dir / csv_file_name
                    try:
                        cand_resolved = cand_csv.resolve()
                        same_parent = cand_resolved.parent == log_dir_resolved
                        is_existing_file = cand_csv.is_file()
                    except (OSError, ValueError) as exc:
                        add_error(
                            str(lf),
                            "run_log",
                            f"Logged run_id '{rid_str}' has invalid csv_file={csv_file_name!r}: {exc}",
                        )
                        expected_resolved_by_rid[rid_str] = None
                        continue
                    actual_csv_p = resolved_csv_by_resolved.get(cand_resolved)
                    if (
                        not same_parent
                        or not is_existing_file
                        or actual_csv_p is None
                        or actual_csv_p.name != csv_file_name
                        or cand_resolved.name != csv_file_name
                    ):
                        add_error(
                            str(lf),
                            "missing_logged_csv",
                            f"Logged run_id '{rid_str}' references missing CSV '{csv_file_name}' in {log_dir}.",
                        )
                        expected_resolved_by_rid[rid_str] = None
                    else:
                        expected_resolved_by_rid[rid_str] = cand_resolved
                        valid_declared_resolved_paths.add(cand_resolved)
                        if actual_csv_p in csv_run_id_by_path and csv_run_id_by_path[actual_csv_p] != rid_str:
                            add_error(
                                str(actual_csv_p),
                                "run_log_metadata_mismatch",
                                f"CSV '{csv_file_name}' has run_id='{csv_run_id_by_path[actual_csv_p]}' != logged '{rid_str}'.",
                            )
                elif rid_str not in seen_in_dir:
                    add_error(
                        str(lf),
                        "missing_logged_csv",
                        f"Logged run_id '{rid_str}' has no corresponding CSV file in {log_dir}.",
                    )

            for csv_p in disk_csvs_in_dir:
                csv_p_resolved = resolved_by_csv.get(csv_p)
                if csv_p in csv_run_id_by_path:
                    rid_obs = csv_run_id_by_path[csv_p]
                    if rid_obs not in logged_runs:
                        add_error(
                            str(csv_p),
                            "unlogged_csv",
                            f"CSV '{csv_p.name}' (run_id='{rid_obs}') is not declared in {lf}.",
                        )
                    elif rid_obs in has_declared_csv_file:
                        exp_res = expected_resolved_by_rid.get(rid_obs)
                        if exp_res is None or csv_p_resolved is None or csv_p_resolved != exp_res:
                            declared_raw = (
                                logged_runs[rid_obs].get("csv_file")
                                if isinstance(logged_runs[rid_obs], dict)
                                else None
                            )
                            add_error(
                                str(csv_p),
                                "unlogged_csv",
                                f"CSV '{csv_p.name}' does not match logged csv_file {declared_raw!r} for run_id='{rid_obs}' in {lf}.",
                            )
                elif valid_declared_resolved_paths and (
                    csv_p_resolved is None or csv_p_resolved not in valid_declared_resolved_paths
                ):
                    add_error(
                        str(csv_p),
                        "unlogged_csv",
                        f"CSV '{csv_p.name}' is not declared in {lf}.",
                    )

        # Two-way reconciliation for each aggregate run_log.json (Shape B), scoped strictly to lf_r.parent
        for agg_dir, (lf_r, agg_runs) in sorted(agg_logged_runs_by_dir.items()):
            try:
                agg_dir_resolved = agg_dir.resolve()
            except (OSError, ValueError) as exc:
                add_error(
                    str(lf_r),
                    "run_log",
                    f"Failed to resolve aggregate log directory {agg_dir}: {exc}",
                )
                continue
            disk_csvs_under_agg = [
                p for p in csv_files if p.parent == agg_dir or agg_dir in p.parents
            ]
            seen_in_agg = {
                csv_run_id_by_path[p]: p
                for p in disk_csvs_under_agg
                if p in csv_run_id_by_path
            }
            agg_expected_resolved_by_rid: Dict[str, Optional[pathlib.Path]] = {}
            agg_has_declared_csv_file: Set[str] = set()
            agg_valid_declared_resolved_paths: Set[pathlib.Path] = set()

            for rid_str, (rinfo, (ds_r, arch_r, _sp_r, g_r, _sha_r)) in sorted(agg_runs.items()):
                if isinstance(rinfo, dict) and "csv_file" in rinfo and rinfo["csv_file"] is not None:
                    agg_has_declared_csv_file.add(rid_str)
                    ok_cf, csv_file_name = _extract_valid_csv_file(rinfo, str(lf_r), rid_str, add_error)
                    if not ok_cf or csv_file_name is None:
                        agg_expected_resolved_by_rid[rid_str] = None
                        continue
                    if not _is_safe_relative_csv_path(csv_file_name):
                        add_error(
                            str(lf_r),
                            "run_log",
                            f"Logged run_id '{rid_str}' has invalid or unsafe csv_file={csv_file_name!r} "
                            f"(expected safe relative '*.csv' path under {agg_dir}).",
                        )
                        agg_expected_resolved_by_rid[rid_str] = None
                        continue
                    if _is_flat_csv_filename(csv_file_name):
                        candidate_specs = [
                            ((csv_file_name,), agg_dir / csv_file_name),
                            ((f"{arch_r}_s{g_r}", csv_file_name), agg_dir / f"{arch_r}_s{g_r}" / csv_file_name),
                            (
                                (ds_r, f"{arch_r}_s{g_r}", csv_file_name),
                                agg_dir / ds_r / f"{arch_r}_s{g_r}" / csv_file_name,
                            ),
                        ]
                    else:
                        expected_rel_parts = tuple(re.split(r"[/\\]", csv_file_name))
                        candidate_specs = [(expected_rel_parts, agg_dir / csv_file_name)]
                    found_csv: Optional[pathlib.Path] = None
                    found_resolved: Optional[pathlib.Path] = None
                    for expected_parts, cand_csv in candidate_specs:
                        try:
                            resolved_c = cand_csv.resolve()
                            rel_parts = resolved_c.relative_to(agg_dir_resolved).parts
                            is_existing_file = cand_csv.is_file()
                        except (OSError, ValueError):
                            continue
                        actual_csv_p = resolved_csv_by_resolved.get(resolved_c)
                        if (
                            is_existing_file
                            and actual_csv_p is not None
                            and (actual_csv_p.parent == agg_dir or agg_dir in actual_csv_p.parents)
                            and rel_parts == expected_parts
                            and actual_csv_p.name == expected_parts[-1]
                        ):
                            found_csv = actual_csv_p
                            found_resolved = resolved_c
                            break
                    if found_csv is None or found_resolved is None:
                        add_error(
                            str(lf_r),
                            "missing_logged_csv",
                            f"Logged run_id '{rid_str}' references missing CSV '{csv_file_name}' under {agg_dir}.",
                        )
                        agg_expected_resolved_by_rid[rid_str] = None
                    else:
                        agg_expected_resolved_by_rid[rid_str] = found_resolved
                        agg_valid_declared_resolved_paths.add(found_resolved)
                        if found_csv in csv_run_id_by_path and csv_run_id_by_path[found_csv] != rid_str:
                            add_error(
                                str(found_csv),
                                "run_log_metadata_mismatch",
                                f"CSV '{found_csv.name}' has run_id='{csv_run_id_by_path[found_csv]}' != logged '{rid_str}'.",
                            )
                elif rid_str not in seen_in_agg:
                    add_error(
                        str(lf_r),
                        "missing_logged_csv",
                        f"Logged run_id '{rid_str}' has no corresponding CSV file under {agg_dir}.",
                    )

            for csv_p in disk_csvs_under_agg:
                csv_p_resolved = resolved_by_csv.get(csv_p)
                if csv_p in csv_run_id_by_path:
                    rid_obs = csv_run_id_by_path[csv_p]
                    if rid_obs not in agg_runs:
                        add_error(
                            str(csv_p),
                            "unlogged_csv",
                            f"CSV '{csv_p.name}' (run_id='{rid_obs}') is not declared in aggregate {lf_r}.",
                        )
                    elif rid_obs in agg_has_declared_csv_file:
                        exp_res = agg_expected_resolved_by_rid.get(rid_obs)
                        if exp_res is None or csv_p_resolved is None or csv_p_resolved != exp_res:
                            rinfo_obs, _meta_r = agg_runs[rid_obs]
                            declared_raw = rinfo_obs.get("csv_file") if isinstance(rinfo_obs, dict) else None
                            add_error(
                                str(csv_p),
                                "unlogged_csv",
                                f"CSV '{csv_p.name}' does not match logged csv_file {declared_raw!r} for run_id='{rid_obs}' in {lf_r}.",
                            )
                elif agg_valid_declared_resolved_paths and (
                    csv_p_resolved is None or csv_p_resolved not in agg_valid_declared_resolved_paths
                ):
                    add_error(
                        str(csv_p),
                        "unlogged_csv",
                        f"CSV '{csv_p.name}' is not declared in aggregate {lf_r}.",
                    )

        # Ensure subdirectories without their own run_log.json are never hidden by a root Shape A run_log.json
        if require_run_log and (run_log_files or invalid_log_dirs):
            for csv_p in csv_files:
                if csv_p.parent in all_logged_runs_by_dir or csv_p.parent in invalid_log_dirs:
                    continue
                covered_by_agg = any(
                    agg_dir == csv_p.parent or agg_dir in csv_p.parents for agg_dir in aggregate_log_dirs
                )
                if not covered_by_agg:
                    add_error(
                        str(csv_p),
                        "unlogged_csv",
                        f"CSV '{csv_p.name}' in {csv_p.parent} has no directory run_log.json "
                        f"and is not covered by an aggregate run_log.json.",
                    )

    # Cross-seed & cross-explainer hyperparams consistency check per (dataset, arch, method)
    method_hp_records: Dict[Tuple[str, str, str], Dict[Optional[str], List[str]]] = {}
    for lf, content, (log_ds, log_arch, log_sp, log_g, _log_sha), norm_runs, is_flat_run in run_log_sources:
        top_hp = content.get("hyperparams") if isinstance(content.get("hyperparams"), dict) else None
        if not is_flat_run and top_hp is not None:
            for m_top, hp_top in sorted(top_hp.items(), key=lambda kv: str(kv[0])):
                if not isinstance(m_top, str) or m_top not in ALLOWED_METHODS:
                    add_error(
                        str(lf),
                        "run_log",
                        f"Top-level hyperparams contains unsupported method key {m_top!r}.",
                    )
                    continue
                ok_top, canon_top = _canonical_hp_or_error(
                    hp_top, str(lf), f"Top-level hyperparams['{m_top}']", add_error
                )
                if not ok_top:
                    continue
                method_hp_records.setdefault((log_ds, log_arch, m_top), {}).setdefault(canon_top, []).append(
                    f"top_level:{log_ds}_{log_arch}_g{log_g}_{m_top} ({lf.parent.name})"
                )
        if norm_runs is not None:
            for rid_str, rinfo in sorted(norm_runs.items()):
                if not isinstance(rinfo, dict):
                    continue
                auth_method = csv_method_by_run_id.get(rid_str) or _extract_method_from_run_id(rid_str)
                declared_method = rinfo.get("method")
                if "method" in rinfo and declared_method is not None:
                    if not isinstance(declared_method, str) or declared_method not in ALLOWED_METHODS:
                        add_error(
                            str(lf),
                            "run_log",
                            f"Run '{rid_str}' declares unsupported method {declared_method!r}.",
                        )
                    elif auth_method is not None and declared_method != auth_method:
                        add_error(
                            str(lf),
                            "run_log_metadata_mismatch",
                            f"Run '{rid_str}' declares method='{declared_method}' != expected '{auth_method}'.",
                        )
                m = auth_method or (
                    declared_method
                    if isinstance(declared_method, str) and declared_method in ALLOWED_METHODS
                    else None
                )
                if not m:
                    continue
                if is_flat_run:
                    hp_val = rinfo.get("hyperparams")
                elif "hyperparams" in rinfo:
                    hp_val = rinfo["hyperparams"]
                elif top_hp is not None and m in top_hp:
                    hp_val = top_hp[m]
                else:
                    hp_val = None
                ok_hp, canon = _canonical_hp_or_error(hp_val, str(lf), f"Run '{rid_str}'", add_error)
                if not ok_hp:
                    continue
                method_hp_records.setdefault((log_ds, log_arch, m), {}).setdefault(canon, []).append(
                    f"{rid_str} ({lf.parent.name})"
                )
        else:
            seen_methods = sorted(
                {
                    info["method"]
                    for info in file_node_sets
                    if (info["dataset"], info["arch"], info["split"], info["gnn_seed"])
                    == (log_ds, log_arch, log_sp, log_g)
                }
            )
            for m in seen_methods:
                hp_val = top_hp[m] if (top_hp is not None and m in top_hp) else None
                ok_hp, canon = _canonical_hp_or_error(
                    hp_val, str(lf), f"Top-level hyperparams['{m}']", add_error
                )
                if not ok_hp:
                    continue
                method_hp_records.setdefault((log_ds, log_arch, m), {}).setdefault(canon, []).append(
                    f"{log_ds}_{log_arch}_g{log_g}_{m} ({lf.name})"
                )

    for (ds, arch, m), canon_map in sorted(method_hp_records.items()):
        if len(canon_map) > 1:
            details = "; ".join(
                f"{('null' if k is None else k)} -> {v[:3]}"
                for k, v in sorted(canon_map.items(), key=lambda kv: (kv[0] is not None, kv[0] or ""))
            )
            add_error(
                f"{ds}/{arch}/{m}",
                "hyperparams_mismatch",
                f"Inconsistent hyperparams across runs for ({ds}, {arch}, {m}): {details}",
            )

    # Verify checkpoint hashes within each (dataset, arch, split, gnn_seed) and against checkpoint files
    verified_hashes: Dict[str, Dict[str, Any]] = {}
    for (dataset, arch, split, gnn_seed), file_hash_map in sorted(group_hashes.items()):
        base_label = f"{dataset}/{arch}_s{gnn_seed}"
        label = base_label if base_label not in verified_hashes else f"{base_label}/{split}"
        unique_hashes = sorted(set(file_hash_map.values()))
        if len(unique_hashes) > 1:
            add_error(
                label,
                "cross_file_checkpoint_hash",
                f"Multiple checkpoint_sha256 values across files: {unique_hashes}",
            )
            continue
        csv_hash = unique_hashes[0]

        # Cross-check checkpoint_sha256 in every matching run_log.json (exact split match)
        # Always runs even under --skip-checkpoint-hash so internal CSV vs run_log drift is caught.
        logged_hashes = list(dict.fromkeys(run_log_hashes_by_key.get((dataset, arch, split, gnn_seed), [])))
        for lf_src, log_sha in logged_hashes:
            if log_sha.strip() != csv_hash:
                add_error(
                    label,
                    "run_log_hash_mismatch",
                    f"{lf_src} checkpoint_sha256={log_sha} != CSV sha256={csv_hash}",
                )

        if skip_checkpoint_hash:
            verified_hashes[label] = {
                "csv_sha256": csv_hash,
                "verified": False,
                "source": "skipped",
            }
            continue

        expected_hash, source_desc, err_msg = _resolve_checkpoint_hash(
            dataset=dataset,
            arch=arch,
            gnn_seed=gnn_seed,
            checkpoints_dir=checkpoints_dir,
            explicit_ckpt_path=checkpoint_path,
            run_log_map=run_log_map,
            split=split,
        )
        if err_msg is not None:
            add_error(label, "checkpoint_hash_lookup", err_msg)
        elif expected_hash != csv_hash:
            add_error(
                label,
                "checkpoint_hash_mismatch",
                f"CSV checkpoint_sha256={csv_hash} != checkpoint {source_desc} sha256={expected_hash}",
            )
        else:
            verified_hashes[label] = {
                "csv_sha256": csv_hash,
                "verified": True,
                "source": source_desc,
            }

    # Verify expected node coverage for every file
    expected_nodes_cache: Dict[Tuple[str, str, str, int], Optional[List[int]]] = {}
    for info in file_node_sets:
        ds = info["dataset"]
        sp = info["split"]
        g = info["gnn_seed"]
        arch = info["arch"]
        cache_key = (ds, arch, sp, g)
        if cache_key not in expected_nodes_cache:
            log_entry = _lookup_run_log(run_log_map, ds, arch, sp, g)
            exp_nodes, err_msg = _load_expected_nodes(
                dataset=ds,
                split=sp,
                data_dir=data_dir,
                expected_nodes_file=expected_nodes_file,
                max_nodes=max_nodes,
                run_log_entry=log_entry,
            )
            if err_msg is not None:
                add_error(info["file"], "expected_nodes_lookup", err_msg)
            expected_nodes_cache[cache_key] = exp_nodes

        exp_nodes = expected_nodes_cache[cache_key]
        if exp_nodes is not None:
            exp_set = set(exp_nodes)
            obs_set = set(info["nodes"])
            missing_nodes = sorted(exp_set - obs_set)
            extra_nodes = sorted(obs_set - exp_set)
            if missing_nodes:
                add_error(
                    info["file"],
                    "missing_nodes",
                    f"Missing {len(missing_nodes)} expected nodes (first 10: {missing_nodes[:10]}).",
                )
            if extra_nodes:
                add_error(
                    info["file"],
                    "extra_nodes",
                    f"Found {len(extra_nodes)} unexpected nodes outside expected list (first 10: {extra_nodes[:10]}).",
                )

    # Optional graph candidate edge verification against data/{ds}/data.pt
    graph_verified_datasets: List[str] = []
    if verify_graph:
        try:
            import torch
            from sna.data.candidate_edges import extract_candidate_edges

            for ds in sorted(datasets_seen):
                data_pt = data_dir / ds / "data.pt"
                try:
                    data_pt_exists = _path_entry_exists(data_pt)
                    data_pt_is_file = data_pt.is_file()
                except (OSError, ValueError) as exc:
                    add_error(str(data_pt), "graph_verification", f"Invalid graph data path {data_pt}: {exc}")
                    continue
                if not data_pt_exists:
                    warnings.append(f"Skipping graph candidate verification for '{ds}': {data_pt} not found.")
                    continue
                if not data_pt_is_file:
                    add_error(str(data_pt), "graph_verification", f"Graph data path is not a regular file: {data_pt}")
                    continue
                data_obj = torch.load(data_pt, map_location="cpu", weights_only=False)
                edge_index = data_obj["edge_index"]
                num_nodes = int(data_obj["x"].size(0))
                expected_cand_by_node: Dict[int, Tuple[int, List[Tuple[int, int, int]]]] = {}
                for (r_ds, r_arch, r_sp, r_g, nid), ref in group_ref.items():
                    if r_ds != ds:
                        continue
                    if nid not in expected_cand_by_node:
                        cand = extract_candidate_edges(edge_index, nid, num_hops=2, num_nodes=num_nodes)
                        expected_sig = [
                            (eid, int(u), int(v)) for eid, (u, v) in zip(cand.edge_ids, cand.candidate_pairs)
                        ]
                        expected_cand_by_node[nid] = (cand.num_candidate_edges, expected_sig)
                    exp_count, expected_sig = expected_cand_by_node[nid]
                    if exp_count == 0:
                        if ref["status"] != "empty_candidates":
                            add_error(
                                ref["file"],
                                "graph_candidate_mismatch",
                                f"Node {nid} has 0 candidate edges in graph but status='{ref['status']}'.",
                            )
                    elif ref["edges"] != expected_sig:
                        add_error(
                            ref["file"],
                            "graph_candidate_mismatch",
                            f"Node {nid} CSV candidate edges (len={len(ref['edges'])}) do not match data.pt 2-hop candidates (len={len(expected_sig)}).",
                        )
                # Verify split mask anti-leakage when masks exist in data.pt
                train_mask = data_obj.get("train_mask")
                for info in file_node_sets:
                    if info["dataset"] != ds:
                        continue
                    sp = info["split"]
                    sp_mask = data_obj.get(f"{sp}_mask")
                    if sp_mask is not None and hasattr(sp_mask, "__len__") and len(sp_mask) == num_nodes:
                        sp_mask_bool = torch.as_tensor(sp_mask, dtype=torch.bool).cpu()
                        bad_split_nodes = [
                            nid for nid in info["nodes"] if nid >= num_nodes or not bool(sp_mask_bool[nid])
                        ]
                        if bad_split_nodes:
                            add_error(
                                info["file"],
                                "split_mask_leakage",
                                f"Nodes {bad_split_nodes[:10]} are not in data.pt '{sp}_mask'.",
                            )
                    if (
                        sp in {"val", "test"}
                        and train_mask is not None
                        and hasattr(train_mask, "__len__")
                        and len(train_mask) == num_nodes
                    ):
                        tr_mask_bool = torch.as_tensor(train_mask, dtype=torch.bool).cpu()
                        leaked_train = [
                            nid for nid in info["nodes"] if nid < num_nodes and bool(tr_mask_bool[nid])
                        ]
                        if leaked_train:
                            add_error(
                                info["file"],
                                "split_mask_leakage",
                                f"Nodes {leaked_train[:10]} overlap with data.pt 'train_mask'.",
                            )
                graph_verified_datasets.append(ds)
        except Exception as exc:
            add_error(str(data_dir), "graph_verification", f"Graph verification failed: {exc}")

    # Verify Saliency determinism if multiple explainer seeds exist for the same (dataset, arch, split, gnn_seed)
    saliency_determinism: Dict[str, Any] = {}
    for g_key, seed_map in sorted(saliency_scores.items()):
        seeds = sorted(seed_map.keys())
        if len(seeds) >= 2:
            base_seed = seeds[0]
            base_nodes = seed_map[base_seed]
            max_diff = 0.0
            for other_seed in seeds[1:]:
                other_nodes = seed_map[other_seed]
                for nid, arr_a in base_nodes.items():
                    if nid in other_nodes and len(arr_a) == len(other_nodes[nid]):
                        diff = float(np.max(np.abs(arr_a - other_nodes[nid]))) if len(arr_a) > 0 else 0.0
                        if diff > max_diff:
                            max_diff = diff
            base_label = f"{g_key[0]}/{g_key[1]}_s{g_key[3]}"
            label = base_label if base_label not in saliency_determinism else f"{base_label}/{g_key[2]}"
            saliency_determinism[label] = {"seeds": seeds, "max_diff": max_diff}
            if max_diff > 0.0:
                add_error(
                    label,
                    "saliency_determinism",
                    f"Saliency differs across seeds {seeds}: max_diff={max_diff:.3e} > 0.",
                )

    status = "PASS" if len(errors) == 0 else "FAIL"
    return {
        "status": status,
        "scores_path": str(scores_path),
        "n_files": len(csv_files),
        "n_rows": total_rows,
        "n_run_logs": len(run_log_files),
        "datasets": sorted(datasets_seen),
        "verified_hashes": verified_hashes,
        "graph_verified_datasets": graph_verified_datasets,
        "saliency_determinism": saliency_determinism,
        "n_errors": len(errors),
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_path", type=pathlib.Path, help="Path to results/scores/{ds} directory or CSV file.")
    parser.add_argument("--checkpoints-dir", type=pathlib.Path, default=ROOT / "results" / "checkpoints")
    parser.add_argument("--checkpoint-path", type=pathlib.Path, default=None, help="Explicit checkpoint .pt file.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=ROOT / "data")
    parser.add_argument("--expected-nodes-file", type=pathlib.Path, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--no-require-run-log", action="store_true")
    parser.add_argument("--no-verify-graph", action="store_true")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    report = validate_scores_dir(
        scores_path=args.scores_path,
        checkpoints_dir=args.checkpoints_dir,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        expected_nodes_file=args.expected_nodes_file,
        max_nodes=args.max_nodes,
        require_run_log=not args.no_require_run_log,
        verify_graph=not args.no_verify_graph,
        allow_failed=args.allow_failed,
        skip_checkpoint_hash=args.skip_checkpoint_hash,
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

    print(
        f"[{report['status']}] Checked {report['n_files']} CSV files "
        f"({report['n_rows']} rows, {report['n_run_logs']} run_log.json) in {report['scores_path']}"
    )
    if report["verified_hashes"]:
        for k, v in report["verified_hashes"].items():
            print(f"  - Hash {k}: {v['csv_sha256'][:16]}... verified={v['verified']} ({v['source']})")
    if report["saliency_determinism"]:
        for k, v in report["saliency_determinism"].items():
            print(f"  - Saliency determinism {k} (seeds={v['seeds']}): max_diff={v['max_diff']:.3e}")
    for w in report["warnings"]:
        print(f"  [WARN] {w}")
    if report["errors"]:
        print(f"Found {len(report['errors'])} error(s):")
        for idx, err in enumerate(report["errors"], 1):
            line_info = f" lines={err['lines']}" if "lines" in err else ""
            print(f"  {idx}. [{err['check']}] {err['file']}{line_info}: {err['message']}")
        print("FAIL")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
