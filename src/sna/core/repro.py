"""Reproducibility utilities: SHA-256 hashing, environment metadata, and manifest generation."""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional


def compute_sha256(file_path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file."""
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Return current git commit hash and dirty flag."""
    cwd = str(repo_root) if repo_root else None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return {"git_commit": commit, "git_dirty": bool(status)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": True}


def get_env_versions() -> Dict[str, str]:
    """Collect Python and key ML library versions."""
    versions: Dict[str, str] = {"python": platform.python_version()}
    for mod_name in ("torch", "torch_geometric", "numpy", "pandas", "scipy", "sklearn", "yaml"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not_installed"
    return versions


def build_manifest(
    *,
    command: str,
    device: str,
    data_files: Optional[Dict[str, pathlib.Path]] = None,
    checkpoint_files: Optional[Dict[str, pathlib.Path]] = None,
    score_files: Optional[Dict[str, pathlib.Path]] = None,
    node_list_file: Optional[pathlib.Path] = None,
    nodes: Optional[List[int]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    error_rows: int = 0,
    extra: Optional[Dict[str, Any]] = None,
    repo_root: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Build a structured manifest dictionary per Section 6.5 of KE_HOACH_CUOI_KY_FINAL.md."""
    git_info = get_git_info(repo_root)
    manifest: Dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "command": command,
        "device": device,
        "versions": get_env_versions(),
        "data_sha256": {
            k: compute_sha256(pathlib.Path(v)) for k, v in (data_files or {}).items()
        },
        "checkpoints_sha256": {
            k: compute_sha256(pathlib.Path(v)) for k, v in (checkpoint_files or {}).items()
        },
        "scores_sha256": {
            k: compute_sha256(pathlib.Path(v)) for k, v in (score_files or {}).items()
        },
        "node_list_file": str(node_list_file) if node_list_file else None,
        "node_list_sha256": compute_sha256(node_list_file) if node_list_file else None,
        "num_nodes": len(nodes) if nodes is not None else None,
        "nodes": nodes,
        "protocol": protocol or {},
        "error_rows": int(error_rows),
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: Dict[str, Any], out_path: pathlib.Path) -> pathlib.Path:
    """Write manifest JSON atomically to out_path."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out

