"""Run FB-3 Error Analysis mechanism tests T2 and T3 across BA-Shapes test nodes."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sna.analysis.mechanisms_t2_t3 import evaluate_dataset_t2_t3
from sna.core.config import DATA_DIR
from sna.core.repro import build_manifest, save_manifest
from sna.models.checkpoint import load_gcn_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2 and T3 mechanism tests on BA-Shapes.")
    parser.add_argument("--dataset", default="bashapes", choices=["bashapes"])
    parser.add_argument("--gnn-seeds", nargs="+", type=int, default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--ckpt-root", type=pathlib.Path, default=ROOT / "results" / "checkpoints")
    parser.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "results" / "analysis")
    args = parser.parse_args()

    data_path = DATA_DIR / args.dataset / "data.pt"
    nodes_path = DATA_DIR / args.dataset / "test_nodes.json"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    with open(nodes_path, "r", encoding="utf-8") as f:
        test_nodes = json.load(f)["nodes"]

    all_rows = []
    ckpt_files = {}
    for g_seed in args.gnn_seeds:
        ckpt_path = args.ckpt_root / args.dataset / f"gcn_s{g_seed}.pt"
        ckpt_files[f"gcn_s{g_seed}"] = ckpt_path
        model, pred, _ = load_gcn_checkpoint(ckpt_path=ckpt_path)
        rows = evaluate_dataset_t2_t3(model=model, data=data, node_ids=test_nodes, pred=pred)
        for r in rows:
            r["gnn_seed"] = g_seed
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "a3_t2_t3_nodes.csv"
    df.to_csv(csv_path, index=False)

    motif_df = df[df["num_motif_edges_2hop"] > 0]
    summary = {
        "total_rows": int(len(df)),
        "rows_with_motif_edges": int(len(motif_df)),
        "mean_fid_motif": float(motif_df["fid_motif"].mean()) if len(motif_df) else float("nan"),
        "mean_fid_1hop": float(motif_df["fid_1hop"].mean()) if len(motif_df) else float("nan"),
        "mean_fid_motif_recomp_x": float(motif_df["fid_motif_recomp_x"].mean()) if len(motif_df) else float("nan"),
        "mean_t3_motif_delta": float(motif_df["t3_motif_delta"].mean()) if len(motif_df) else float("nan"),
        "pct_fid_motif_lt_0_10": float((motif_df["fid_motif"] < 0.10).mean()) if len(motif_df) else float("nan"),
    }
    summary_path = args.out_dir / "a3_t2_t3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest = build_manifest(
        command="python " + " ".join(sys.argv),
        device="cpu",
        data_files={args.dataset: data_path},
        checkpoint_files=ckpt_files,
        node_list_file=nodes_path,
        nodes=test_nodes,
        protocol={"tests": ["T2", "T3"], "gnn_seeds": args.gnn_seeds},
        repo_root=ROOT,
    )
    save_manifest(manifest, args.out_dir / "a3_t2_t3_manifest.json")
    print(f"Saved T2/T3 results to {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()

