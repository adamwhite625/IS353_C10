"""GATE 0 Step 3 & F8 Pre/Post Parity Verification Script (Khoi - PN2).

Verifies:
1. Asset SHA-256 hashes and 100% prediction match for midterm checkpoints.
2. Exact reproduction of Saliency and GNNExplainer scores (seed=42, CPU) on the
   20 dev nodes of BA-Shapes and Cora against notebooks/01_midterm/edge_scores.csv.
3. Optional snapshot save/compare to prove F8 parity (before vs after refactor:
   max_diff < 1e-6).
4. Empirical demonstration of PGExplainer non-reproducibility due to F1
   (MLP initialized before seeding) per Section 6.7 Step 3 of the group plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import sna
from sna.explainers.pgexplainer import PGExplainer, PGExplainerModel


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_assets_and_preds(data_root: pathlib.Path, datasets: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for ds in datasets:
        ds_dir = data_root / ds
        hashes = {}
        for fname in ("data.pt", "gcn_ckpt.pt", "config.json", "val_nodes.json"):
            fpath = ds_dir / fname
            hashes[fname] = sha256_file(fpath)

        data = torch.load(ds_dir / "data.pt", map_location="cpu", weights_only=False)
        ck = torch.load(ds_dir / "gcn_ckpt.pt", map_location="cpu", weights_only=False)
        model = sna.GCN(
            in_dim=ck["in_dim"],
            hidden_dim=ck["hidden"],
            out_dim=ck["n_classes"],
            dropout=ck.get("dropout", 0.0),
        )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            pred = model(data["x"], data["edge_index"]).argmax(dim=-1).cpu()
        saved_pred = torch.as_tensor(ck["pred"]).cpu()
        pred_match = bool(torch.equal(pred, saved_pred))
        n_match = int((pred == saved_pred).sum().item())
        report[ds] = {
            "hashes": hashes,
            "pred_match": pred_match,
            "n_match": n_match,
            "n_total": int(len(saved_pred)),
        }
    return report


def run_dev_scores(
    data_root: pathlib.Path,
    datasets: List[str],
    seed: int = 42,
    n_nodes: int = 20,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        ds_dir = data_root / ds
        data = torch.load(ds_dir / "data.pt", map_location="cpu", weights_only=False)
        ck = torch.load(ds_dir / "gcn_ckpt.pt", map_location="cpu", weights_only=False)
        model = sna.GCN(
            in_dim=ck["in_dim"],
            hidden_dim=ck["hidden"],
            out_dim=ck["n_classes"],
            dropout=ck.get("dropout", 0.0),
        )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            pred = model(data["x"], data["edge_index"]).argmax(dim=-1).cpu()

        with open(ds_dir / "val_nodes.json", "r", encoding="utf-8-sig") as f:
            val_nodes = [int(x) for x in json.load(f)["nodes"][:n_nodes]]

        ds_out: Dict[str, Any] = {}
        for node in val_nodes:
            cand = sna.extract_candidate_edges(
                data["edge_index"], node, num_hops=2, num_nodes=data["x"].size(0)
            )
            tc = int(pred[node].item())

            _, sal_scores, sal_rt, sal_st, sal_err = sna.explain_edges(
                model=model,
                data=data,
                node_id=node,
                target_class=tc,
                candidate_edges=cand,
                config={"method": "saliency"},
                seed=seed,
            )
            if sal_st != "success":
                raise RuntimeError(f"Saliency failed on {ds} node {node}: {sal_err}")

            _, gnn_scores, gnn_rt, gnn_st, gnn_err = sna.explain_edges(
                model=model,
                data=data,
                node_id=node,
                target_class=tc,
                candidate_edges=cand,
                config={"method": "gnnexplainer", "epochs": 200, "lr": 0.01, "size_loss_mode": "mean"},
                seed=seed,
            )
            if gnn_st != "success":
                raise RuntimeError(f"GNNExplainer failed on {ds} node {node}: {gnn_err}")

            ds_out[str(node)] = {
                "target_class": tc,
                "edge_ids": list(cand.edge_ids),
                "pairs": [[int(u), int(v)] for u, v in cand.candidate_pairs],
                "saliency": [float(s) for s in sal_scores],
                "gnnexplainer": [float(s) for s in gnn_scores],
                "saliency_runtime_sec": float(sal_rt),
                "gnnexplainer_runtime_sec": float(gnn_rt),
            }
        results[ds] = ds_out
    return results


def compare_with_midterm_csv(
    computed: Dict[str, Dict[str, Any]],
    csv_path: pathlib.Path,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    summary: Dict[str, Any] = {
        "midterm_csv": str(csv_path),
        "midterm_csv_sha256": sha256_file(csv_path),
        "comparisons": {},
    }
    for ds, nodes_map in computed.items():
        summary["comparisons"][ds] = {}
        for method in ("saliency", "gnnexplainer"):
            diffs_raw: List[float] = []
            diffs_round6: List[float] = []
            worst_entry: Dict[str, Any] = {}
            max_raw = -1.0
            n_edges = 0
            n_nodes_matched = 0

            for node_str, info in nodes_map.items():
                node = int(node_str)
                part = df[
                    (df["dataset"] == ds)
                    & (df["method"] == method)
                    & (df["node_id"] == node)
                ].sort_values("edge_id")

                expected_pairs = [tuple(p) for p in info["pairs"]]
                csv_pairs = list(zip(part["u"].astype(int), part["v"].astype(int)))
                if csv_pairs != expected_pairs or part["edge_id"].astype(int).tolist() != info["edge_ids"]:
                    raise ValueError(f"Edge alignment mismatch on {ds} {method} node {node}")
                if int(part["target_class"].iloc[0]) != info["target_class"]:
                    raise ValueError(f"Target class mismatch on {ds} {method} node {node}")

                csv_scores = part["importance_score"].to_numpy(dtype=float)
                new_scores = np.asarray(info[method], dtype=float)
                round6_scores = np.round(new_scores, 6)

                raw_d = np.abs(new_scores - csv_scores)
                r6_d = np.abs(round6_scores - csv_scores)

                local_max_idx = int(np.argmax(raw_d))
                if float(raw_d[local_max_idx]) > max_raw:
                    max_raw = float(raw_d[local_max_idx])
                    worst_entry = {
                        "node_id": node,
                        "edge_id": int(info["edge_ids"][local_max_idx]),
                        "pair": info["pairs"][local_max_idx],
                        "computed": float(new_scores[local_max_idx]),
                        "csv": float(csv_scores[local_max_idx]),
                        "diff": max_raw,
                    }

                diffs_raw.extend(raw_d.tolist())
                diffs_round6.extend(r6_d.tolist())
                n_edges += len(csv_scores)
                n_nodes_matched += 1

            summary["comparisons"][ds][method] = {
                "n_nodes": n_nodes_matched,
                "n_edges": n_edges,
                "max_diff_raw": float(np.max(diffs_raw)),
                "mean_diff_raw": float(np.mean(diffs_raw)),
                "max_diff_rounded_6dp": float(np.max(diffs_round6)),
                "mean_diff_rounded_6dp": float(np.mean(diffs_round6)),
                "worst_entry": worst_entry,
            }
    return summary


def compare_with_snapshot(
    computed: Dict[str, Dict[str, Any]],
    snapshot_path: pathlib.Path,
) -> Dict[str, Any]:
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    for ds, nodes_map in computed.items():
        out[ds] = {}
        for method in ("saliency", "gnnexplainer"):
            diffs = []
            for node_str, info in nodes_map.items():
                a = np.asarray(info[method], dtype=float)
                b = np.asarray(snap[ds][node_str][method], dtype=float)
                diffs.append(float(np.max(np.abs(a - b))))
            out[ds][method] = {
                "n_nodes": len(diffs),
                "max_diff": float(np.max(diffs)),
                "pass_1e_6": bool(float(np.max(diffs)) < 1e-6),
            }
    return out


def check_pgexplainer_non_reproducibility(
    data_root: pathlib.Path,
    csv_path: pathlib.Path,
    seed: int = 42,
    epochs: int = 10,
    n_nodes: int = 20,
) -> Dict[str, Any]:
    """Demonstrates bug F1: PGExplainerModel initializes MLP BEFORE setting seed."""
    ds_dir = data_root / "bashapes"
    data = torch.load(ds_dir / "data.pt", map_location="cpu", weights_only=False)
    ck = torch.load(ds_dir / "gcn_ckpt.pt", map_location="cpu", weights_only=False)
    model = sna.GCN(ck["in_dim"], ck["hidden"], ck["n_classes"], ck.get("dropout", 0.0))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(data["x"], data["edge_index"]).argmax(dim=-1).cpu()
    with open(ds_dir / "val_nodes.json", "r", encoding="utf-8-sig") as f:
        val_nodes = [int(x) for x in json.load(f)["nodes"][:n_nodes]]

    # Two consecutive runs using the exact current pattern (init before seed)
    run_scores = []
    for _ in range(2):
        pg = PGExplainerModel(model=model, device=torch.device("cpu"))
        pg.train_on_graph(data=data, epochs=epochs, lr=0.005, seed=seed)
        expl = PGExplainer(trained_model=pg)
        scores_all = []
        for node in val_nodes:
            cand = sna.extract_candidate_edges(
                data["edge_index"], node, num_hops=2, num_nodes=data["x"].size(0)
            )
            res = expl.explain(model, data, node, int(pred[node]), cand, seed=seed)
            scores_all.extend(res.scores)
        run_scores.append(np.asarray(scores_all, dtype=float))

    diff_two_runs = np.abs(run_scores[0] - run_scores[1])

    # Also compare run 1 (if epochs=40) or note against midterm CSV
    df = pd.read_csv(csv_path)
    csv_pg = []
    for node in val_nodes:
        part = df[
            (df["dataset"] == "bashapes")
            & (df["method"] == "pgexplainer")
            & (df["node_id"] == node)
        ].sort_values("edge_id")
        csv_pg.extend(part["importance_score"].to_numpy(dtype=float).tolist())
    diff_vs_csv = np.abs(run_scores[0] - np.asarray(csv_pg, dtype=float))

    return {
        "dataset": "bashapes",
        "seed": seed,
        "epochs": epochs,
        "n_edges": int(len(run_scores[0])),
        "max_diff_between_two_same_seed_runs": float(np.max(diff_two_runs)),
        "mean_diff_between_two_same_seed_runs": float(np.mean(diff_two_runs)),
        "max_diff_run1_vs_midterm_csv": float(np.max(diff_vs_csv)),
        "reproducible_before_f1": bool(float(np.max(diff_two_runs)) == 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=ROOT / "data")
    parser.add_argument(
        "--midterm-csv",
        type=pathlib.Path,
        default=ROOT / "notebooks" / "01_midterm" / "edge_scores.csv",
    )
    parser.add_argument("--datasets", nargs="+", default=["bashapes", "cora"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--save-snapshot", type=pathlib.Path, default=None)
    parser.add_argument("--compare-snapshot", type=pathlib.Path, default=None)
    parser.add_argument("--check-pg-f1", action="store_true")
    parser.add_argument("--pg-epochs", type=int, default=40)
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    t0 = time.time()
    assets = verify_assets_and_preds(args.data_root, args.datasets)
    computed = run_dev_scores(args.data_root, args.datasets, seed=args.seed, n_nodes=args.n_nodes)

    if args.save_snapshot is not None:
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.save_snapshot.write_text(json.dumps(computed, indent=2), encoding="utf-8")

    csv_cmp = compare_with_midterm_csv(computed, args.midterm_csv)
    snap_cmp = None
    if args.compare_snapshot is not None:
        snap_cmp = compare_with_snapshot(computed, args.compare_snapshot)

    pg_check = None
    if args.check_pg_f1:
        pg_check = check_pgexplainer_non_reproducibility(
            args.data_root,
            args.midterm_csv,
            seed=args.seed,
            epochs=args.pg_epochs,
            n_nodes=args.n_nodes,
        )

    payload = {
        "elapsed_sec": round(time.time() - t0, 3),
        "seed": args.seed,
        "n_nodes": args.n_nodes,
        "device": "cpu",
        "assets": assets,
        "midterm_csv_comparison": csv_cmp,
        "snapshot_comparison": snap_cmp,
        "pgexplainer_f1_check": pg_check,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
