"""F10 Benchmark: Custom GNNExplainer vs Official PyG GNNExplainer + set_masks Audit (Khoi - PN2).

Compares on the 20 dev nodes of BA-Shapes and Cora (CPU, same midterm gcn_ckpt.pt, seed=42):
  1. custom_mean      : Current repo GNNExplainer (loss_size = 0.005 * mean, coeff_ent = 0.1, edge_weight)
  2. custom_sum       : Repo GNNExplainer with PyG-style sum size-loss (loss_size = 0.005 * sum, coeff_ent = 0.1, edge_weight)
  3. custom_sum_ent1  : Repo GNNExplainer with PyG-style coeffs (loss_size = 0.005 * sum, coeff_ent = 1.0, edge_weight)
  4. pyg_official     : torch_geometric.explain.GNNExplainer (edge_reduction='sum', edge_size=0.005, edge_ent=1.0, set_masks)

Measures:
  - Mask distribution stats (median, mean, min, max, % > 0.9, % < 0.1, gap5-6)
  - Quality: Fid- (Drop)@5, Fid+ (Inv)@5, Motif AUROC & AP & Precision@5 (BA-Shapes)
  - Pairwise agreement: kappa@5 (Appendix A formula), Jaccard@5, Kendall tau-b
  - Quantitative edge_weight=0 vs PyG set_masks=0 vs Hard Edge Deletion audit (Appendix B Q&A)

Usage:
    python scripts/check_gnnexplainer_ref.py --json-out results/f10_gnnexplainer_ref.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import sna
from sna.evaluation.core import (
    agreement,
    fidelity_selected,
    ground_truth_metrics,
    rank_edges,
    top_indices,
)
from sna.explainers.gnnexplainer import GNNExplainer as CustomGNNExplainer
from sna.explainers.saliency import SaliencyExplainer
from sna.models.gat import GAT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tie_seed_for(dataset: str, node: int, method: str, repeat: int = 0) -> int:
    text = f"{dataset}:{node}:{method}:{repeat}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def kappa_at_k(order_a: np.ndarray, order_b: np.ndarray, k: int) -> float:
    """Chance-corrected top-k overlap kappa@k (Appendix A of KE_HOACH_CUOI_KY_FINAL.md).

    kappa@k = (|A cap B| - k_eff^2 / m) / (k_eff - k_eff^2 / m),
    undefined (NaN) when m == 0 or k_eff == m or k_eff == 0.
    """
    m = len(order_a)
    if m == 0 or len(order_b) != m:
        return float("nan")
    k_eff = min(int(k), m)
    if k_eff == 0 or k_eff == m:
        return float("nan")
    sa = set(top_indices(order_a, k_eff).tolist())
    sb = set(top_indices(order_b, k_eff).tolist())
    inter = len(sa & sb)
    expected = (k_eff * k_eff) / float(m)
    denom = k_eff - expected
    if denom <= 0.0:
        return float("nan")
    return float((inter - expected) / denom)


def _mean_std(vals: List[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None, "n_valid": 0, "n_total": len(vals)}
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "n_valid": int(len(arr)),
        "n_total": int(len(vals)),
    }


def run_pyg_gnnexplainer_for_node(
    model: torch.nn.Module,
    data: Dict[str, Any],
    node_id: int,
    cand: Any,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> Tuple[List[float], List[float], float]:
    """Run official PyG GNNExplainer once and return (mean_agg_scores, max_agg_scores, runtime_sec)."""
    set_seed(seed)
    t0 = time.perf_counter()

    x = data["x"].cpu()
    edge_index = data["edge_index"].cpu()
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=PyGGNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    try:
        explanation = explainer(x, edge_index, index=node_id)
        edge_mask = explanation.edge_mask.detach().cpu().numpy()
    finally:
        clear_masks(model)

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edge_lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src, dst))}

    scores_mean: List[float] = []
    scores_max: List[float] = []
    for u, v in cand.candidate_pairs:
        idx_uv = edge_lookup.get((u, v))
        idx_vu = edge_lookup.get((v, u))
        vals = []
        if idx_uv is not None:
            vals.append(float(edge_mask[idx_uv]))
        if idx_vu is not None:
            vals.append(float(edge_mask[idx_vu]))
        if not vals:
            scores_mean.append(0.0)
            scores_max.append(0.0)
        else:
            scores_mean.append(float(np.mean(vals)))
            scores_max.append(float(max(vals)))

    return scores_mean, scores_max, float(time.perf_counter() - t0)


def audit_set_masks_vs_edge_weight(
    data_root: pathlib.Path,
    datasets: List[str],
    n_nodes: int = 20,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Quantitatively compare Hard Edge Deletion vs edge_weight=0 vs PyG set_masks=0.

    Answers Appendix B Q&A: 'Vi sao khong dung set_masks cua PyG?' with real measurements.
    """
    audit: Dict[str, Any] = {"gcn": {}, "gat_synthetic_check": {}}

    for ds in datasets:
        ds_dir = data_root / ds
        data = torch.load(ds_dir / "data.pt", map_location="cpu", weights_only=False)
        ck = torch.load(ds_dir / "gcn_ckpt.pt", map_location="cpu", weights_only=False)
        model = sna.GCN(ck["in_dim"], ck["hidden"], ck["n_classes"], ck.get("dropout", 0.0))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        x = data["x"].cpu()
        edge_index = data["edge_index"].cpu()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=-1).cpu()

        with open(ds_dir / "val_nodes.json", "r", encoding="utf-8-sig") as f:
            val_nodes = [int(v) for v in json.load(f)["nodes"][:n_nodes]]

        sal = SaliencyExplainer()
        endpoints = edge_index.t().tolist()

        ew_logit_diffs: List[float] = []
        ew_prob_diffs: List[float] = []
        sm_logit_diffs: List[float] = []
        sm_prob_diffs: List[float] = []
        sm_pred_flips = 0
        ew_pred_flips = 0

        for node in val_nodes:
            tc = int(pred[node].item())
            cand = sna.extract_candidate_edges(edge_index, node, num_hops=2, num_nodes=x.size(0))
            res = sal.explain(model, data, node, tc, cand, seed=0)
            order = rank_edges(res.scores, seed=tie_seed_for(ds, node, "saliency", 0))
            selected = top_indices(order, top_k)
            chosen = {tuple(cand.candidate_pairs[i]) for i in selected}

            keep_mask = torch.tensor(
                [(min(u, v), max(u, v)) not in chosen for u, v in endpoints],
                dtype=torch.bool,
            )
            w_zero = keep_mask.to(torch.float32)

            with torch.no_grad():
                # 1. Hard edge deletion (ground-truth Fidelity Drop definition)
                logits_hard = model(x, edge_index[:, keep_mask])[node]
                prob_hard = float(logits_hard.softmax(-1)[tc].item())
                pred_hard = int(logits_hard.argmax(-1).item())

                # 2. Passing edge_weight=0 into GCN.forward (pre-normalization)
                logits_ew = model(x, edge_index, edge_weight=w_zero)[node]
                prob_ew = float(logits_ew.softmax(-1)[tc].item())
                pred_ew = int(logits_ew.argmax(-1).item())

                # 3. PyG set_masks with mask=0 (post-normalization in MessagePassing)
                try:
                    set_masks(model, w_zero.clone(), edge_index, apply_sigmoid=False)
                    logits_sm = model(x, edge_index)[node]
                    prob_sm = float(logits_sm.softmax(-1)[tc].item())
                    pred_sm = int(logits_sm.argmax(-1).item())
                finally:
                    clear_masks(model)

            ew_logit_diffs.append(float((logits_ew - logits_hard).abs().max().item()))
            ew_prob_diffs.append(abs(prob_ew - prob_hard))
            sm_logit_diffs.append(float((logits_sm - logits_hard).abs().max().item()))
            sm_prob_diffs.append(abs(prob_sm - prob_hard))
            if pred_ew != pred_hard:
                ew_pred_flips += 1
            if pred_sm != pred_hard:
                sm_pred_flips += 1

        audit["gcn"][ds] = {
            "n_nodes": len(val_nodes),
            "top_k": top_k,
            "edge_weight_vs_hard_del": {
                "max_logit_diff": float(np.max(ew_logit_diffs)),
                "mean_logit_diff": float(np.mean(ew_logit_diffs)),
                "max_prob_diff": float(np.max(ew_prob_diffs)),
                "mean_prob_diff": float(np.mean(ew_prob_diffs)),
                "pred_disagreements": ew_pred_flips,
            },
            "set_masks_vs_hard_del": {
                "max_logit_diff": float(np.max(sm_logit_diffs)),
                "mean_logit_diff": float(np.mean(sm_logit_diffs)),
                "max_prob_diff": float(np.max(sm_prob_diffs)),
                "mean_prob_diff": float(np.mean(sm_prob_diffs)),
                "pred_disagreements": sm_pred_flips,
            },
        }

    # GAT synthetic check on BA-Shapes graph (untrained deterministic GAT seed=0)
    ba_data = torch.load(data_root / "bashapes" / "data.pt", map_location="cpu", weights_only=False)
    set_seed(0)
    gat = GAT(in_dim=ba_data["x"].size(1), hidden_dim=16, out_dim=4, heads=4, dropout=0.0).eval()
    x_ba = ba_data["x"].cpu()
    ei_ba = ba_data["edge_index"].cpu()
    endpoints_ba = ei_ba.t().tolist()
    with open(data_root / "bashapes" / "val_nodes.json", "r", encoding="utf-8-sig") as f:
        ba_val = [int(v) for v in json.load(f)["nodes"][:n_nodes]]

    gat_sm_logit_diffs = []
    gat_sm_prob_diffs = []
    for node in ba_val:
        cand = sna.extract_candidate_edges(ei_ba, node, num_hops=2, num_nodes=x_ba.size(0))
        chosen = set(cand.candidate_pairs[: min(top_k, len(cand.candidate_pairs))])
        keep_mask = torch.tensor(
            [(min(u, v), max(u, v)) not in chosen for u, v in endpoints_ba],
            dtype=torch.bool,
        )
        w_zero = keep_mask.to(torch.float32)
        with torch.no_grad():
            logits_hard = gat(x_ba, ei_ba[:, keep_mask])[node]
            tc = int(logits_hard.argmax(-1).item())
            prob_hard = float(logits_hard.softmax(-1)[tc].item())
            try:
                set_masks(gat, w_zero.clone(), ei_ba, apply_sigmoid=False)
                logits_sm = gat(x_ba, ei_ba)[node]
                prob_sm = float(logits_sm.softmax(-1)[tc].item())
            finally:
                clear_masks(gat)
        gat_sm_logit_diffs.append(float((logits_sm - logits_hard).abs().max().item()))
        gat_sm_prob_diffs.append(abs(prob_sm - prob_hard))

    audit["gat_synthetic_check"] = {
        "note": f"Untrained 2-layer GAT (seed=0) on BA-Shapes {len(ba_val)} dev nodes deleting first {top_k} candidate pairs",
        "max_logit_diff_set_masks_vs_hard_del": float(np.max(gat_sm_logit_diffs)),
        "mean_logit_diff_set_masks_vs_hard_del": float(np.mean(gat_sm_logit_diffs)),
        "max_prob_diff_set_masks_vs_hard_del": float(np.max(gat_sm_prob_diffs)),
        "mean_prob_diff_set_masks_vs_hard_del": float(np.mean(gat_sm_prob_diffs)),
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=ROOT / "data")
    parser.add_argument("--datasets", nargs="+", default=["bashapes", "cora"])
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    t_all = time.perf_counter()
    variants = ["custom_mean", "custom_sum", "custom_sum_ent1", "pyg_official", "pyg_official_max"]
    custom_expl = CustomGNNExplainer()

    report: Dict[str, Any] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "budget_k": args.budget,
        "n_nodes": args.n_nodes,
        "device": "cpu",
        "datasets": {},
    }

    for ds in args.datasets:
        ds_dir = args.data_root / ds
        data = torch.load(ds_dir / "data.pt", map_location="cpu", weights_only=False)
        ck = torch.load(ds_dir / "gcn_ckpt.pt", map_location="cpu", weights_only=False)
        model = sna.GCN(ck["in_dim"], ck["hidden"], ck["n_classes"], ck.get("dropout", 0.0))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            pred = model(data["x"], data["edge_index"]).argmax(dim=-1).cpu()
            orig_probs = model(data["x"], data["edge_index"]).softmax(dim=-1).cpu()

        with open(ds_dir / "val_nodes.json", "r", encoding="utf-8-sig") as f:
            val_nodes = [int(v) for v in json.load(f)["nodes"][: args.n_nodes]]

        gt_mask = data.get("gt_edge_mask") if ds == "bashapes" else None

        scores_store: Dict[str, Dict[int, np.ndarray]] = {v: {} for v in variants}
        orders_store: Dict[str, Dict[int, np.ndarray]] = {v: {} for v in variants}
        metrics_store: Dict[str, Dict[str, List[float]]] = {
            v: {
                "fid_drop": [],
                "fid_inv": [],
                "roc_auc": [],
                "average_precision": [],
                "precision": [],
                "runtime_sec": [],
                "node_range": [],
                "gap_5_6": [],
            }
            for v in variants
        }

        for node in val_nodes:
            tc = int(pred[node].item())
            orig_p = float(orig_probs[node, tc].item())
            cand = sna.extract_candidate_edges(
                data["edge_index"], node, num_hops=2, num_nodes=data["x"].size(0)
            )
            pairs = cand.candidate_pairs
            m = len(pairs)

            # 1. custom_mean (current midterm formula)
            r_mean = custom_expl.explain(
                model,
                data,
                node,
                tc,
                cand,
                config={
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "coeff_size": 0.005,
                    "coeff_ent": 0.1,
                    "size_loss_mode": "mean",
                },
                seed=args.seed,
            )
            # 2. custom_sum (size-loss sum like PyG, keep coeff_ent=0.1)
            r_sum = custom_expl.explain(
                model,
                data,
                node,
                tc,
                cand,
                config={
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "coeff_size": 0.005,
                    "coeff_ent": 0.1,
                    "size_loss_mode": "sum",
                },
                seed=args.seed,
            )
            # 3. custom_sum_ent1 (size-loss sum + coeff_ent=1.0 like PyG, still edge_weight)
            r_sum_ent1 = custom_expl.explain(
                model,
                data,
                node,
                tc,
                cand,
                config={
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "coeff_size": 0.005,
                    "coeff_ent": 1.0,
                    "size_loss_mode": "sum",
                },
                seed=args.seed,
            )
            # 4 & 5. Official PyG GNNExplainer (single run, both mean & max undirected aggregation)
            pyg_scores_mean, pyg_scores_max, pyg_rt = run_pyg_gnnexplainer_for_node(
                model, data, node, cand, epochs=args.epochs, lr=args.lr, seed=args.seed
            )

            variant_outputs = {
                "custom_mean": (r_mean.scores, r_mean.runtime_sec),
                "custom_sum": (r_sum.scores, r_sum.runtime_sec),
                "custom_sum_ent1": (r_sum_ent1.scores, r_sum_ent1.runtime_sec),
                "pyg_official": (pyg_scores_mean, pyg_rt),
                "pyg_official_max": (pyg_scores_max, pyg_rt),
            }

            for vname, (sc_list, rt) in variant_outputs.items():
                arr = np.asarray(sc_list, dtype=float)
                scores_store[vname][node] = arr
                order = rank_edges(arr, seed=tie_seed_for(ds, node, vname, 0))
                orders_store[vname][node] = order
                selected = top_indices(order, args.budget)

                fid = fidelity_selected(model, data, node, tc, pairs, selected, orig_prob=orig_p)
                gt = ground_truth_metrics(pairs, arr, data["edge_index"], gt_mask, selected)

                sorted_desc = np.sort(arr)[::-1]
                gap_5_6 = float(sorted_desc[args.budget - 1] - sorted_desc[args.budget]) if m > args.budget else float("nan")

                ms = metrics_store[vname]
                ms["fid_drop"].append(fid["fid_drop"])
                ms["fid_inv"].append(fid["fid_inv"])
                ms["roc_auc"].append(gt["roc_auc"])
                ms["average_precision"].append(gt["average_precision"])
                ms["precision"].append(gt["precision"])
                ms["runtime_sec"].append(float(rt))
                ms["node_range"].append(float(np.ptp(arr)) if m > 0 else float("nan"))
                ms["gap_5_6"].append(gap_5_6)

        # Summarize distributions and metrics
        ds_summary: Dict[str, Any] = {"variants": {}, "pairwise_agreement": {}}
        for vname in variants:
            pooled = np.concatenate([scores_store[vname][n] for n in val_nodes])
            ms = metrics_store[vname]
            valid_gaps = [g for g in ms["gap_5_6"] if np.isfinite(g)]
            ds_summary["variants"][vname] = {
                "score_distribution": {
                    "n_edges": int(len(pooled)),
                    "min": float(np.min(pooled)),
                    "p25": float(np.percentile(pooled, 25)),
                    "median": float(np.median(pooled)),
                    "mean": float(np.mean(pooled)),
                    "p75": float(np.percentile(pooled, 75)),
                    "max": float(np.max(pooled)),
                    "std": float(np.std(pooled, ddof=1)),
                    "pct_gt_0_9": float(np.mean(pooled > 0.9) * 100.0),
                    "pct_lt_0_1": float(np.mean(pooled < 0.1) * 100.0),
                    "median_node_range": float(np.nanmedian(ms["node_range"])),
                    "median_gap_5_6": float(np.median(valid_gaps)) if valid_gaps else None,
                },
                "metrics_k5": {
                    "fid_drop": _mean_std(ms["fid_drop"]),
                    "fid_inv": _mean_std(ms["fid_inv"]),
                    "roc_auc": _mean_std(ms["roc_auc"]),
                    "average_precision": _mean_std(ms["average_precision"]),
                    "precision": _mean_std(ms["precision"]),
                    "runtime_sec": _mean_std(ms["runtime_sec"]),
                },
            }

        # Pairwise agreement (kappa@5, Jaccard@5, Kendall tau-b)
        for va, vb in itertools.combinations(variants, 2):
            kappas, jacs, taus = [], [], []
            for node in val_nodes:
                sa = scores_store[va][node]
                sb = scores_store[vb][node]
                oa = orders_store[va][node]
                ob = orders_store[vb][node]
                kappas.append(kappa_at_k(oa, ob, args.budget))
                agr = agreement(sa, sb, args.budget, order_a=oa, order_b=ob)
                jacs.append(agr["jaccard"])
                taus.append(agr["kendall_tau_b"])
            ds_summary["pairwise_agreement"][f"{va}__vs__{vb}"] = {
                "kappa_at_5": _mean_std(kappas),
                "jaccard_at_5": _mean_std(jacs),
                "kendall_tau_b": _mean_std(taus),
            }

        report["datasets"][ds] = ds_summary

    report["set_masks_vs_edge_weight_audit"] = audit_set_masks_vs_edge_weight(
        args.data_root, args.datasets, n_nodes=args.n_nodes, top_k=args.budget
    )
    report["elapsed_sec"] = round(time.perf_counter() - t_all, 3)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
