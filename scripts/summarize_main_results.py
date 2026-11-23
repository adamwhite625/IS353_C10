"""Kết quả chính trên tập test (Mục 5 của báo cáo): Fidelity, AUROC/AP/precision, kèm CI 95%.

Đọc ``metrics.csv`` do ``scripts/evaluate.py`` sinh. Với mỗi (phương pháp, budget, metric):
1. lấy trung bình theo node: qua các seed explainer (hoặc repeat) trong từng g, rồi qua 5 g (mỗi GNN một trọng số);
2. bootstrap trên node (2000 lần, 95%) cho trung bình đó;
3. thêm SD giữa 5 seed GNN của trung bình theo từng g (độ nhạy theo mô hình).
Bảng 5-3 so sánh với số giữa kỳ (T06, 20 node val) chỉ để đối chiếu; hai tập node khác nhau.

Ví dụ:
  python scripts/summarize_main_results.py --eval-root results/eval --out results/analysis/main
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sna.core.provenance import execution_metadata, sha256, write_json
from sna.evaluation.stats import bootstrap_ci

METRICS = ["fid_drop", "fid_inv", "roc_auc", "average_precision", "precision"]
ORDER = ["saliency", "gnnexplainer", "pgexplainer", "nearest_first", "random"]
DATASETS = ["bashapes", "cora"]


def per_node_means(frame, metric):
    """Trung bình theo (method, budget, node): trước hết qua các seed explainer/repeat trong từng g,
    rồi qua các g (mỗi GNN một trọng số). Cách này tránh tính GNN 1000 hai lần ở Saliency
    (chạy e = 0 và e = 1 trên GNN 1000, chỉ e = 0 trên các GNN khác); bỏ NaN."""
    within_g = frame.groupby(["method", "budget", "g", "node_id"])[metric].mean()
    return within_g.groupby(level=["method", "budget", "node_id"]).mean().rename(metric)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", type=Path, default=Path("results/eval"))
    p.add_argument("--out", type=Path, default=Path("results/analysis/main"))
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--midterm-summary", type=Path,
                   default=Path("notebooks/05_duc_kha_evaluation/results/summary.csv"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    main_rows, by_g_rows, inputs = [], [], {}
    for ds in args.datasets:
        path = args.eval_root / f"{ds}_test" / "metrics.csv"
        inputs[f"{ds}/metrics.csv"] = sha256(path)
        m = pd.read_csv(path)
        m = m[m.status == "success"]
        for metric in METRICS:
            nodes = per_node_means(m, metric)
            for (method, budget), part in nodes.groupby(level=["method", "budget"]):
                r = bootstrap_ci(part.dropna().to_numpy(float), seed=0)
                main_rows.append({"dataset": ds, "method": method, "budget": budget, "metric": metric,
                                  "mean": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                  "n_valid_nodes": r["n"], "n_total_nodes": int(part.size)})
            # SD giữa seed GNN: trung bình node trong từng g rồi SD (ddof=1) qua 5 g
            per_g = m.groupby(["method", "budget", "g", "node_id"])[metric].mean().groupby(
                ["method", "budget", "g"]).mean()
            for (method, budget), part in per_g.groupby(level=["method", "budget"]):
                by_g_rows.append({"dataset": ds, "method": method, "budget": budget, "metric": metric,
                                  "mean_over_g": part.mean(), "sd_over_g": part.std(ddof=1),
                                  "min_over_g": part.min(), "max_over_g": part.max(), "n_g": int(part.size)})
    main_df = pd.DataFrame(main_rows)
    by_g_df = pd.DataFrame(by_g_rows)

    # Đối chiếu số giữa kỳ (T06, BA-Shapes và Cora, 20 node val, k = 5, một seed)
    mid = pd.DataFrame()
    if args.midterm_summary.is_file():
        inputs["midterm_summary.csv"] = sha256(args.midterm_summary)
        mid = pd.read_csv(args.midterm_summary)

    outputs = {}
    for name, frame in (("main_by_method_budget", main_df), ("main_sd_over_gnn_seeds", by_g_df)):
        out = args.out / f"{name}.csv"
        frame.to_csv(out, index=False, na_rep="NaN", lineterminator="\n")
        outputs[out.name] = sha256(out)
    if len(mid):
        out = args.out / "midterm_reference.csv"
        mid.to_csv(out, index=False, na_rep="NaN", lineterminator="\n")
        outputs[out.name] = sha256(out)
    write_json(args.out / "manifest.json", {
        **execution_metadata(), "argv": sys.argv, "inputs": inputs, "outputs": outputs,
        "bootstrap": {"unit": "node", "repeats": 2000, "confidence": 0.95, "seed": 0},
        "aggregation": "trung bình theo node: qua e/repeat trong từng g rồi qua g (mỗi GNN một trọng số); CI bootstrap trên node"})
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
