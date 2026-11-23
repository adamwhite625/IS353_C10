"""Phân tích FB-2: bất ổn định nội tại, nhạy theo GNN và bất đồng giữa phương pháp.

Đọc đầu ra của ``scripts/evaluate.py`` (``agreement.csv``, ``pred_change.csv``,
``metrics.csv``, ``manifest.json``) và sinh Bảng A2-1, A2-2, A2-3 cùng Hình A2 theo
Mục 7.3-7.5 của ``docs/KE_HOACH_CUOI_KY_FINAL.md``. Mọi đại lượng tính cho TỪNG NODE
rồi mới lấy trung bình/CI trên node (đơn vị bootstrap là node).

Định nghĩa (Agr = κ@5, chính; Jaccard@5 là phụ):
  I_within(M, v)   trung bình Agr giữa (g,e) và (g,e') của cùng phương pháp M
  I_gnn(M, v)      trung bình Agr giữa (g,e) và (g',e), chỉ khi hai GNN dự đoán cùng lớp
  D_between(A,B,v) trung bình Agr giữa A_{g,e} và B_{g,e}
  D_between_avg    Agr giữa thứ hạng trung bình qua seed explainer của A và của B
  Delta_A(v) = I_within(A,v) - D_between(A,B,v) (Mục 7.4)

Ví dụ:
  python scripts/analyze_variance.py --eval-root results/eval --out results/analysis/a2
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sna.core.provenance import execution_metadata, sha256, write_json
from sna.evaluation.stats import bootstrap_ci, compare_many, compare_paired, summarize

METHODS = ["saliency", "gnnexplainer", "pgexplainer"]
LABEL = {"saliency": "Saliency", "gnnexplainer": "GNNExplainer", "pgexplainer": "PGExplainer"}
DATASETS = ["bashapes", "cora"]
SEED = 0
FIXED_GNN = 1000


def per_node(frame, value="kappa"):
    """Trung bình theo node (bỏ NaN); trả Series chỉ số node_id."""
    return frame.groupby("node_id")[value].mean()


def load(eval_root, dataset):
    folder = eval_root / f"{dataset}_test"
    agreement = pd.read_csv(folder / "agreement.csv")
    change = pd.read_csv(folder / "pred_change.csv")
    metrics = pd.read_csv(folder / "metrics.csv")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return agreement, change, metrics, manifest, folder


def build_components(agreement, budget, value):
    """Tính mọi thành phần theo node cho một dataset. Trả dict các Series."""
    a = agreement[agreement.budget == budget]
    out = {"within_g": {}, "within_all": {}, "cross": {}, "cross_diff": {}, "between": {}, "between_avg": {},
           "ref_nf": {}}
    within = a[a.kind == "within"]
    for m in METHODS:
        part = within[within.method_a == m]
        out["within_all"][m] = per_node(part, value)
        out["within_g"][m] = per_node(part[part.g_a == FIXED_GNN], value)
        out["cross"][m] = per_node(a[(a.kind == "cross_gnn") & (a.method_a == m) & a.pred_same], value)
        out["cross_diff"][m] = per_node(
            a[(a.kind == "cross_gnn_diff_e") & (a.method_a == m) & a.pred_same], value)
        out["ref_nf"][m] = per_node(a[(a.kind == "ref_nearest_first") & (a.method_a == m)], value)
    for x, y in combinations(METHODS, 2):
        pair = a[(a.method_a == x) & (a.method_b == y)]
        out["between"][(x, y)] = per_node(pair[pair.kind == "between"], value)
        out["between_avg"][(x, y)] = per_node(pair[pair.kind == "between_avg"], value)
    out["ref_random"] = per_node(a[a.kind == "ref_random"], value)
    return out


def fmt_ci(values, seed=SEED):
    r = bootstrap_ci(np.asarray(values, dtype=float), seed=seed)
    return r["estimate"], r["ci_low"], r["ci_high"], r["n"]


def table_a2_1(dataset, comp, change, budget, value):
    rows = []
    rate = change.pred_change_rate.mean() if len(change) else np.nan
    for m in METHODS:
        row = {"dataset": dataset, "method": m, "budget": budget, "metric": value}
        for key, series in (("I_within_g1000", comp["within_g"][m]), ("I_within_5gnn", comp["within_all"][m]),
                            ("I_gnn", comp["cross"][m]), ("I_gnn_diff_e", comp["cross_diff"][m]),
                            ("ref_nearest_first", comp["ref_nf"][m])):
            est, lo, hi, n = fmt_ci(series.dropna())
            row.update({key: est, key + "_lo": lo, key + "_hi": hi, key + "_n": n,
                        key + "_sd": summarize(series.dropna())["sd"]})
        row["pred_change_rate_mean_over_gnn_pairs"] = rate
        row["pred_change_rate_max_over_gnn_pairs"] = change.pred_change_rate.max() if len(change) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def table_a2_2(dataset, comp, budget, value):
    """Mỗi cặp: D_between, D_between_avg, và Δ so với I_within của từng phương pháp.

    Họ kiểm định để hiệu chỉnh Holm: mọi phép so sánh (cặp × {Δ_A, Δ_B, Δ_min}) trong cùng dataset.
    """
    cases = {}
    meta = []
    for x, y in combinations(METHODS, 2):
        d = comp["between"][(x, y)]
        d_avg = comp["between_avg"][(x, y)]
        for who, m in (("A", x), ("B", y)):
            cases[f"{x}-{y}|within({m}) vs between"] = (comp["within_all"][m], d)
            meta.append((x, y, f"Delta_{who}", m))
        # Δ so với min(I_within(A), I_within(B)) tính theo node
        lo = pd.concat([comp["within_all"][x], comp["within_all"][y]], axis=1).min(axis=1)
        cases[f"{x}-{y}|min within vs between"] = (lo, d)
        meta.append((x, y, "Delta_min", "min"))
        cases[f"{x}-{y}|between_avg vs between"] = (d_avg, d)
        meta.append((x, y, "avg_minus_between", "avg"))
    # căn thẳng theo chỉ số node chung trước khi so sánh cặp
    aligned = {}
    for name, (s1, s2) in cases.items():
        idx = s1.index.intersection(s2.index)
        aligned[name] = (s1.loc[idx].to_numpy(float), s2.loc[idx].to_numpy(float))
    res = compare_many(aligned, seed=SEED)
    rows = []
    for (x, y, label, m), (_, r) in zip(meta, res.iterrows()):
        d = comp["between"][(x, y)]
        d_avg = comp["between_avg"][(x, y)]
        row = {"dataset": dataset, "pair": f"{x} vs {y}", "comparison": label, "reference": m,
               "budget": budget, "metric": value,
               "D_between": d.mean(), "D_between_avg": d_avg.mean(),
               "n": int(r["n"]), "mean_a": r["mean_a"], "mean_b": r["mean_b"],
               "mean_diff": r["mean_diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
               "p_raw": r["p_raw"], "p_holm": r["p_holm"], "effect_median_diff": r["effect"]}
        rows.append(row)
    return pd.DataFrame(rows)


def table_a2_3(dataset, metrics, budget):
    """SD của Fid-@5 theo seed explainer trong từng node, so với chênh lệch giữa các phương pháp."""
    m = metrics[(metrics.kind == "explainer") & (metrics.budget == budget) & (metrics.status == "success")]
    rows = []
    sd_node = {}
    for meth in METHODS:
        part = m[m.method == meth]
        # SD (ddof=1) qua explainer_seed, trong từng (g, node); rồi trung bình theo node qua g
        sd = part.groupby(["g", "node_id"]).fid_drop.std(ddof=1)
        sd_node[meth] = sd.groupby("node_id").mean()
        rows.append({"dataset": dataset, "item": f"mean_within_node_sd_{meth}",
                     "value": sd_node[meth].mean(), "n_nodes": int(sd_node[meth].notna().sum())})
    mean_fid = {meth: m[m.method == meth].groupby("node_id").fid_drop.mean() for meth in METHODS}
    for x, y in combinations(METHODS, 2):
        diff = (mean_fid[x] - mean_fid[y]).abs()
        rows.append({"dataset": dataset, "item": f"mean_abs_method_diff_{x}_vs_{y}", "value": diff.mean(),
                     "n_nodes": int(diff.notna().sum())})
        for who in (x, y):
            idx = sd_node[who].index.intersection(diff.index)
            share = (sd_node[who].loc[idx] > diff.loc[idx]).mean()
            rows.append({"dataset": dataset, "item": f"share_nodes_sd_{who}_gt_diff_{x}_vs_{y}",
                         "value": share, "n_nodes": len(idx)})
    return pd.DataFrame(rows)


def figure_a2(dataset, comp, out_path, budget):
    labels = [LABEL[m] for m in METHODS]
    groups = ("within (5 GNN)", "cross-GNN, cùng e", "cross-GNN, khác e", "between")
    cols = ("#4c78a8", "#f58518", "#e8b34a", "#54a24b")
    est = np.zeros((len(METHODS), 4))
    err = np.zeros((2, len(METHODS), 4))
    for i, m in enumerate(METHODS):
        others = [(x, y) for x, y in combinations(METHODS, 2) if m in (x, y)]
        btw = pd.concat([comp["between"][p] for p in others], axis=1).mean(axis=1)
        for j, series in enumerate((comp["within_all"][m], comp["cross"][m], comp["cross_diff"][m], btw)):
            e, lo, hi, _ = fmt_ci(series.dropna())
            est[i, j] = e
            err[0, i, j], err[1, i, j] = e - lo, hi - e
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    width = 0.2
    x = np.arange(len(METHODS))
    for j, (g, c) in enumerate(zip(groups, cols)):
        ax.bar(x + (j - 1.5) * width, est[:, j], width, yerr=err[:, :, j], capsize=3, label=g, color=c)
    rnd = comp["ref_random"].dropna().mean()
    ax.axhline(rnd, color="gray", ls="--", lw=1, label=f"Random vs Random (κ={rnd:.2f})")
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"κ@{budget} (trung bình theo node, CI 95%)")
    ax.set_title(f"Hình A2 — {dataset}: within / cross-GNN / between")
    ax.set_ylim(min(-0.1, est.min() - 0.1), 1.05)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", type=Path, default=Path("results/eval"))
    p.add_argument("--out", type=Path, default=Path("results/analysis/a2"))
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--metric", default="kappa", choices=["kappa", "jaccard"])
    p.add_argument("--figures", type=Path, default=Path("docs/figures/final"))
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    t1, t2, t3, inputs = [], [], [], {}
    for ds in args.datasets:
        agreement, change, metrics, manifest, folder = load(args.eval_root, ds)
        if manifest.get("status") not in ("completed", "completed_with_errors"):
            raise SystemExit(f"{ds}: manifest.status={manifest.get('status')!r}, cần chạy xong evaluate.py")
        for name in ("agreement.csv", "pred_change.csv", "metrics.csv", "manifest.json"):
            inputs[f"{ds}/{name}"] = sha256(folder / name)
        comp = build_components(agreement, args.budget, args.metric)
        t1.append(table_a2_1(ds, comp, change, args.budget, args.metric))
        t2.append(table_a2_2(ds, comp, args.budget, args.metric))
        t3.append(table_a2_3(ds, metrics, args.budget))
        figure_a2(ds, comp, args.figures / f"a2_{ds}.png", args.budget)

    outputs = {}
    for name, frames in (("a2_1_within_cross_gnn", t1), ("a2_2_between", t2), ("a2_3_fid_sd_vs_method_diff", t3)):
        path = args.out / f"{name}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, na_rep="NaN", lineterminator="\n")
        outputs[path.name] = sha256(path)
    for ds in args.datasets:
        f = args.figures / f"a2_{ds}.png"
        outputs[f"figures/{f.name}"] = sha256(f)
    write_json(args.out / "manifest.json", {
        **execution_metadata(), "argv": sys.argv, "budget": args.budget, "metric": args.metric,
        "bootstrap": {"unit": "node", "repeats": 2000, "confidence": 0.95, "seed": SEED},
        "paired_test": "wilcoxon_signed_rank (zero_method=wilcox)", "adjustment": "holm (theo dataset)",
        "inputs": inputs, "outputs": outputs})
    print(f"Wrote {args.out} ({', '.join(args.datasets)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
