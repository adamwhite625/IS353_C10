"""Recompute every number used on the final-term slides from the result CSVs.
Writes numbers.json (consumed by build_deck.js) and prints a check table.
Run from the repo/worktree root:  python docs/slides_cuoi_ky/verify_numbers.py
"""
import json
import pandas as pd

R = "results/analysis/"
N = {}

# ---- main table (k=5)
m = pd.read_csv(R + "main/main_by_method_budget.csv")
m5 = m[m.budget == 5]
for ds in ["bashapes", "cora"]:
    for meth in ["saliency", "gnnexplainer", "pgexplainer", "nearest_first", "random"]:
        for met in ["fid_drop", "fid_inv", "roc_auc"]:
            r = m5[(m5.dataset == ds) & (m5.method == meth) & (m5.metric == met)]
            if len(r):
                r = r.iloc[0]
                N[f"main.{ds}.{meth}.{met}"] = [r["mean"], r.ci_low, r.ci_high]

# ---- A2-1
a1 = pd.read_csv(R + "a2/a2_1_within_cross_gnn.csv")
for _, r in a1.iterrows():
    k = f"a21.{r.dataset}.{r.method}"
    N[k] = dict(within=r.I_within_5gnn, gnn_same=r.I_gnn, gnn_diff=r.I_gnn_diff_e,
                within_lo=r.I_within_5gnn_lo, within_hi=r.I_within_5gnn_hi,
                pred_change=r.pred_change_rate_mean_over_gnn_pairs)

# ---- A2-2
a2 = pd.read_csv(R + "a2/a2_2_between.csv")
for _, r in a2[a2.comparison.isin(["Delta_A", "Delta_B", "Delta_min"])].iterrows():
    N[f"a22.{r.dataset}.{r.pair}.{r.comparison}"] = [r.mean_diff, r.ci_low, r.ci_high, r.p_holm]
    N[f"a22.{r.dataset}.{r.pair}.D"] = [r.D_between, r.D_between_avg]
avg = a2[a2.comparison == "avg_minus_between"]
for _, r in avg.iterrows():
    N[f"a22.{r.dataset}.{r.pair}.avg_minus"] = [r.mean_diff, r.ci_low, r.ci_high, r.p_holm]

# ---- A3-0
c = pd.read_csv(R + "a3_final/a3_0_conflicts.csv")
avgc = c[c.tie_seed.astype(str) == "average"]
N["a30.conflicts"] = [int(avgc.conflict.sum()), len(avgc)]
N["a30.fid_win"] = avgc.fid_winners.value_counts().to_dict()
N["a30.auc_win"] = avgc.auc_winners.value_counts().to_dict()
am = pd.read_csv(R + "a3_final/a3_0_metrics.csv")
g = am.groupby(["method", "metric"])["mean"].agg(["mean", "min", "max"])
for (meth, met), r in g.iterrows():
    N[f"a30.{meth}.{met}"] = [r["mean"], r["min"], r["max"]]

# ---- A3-1 quadrants (0.10, 0.75) and node conflicts
b = pd.read_csv(R + "a3_final/a3_1_by_node.csv")
b = b[(b.fid_threshold == .10) & (b.auc_threshold == .75)]
b = b[b.roc_auc.notna()]
q = b.groupby(["method", "quadrant"]).size().unstack(fill_value=0)
N["a31.pairs"] = int(b.groupby(["gnn_seed", "node_id"]).ngroups)
N["a31.q"] = {k: v for k, v in q.to_dict("index").items()}
nc = pd.read_csv(R + "a3_final/a3_1_node_conflicts.csv")
ncd = nc[nc.conflict.notna() & nc.auc_winners.notna()]
N["a31.node_conflict"] = [int((ncd.conflict == True).sum()), len(ncd)]

# ---- T2 / T3 / T1
t = pd.read_csv(R + "a3_final/t2_t3_conflict_cohort.csv")
mm = t[t.intervention == "motif_2hop"].merge(
    nc[["gnn_seed", "node_id", "auc_winners"]], on=["gnn_seed", "node_id"], how="left")
mm["defined"] = mm.auc_winners.notna()
conf = mm[mm.conflict == True]
N["t2.conflict"] = [int((conf.fid_drop_fixed < .01).sum()), len(conf)]
oth = mm[(mm.conflict == False) & mm.defined]
N["t2.other"] = [int((oth.fid_drop_fixed < .01).sum()), len(oth)]
N["t2.mean_drop"] = [conf.fid_drop_fixed.mean(), oth.fid_drop_fixed.mean()]
# T1: mean 1-hop fraction of top-5 of Fid-winners vs AUC-winners among conflict pairs
bb = pd.read_csv(R + "a3_final/a3_1_by_node.csv")
bb = bb[(bb.fid_threshold == .10) & (bb.auc_threshold == .75) & (bb.conflict == True)]
fw, aw = [], []
for (gs, nid), grp in bb.groupby(["gnn_seed", "node_id"]):
    fm = str(grp.fid_winners.iloc[0]).split("|")
    am_ = str(grp.auc_winners.iloc[0]).split("|")
    f = grp[grp.method.isin(fm)].one_hop_fraction.mean()
    a = grp[grp.method.isin(am_)].one_hop_fraction.mean()
    fw.append(f); aw.append(a)
N["t1.means"] = [sum(fw) / len(fw), sum(aw) / len(aw), len(fw)]
# T3 refreshed
t3 = pd.read_csv(R + "a3_final/t2_t3_conflict_cohort.csv")
cf = t3[t3.conflict == True]
N["t3.one_hop"] = [cf[cf.intervention == "one_hop"].fid_drop_fixed.mean(),
                   cf[cf.intervention == "one_hop"].fid_drop_refreshed.mean()]

# ---- D1
d = pd.read_csv(R + "d1/d1_metrics.csv")
d5 = d[d.budget == 5]
for _, r in d5.iterrows():
    N[f"d1.{r.dataset}.{r.method}.{r.metric}.{r.variant}"] = [r["mean"], r.ci_low, r.ci_high]
ds_ = pd.read_csv(R + "d1/d1_vs_single_seed.csv")
x = ds_[(ds_.budget == 5) & (ds_.reference == "single_mean")]
N["d1.sig"] = {
    ds: [int((x[x.dataset == ds].p_holm < .05).sum()), len(x[x.dataset == ds])]
    for ds in ["bashapes", "cora"]}
st = pd.read_csv(R + "d1/d1_stability.csv")
st = st[(st.budget == 5) & (st.scope == "5gnn")]
for _, r in st.iterrows():
    N[f"d1stab.{r.dataset}.{r.method}.{r.design}"] = r["mean"]

# ---- checkpoints
ck = pd.read_csv("results/checkpoints/checkpoint_table.csv")
N["ckpt"] = ck[["dataset", "gnn_seed", "checkpoint_sha256", "data_sha256", "test_acc"]].to_dict("records")

# ---- PG thread sensitivity
p = pd.read_csv(R + "a2/pg_thread_sensitivity.csv")
N["pgthread"] = dict(kappa_mean=p.kappa5_old_vs_new.mean(), kappa_min=p.kappa5_old_vs_new.min(),
                     auc_diff_mean=(p.auc_old - p.auc_new).abs().mean(),
                     auc_diff_max=(p.auc_old - p.auc_new).abs().max(), n=len(p))

# ---- cases
cm = pd.read_csv(R + "a3_final/case_metrics.csv")
cm = cm[cm.gnn_seed == 1000]
for nid in [665, 596, 342, 106]:
    for _, r in cm[cm.node_id == nid].iterrows():
        N[f"case.{nid}.{r.method}"] = [r.fid_drop, r.roc_auc]

# ---- T5 negative-Fidelity runs (PGExplainer) and node 665 curve
mech = pd.read_csv(R + "a3_final/a3_2_mechanisms.csv")
for coh in ["conflict", "all"]:
    pg = mech[(mech.method == "pgexplainer") & (mech.cohort == coh)]
    N[f"t5.pg.{coh}"] = [int(pg.t5_negative_runs.sum()), int(pg.t5_total_runs.sum())]
cv = pd.read_csv(R + "a3_final/case_deletion_curves.csv")
cv = cv[(cv.node_id == 665) & (cv.gnn_seed == 1000) & (cv.method == "saliency")].sort_values("k")
N["curve665.saliency"] = cv.probability.tolist()
N["delta_min"] = [float(r.mean_diff) for _, r in a2[a2.comparison == "Delta_min"].iterrows()]

# ---- B1: kappa@k between pairs and Random-Random
for ds in ["bashapes", "cora"]:
    ag = pd.read_csv(f"results/eval/{ds}_test/agreement_summary.csv")
    ag = ag[ag.metric == "kappa"]
    for _, r in ag[ag.kind == "between"].iterrows():
        N[f"b1.{ds}.{r.method_a}-{r.method_b}.{r.budget}"] = [r["mean"], int(r.n_valid_nodes)]
    for _, r in ag[ag.kind == "ref_random"].iterrows():
        N[f"b1.{ds}.random.{r.budget}"] = [r["mean"], int(r.n_valid_nodes)]

json.dump(N, open("docs/slides_cuoi_ky/numbers.json", "w"), indent=1, default=float)
for k in ["a30.conflicts", "a31.pairs", "a31.q", "a31.node_conflict", "t2.conflict", "t2.other",
          "t2.mean_drop", "t1.means", "t3.one_hop", "d1.sig", "pgthread", "a30.fid_win", "a30.auc_win"]:
    print(k, N[k])
