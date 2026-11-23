"""Generate the BA-Shapes FB-3 tables and case studies from verified edge scores."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from sna.core.provenance import execution_metadata, protocol, sha256, verify_protocol_data, write_json
from sna.data import extract_candidate_edges
from sna.evaluation.core import baseline_scores, fidelity_selected, ground_truth_metrics, rank_edges
from sna.evaluation.runner import read_scores, seed_for
from sna.models import GCN
from train_gnn import load_run_model


METHODS = ("saliency", "gnnexplainer", "pgexplainer", "nearest_first")
KEY = ["gnn_seed", "node_id", "checkpoint_sha256"]
MAIN = ["fid_drop", "roc_auc"]


def load_inputs(args, cfg):
    methods = ("saliency", "pgexplainer") if args.saliency_scores else METHODS[:3]
    if args.midterm:
        root = args.midterm
        data_path = root / "data.pt"
        score_paths = [args.scores or Path("notebooks/01_midterm/edge_scores.csv")]
        archived = json.loads(Path("results/gate0/verification.json").read_text())
        for name in ("data.pt", "gcn_ckpt.pt", "val_nodes.json"):
            if sha256(root / name) != archived["assets_sha256"][f"bashapes/{name}"]:
                raise ValueError(f"Unexpected T06 asset: {root / name}")
        if sha256(score_paths[0]) != archived["source_scores_sha256"]:
            raise ValueError("T05 score hash differs from T06")
        raw = read_scores(score_paths[0]).query("dataset == 'bashapes'").assign(
            gnn_seed=0, explainer_seed=42, checkpoint_sha256=sha256(root / "gcn_ckpt.pt"))
        nodes_path = root / "val_nodes.json"
        nodes = json.loads(nodes_path.read_text(encoding="utf-8-sig"))["nodes"][:20]
        seeds, explainer_seeds = [0], [42]
        checkpoints = {0: root / "gcn_ckpt.pt"}
    else:
        data_path = args.data
        data_digest = verify_protocol_data("bashapes", data_path)
        nodes_path = args.nodes
        node_info = json.loads(nodes_path.read_text(encoding="utf-8-sig"))
        nodes = node_info["nodes"]
        node_bytes = nodes_path.read_bytes().replace(b"\r\n", b"\n")
        node_hashes = {hashlib.sha256(blob).hexdigest()
                       for blob in (node_bytes, node_bytes.replace(b"\n", b"\r\n"))}
        if node_info["data_sha256"] != data_digest:
            raise ValueError("Test node list has a different data hash")
        selection_path = args.checkpoints / "selection.json"
        seeds = json.loads(selection_path.read_text(encoding="utf-8"))["selected_seeds"]
        explainer_seeds = cfg["explainer_seeds"]
        expected = {(g, m, e) for g in seeds for m in methods
                    for e in (([0, 1] if g == cfg["fixed_gnn_seed"] else [0])
                              if m == "saliency" else explainer_seeds)}
        score_root = args.scores or Path("results/scores/bashapes")
        checkpoints = {g: args.checkpoints / f"gcn_s{g}.pt" for g in seeds}
        missing_checkpoints = [path.name for path in checkpoints.values() if not path.is_file()]
        if missing_checkpoints or not score_root.is_dir():
            raise FileNotFoundError(
                f"BA-Shapes handoff incomplete. Missing H1 checkpoints: {missing_checkpoints}. "
                f"H2 directory {score_root} exists: {score_root.is_dir()}.")
        score_paths, frames, logs = [], [], []
        for g, method, e in sorted(expected):
            root = args.saliency_scores if method == "saliency" and args.saliency_scores else score_root
            folder = root / f"gcn_s{g}"
            path = folder / f"{method}_e{e}.csv"
            log_path = folder / "run_log.json"
            log = json.loads(log_path.read_text(encoding="utf-8"))
            run = f"bashapes_gcn_g{g}_{method}_e{e}_test"
            frame = pd.read_csv(path, float_precision="round_trip")
            required = {"run_id": run, "dataset": "bashapes", "split": "test", "arch": "gcn",
                        "gnn_seed": g, "method": method, "explainer_seed": e, "seed": e,
                        "status": "success", "checkpoint_sha256": log["checkpoint_sha256"]}
            if any(set(frame[key]) != {value} for key, value in required.items()):
                raise ValueError(f"Invalid identity/status in {path}")
            if (log["data_sha256"] != data_digest or log["node_list_sha256"] not in node_hashes
                    or log["node_ids"] != nodes
                    or log["runs_by_id"][run]["csv_sha256"] != sha256(path)):
                raise ValueError(f"Score provenance mismatch in {path}")
            if set(frame.node_id) != set(nodes) or not np.isfinite(frame.importance_score).all():
                raise ValueError(f"Missing nodes or invalid scores in {path}")
            score_paths.append(path)
            logs.append(log_path)
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True)
        score_paths.extend(sorted(set(logs)) + [selection_path])
    data = torch.load(data_path, weights_only=True, map_location="cpu")
    split = "val" if args.midterm else "test"
    if len(set(nodes)) != len(nodes) or not data[f"{split}_mask"][nodes].all():
        raise ValueError(f"Invalid {split} node list")
    if not args.midterm and set(nodes) != set(data["test_mask"].nonzero().flatten().tolist()):
        raise ValueError("FB-3 requires every BA-Shapes test node")
    inputs = [data_path, nodes_path, *score_paths, *checkpoints.values()]
    if not args.midterm:
        inputs.extend(path.with_suffix(".json") for path in checkpoints.values())
    return data, data_path, nodes, seeds, explainer_seeds, checkpoints, raw, inputs, (*methods, "nearest_first")


def historical_interventions(model, data, node, target, pairs, motif, original):
    x, edges = data["x"], data["edge_index"]
    before = torch.bincount(edges[0], minlength=len(x)).to(x.dtype)
    if not torch.equal(x[:, 0], before) or not torch.allclose(x[:, 1], before.sqrt()):
        raise ValueError("Historical T3 expects degree and square root in columns 0 and 1")
    canonical = [tuple(sorted(pair)) for pair in edges.T.tolist()]
    rows = []
    for name, removed in (("motif_2hop", set(pairs) & motif),
                          ("one_hop", {pair for pair in pairs if node in pair})):
        retained = edges[:, torch.tensor([pair not in removed for pair in canonical])]
        after = torch.bincount(retained[0], minlength=len(x)).to(x.dtype)
        refreshed = x.clone()
        refreshed[:, 0] = after
        refreshed[:, 1] = after.sqrt()
        with torch.no_grad():
            fixed = float(model(x, retained)[node].softmax(-1)[target])
            changed = float(model(refreshed, retained)[node].softmax(-1)[target])
        rows.append(dict(node_id=node, intervention=name, target_class=target,
                         removed_pairs=len(removed), fid_drop_fixed=original - fixed,
                         fid_drop_refreshed=original - changed,
                         delta_refreshed_minus_fixed=fixed - changed))
    return rows


def compute(args, cfg, loaded):
    data, data_path, nodes, seeds, _, checkpoints, raw, _, methods = loaded
    edges, x, y = data["edge_index"], data["x"], data["y"]
    degree = torch.bincount(edges[0], minlength=len(x))
    motif = {tuple(sorted(edge)) for edge, hit in zip(edges.T.tolist(), data["gt_edge_mask"]) if hit}
    groups = raw.groupby(["gnn_seed", "node_id", "method", "explainer_seed"])
    candidates = {node: sorted(extract_candidate_edges(edges, node, num_nodes=len(x)).candidate_pairs)
                  for node in nodes}
    rows, selected_edges, curves, historical_t2_t3 = [], [], [], []
    for g in seeds:
        path = checkpoints[g]
        digest = sha256(path)
        if args.midterm:
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                ck = torch.load(path, weights_only=True, map_location="cpu")
            model = GCN(ck["in_dim"], ck["hidden"], ck["n_classes"], ck.get("dropout", 0.0))
            model.load_state_dict(ck["state_dict"])
            prediction = torch.as_tensor(ck["pred"])
        else:
            meta = json.loads(path.with_suffix(".json").read_text())
            if digest != meta["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            model, prediction, _ = load_run_model(path, data, data_path)
        with torch.no_grad():
            probabilities = model.eval()(x, edges).softmax(-1)
        if not torch.equal(probabilities.argmax(-1), prediction):
            raise ValueError(f"Prediction mismatch: {path}")
        for node in nodes:
            target = int(prediction[node])
            original = float(probabilities[node, target])
            pairs = candidates[node]
            membership = np.array([pair in motif for pair in pairs])
            local = np.array([node in pair for pair in pairs])
            if args.midterm and node == cfg["error_analysis"]["historical_node"]:
                historical_t2_t3 = historical_interventions(model, data, node, target, pairs, motif, original)
            ranks = {}
            for method in methods[:-1]:
                e_list = ([42] if args.midterm else cfg["explainer_seeds"])
                if not args.midterm and method == "saliency":
                    e_list = [0, 1] if g == cfg["fixed_gnn_seed"] else [0]
                for e in e_list:
                    frame = groups.get_group((g, node, method, e)).sort_values(["u", "v"])
                    if (list(zip(frame.u, frame.v)) != pairs or frame.edge_id.duplicated().any()
                            or set(frame.target_class) != {target}
                            or set(frame.checkpoint_sha256) != {digest}):
                        raise ValueError(f"Wrong candidates, target or checkpoint: {g}/{node}/{method}/{e}")
                    ids = tuple(frame.edge_id)
                    if ranks and ids != next(iter(ranks.values()))[1]:
                        raise ValueError(f"Edge ID mappings differ at {g}/{node}")
                    ranks[method, e] = (frame.importance_score.to_numpy(), ids)
            if ("saliency", 1) in ranks and not np.array_equal(ranks["saliency", 0][0], ranks["saliency", 1][0]):
                raise ValueError(f"Saliency differs across seeds at {g}/{node}")
            ranks["nearest_first", 0] = (baseline_scores(pairs, node, "nearest_first"), None)
            cache = {}

            def evaluate(chosen):
                key = tuple(sorted(chosen))
                if key not in cache:
                    cache[key] = fidelity_selected(model, data, node, target, pairs, key, original)
                return cache[key]

            for (method, e), (score, _) in ranks.items():
                if method == "saliency" and e == 1:
                    continue
                ordered = np.sort(score)[::-1]
                gap = float(ordered[4] - ordered[5]) if len(score) >= 6 else np.nan
                counts = pd.Series(score).value_counts()
                gt = ground_truth_metrics(pairs, score, edges, data["gt_edge_mask"], [])
                base = dict(dataset="bashapes", gnn_seed=g, node_id=node, checkpoint_sha256=digest,
                            method=method, explainer_seed=e, target_class=target, true_label=int(y[node]))
                for tie in cfg["tie_sensitivity_seeds"]:
                    order = rank_edges(score, seed_for("bashapes", node, method, tie))
                    chosen = order[:5]
                    direct = int(local[chosen].sum())
                    fidelity = evaluate(chosen)
                    rows.append({**base, "tie_seed": tie, **fidelity, "roc_auc": gt["roc_auc"],
                                 "average_precision": gt["average_precision"], "auc_reason": gt["auc_reason"],
                                 "precision_at_5": membership[chosen].sum() / 5,
                                 "one_hop_count": direct, "one_hop_fraction": direct / len(chosen),
                                 "tied_fraction": counts[counts > 1].sum() / len(score),
                                 "gap_5_6": gap, "candidates": len(score), "motif_edges": gt["gt_positives"],
                                 "degree_before": int(degree[node]), "degree_after": int(degree[node]) - direct})
                    for rank, i in enumerate(chosen, 1):
                        u, v = pairs[i]
                        edge = {**base, "tie_seed": tie, "rank": rank, "u": u, "v": v,
                                "score": score[i], "motif": bool(membership[i]), "one_hop": bool(local[i]),
                                "degree_u": int(degree[u]), "degree_v": int(degree[v]),
                                "label_u": int(y[u]), "label_v": int(y[v])}
                        if fidelity["fid_drop"] < 0 or (tie == 0 and e in (0, 42)):
                            selected_edges.append(edge)
                    if tie == 0 and e in (0, 42):
                        for k in range(len(chosen) + 1):
                            probability = original if k == 0 else evaluate(chosen[:k])["delete_prob"]
                            curves.append({**base, "k": k, "probability": probability})
    return pd.DataFrame(rows), pd.DataFrame(selected_edges), pd.DataFrame(curves), candidates, motif, pd.DataFrame(historical_t2_t3)


def winners(part):
    result = {}
    for metric in MAIN:
        best = part[metric].max()
        result[metric] = set(part.loc[np.isclose(part[metric], best, rtol=0, atol=1e-12), "method"])
    fid, auc = (result[metric] for metric in MAIN)
    return dict(fid_winners="|".join(sorted(fid)), auc_winners="|".join(sorted(auc)),
                conflict=None if not fid or not auc else not bool(fid & auc))


def configuration_tables(rows, seeds, explainer_seeds, cfg, methods=METHODS):
    stats, comparisons = [], []
    rng = np.random.default_rng(0)
    for g, e in product(seeds, explainer_seeds):
        part = rows[(rows.gnn_seed == g) &
                    ((rows.explainer_seed == e) | rows.method.isin(["saliency", "nearest_first"]))]
        by_node = part.groupby(["method", "node_id"], as_index=False)[MAIN].mean()
        for (method, metric) in product(methods, MAIN):
            values = by_node.loc[by_node.method == method, metric].dropna().to_numpy()
            lo = hi = np.nan
            if len(values):
                draws = rng.choice(values, (cfg["statistics"]["bootstrap"]["repeats"], len(values)))
                lo, hi = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            stats.append(dict(gnn_seed=g, explainer_seed=e, method=method, metric=metric,
                              mean=values.mean() if len(values) else np.nan, ci_low=lo, ci_high=hi,
                              n_valid=len(values), n_total=part.node_id.nunique()))
        primary = by_node.groupby("method", as_index=False)[MAIN].mean()
        comparisons.append(dict(gnn_seed=g, explainer_seed=e, tie_seed="average", **winners(primary)))
        for tie, sensitivity in part.groupby("tie_seed"):
            # The baseline is averaged over its five repeats within each node.
            means = sensitivity[sensitivity.method != "nearest_first"].groupby("method")[MAIN].mean()
            means.loc["nearest_first"] = primary.set_index("method").loc["nearest_first", MAIN]
            comparisons.append(dict(gnn_seed=g, explainer_seed=e, tie_seed=tie,
                                    **winners(means.reset_index())))
    return pd.DataFrame(stats), pd.DataFrame(comparisons)


def node_tables(rows, cfg):
    node = rows.groupby(KEY + ["method", "true_label", "target_class"], as_index=False).agg(
        fid_drop=("fid_drop", "mean"), fid_inv=("fid_inv", "mean"), roc_auc=("roc_auc", "mean"),
        precision_at_5=("precision_at_5", "mean"), one_hop_fraction=("one_hop_fraction", "mean"),
        tied_fraction=("tied_fraction", "mean"), gap_5_6=("gap_5_6", "median"))
    conflicts = pd.DataFrame([{**dict(zip(KEY, key)), **winners(part)}
                              for key, part in node.groupby(KEY)])
    node = node.merge(conflicts, on=KEY, validate="many_to_one")
    quadrant_rows = []
    for fid, auc in cfg["error_analysis"]["quadrant_thresholds"]:
        quadrant = np.select(
            [node.roc_auc.isna(), (node.fid_drop >= fid) & (node.roc_auc >= auc),
             node.fid_drop >= fid, node.roc_auc >= auc],
            ["undefined", "Q1", "Q2", "Q3"], default="Q4")
        quadrant_rows.append(node.assign(fid_threshold=fid, auc_threshold=auc, quadrant=quadrant))
    quadrants = pd.concat(quadrant_rows, ignore_index=True)
    count = quadrants.groupby(["fid_threshold", "auc_threshold", "gnn_seed", "method", "quadrant"],
                               as_index=False).size()
    return node, conflicts, quadrants, count


def correlation(a, b):
    sample = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(sample) < 2 or min(sample.nunique()) < 2:
        return np.nan, len(sample)
    return float(spearmanr(sample.a, sample.b).statistic), len(sample)


def mechanisms(rows, node, conflicts, partner):
    repeated = rows.groupby(KEY + ["method", "explainer_seed"], as_index=False).agg(
        gap=("gap_5_6", "first"), fidelity_sd=("fid_drop", "std"),
        fidelity_min=("fid_drop", "min"), fidelity_max=("fid_drop", "max"))
    repeated["fidelity_range"] = repeated.fidelity_max - repeated.fidelity_min
    repeated = repeated.merge(conflicts[KEY + ["conflict"]], on=KEY, validate="many_to_one")
    audit = rows.assign(negative=rows.fid_drop < 0).merge(
        conflicts[KEY + ["conflict"]], on=KEY, validate="many_to_one")
    result = []
    for (g, method), part in node.groupby(["gnn_seed", "method"]):
        tie = repeated[(repeated.gnn_seed == g) & (repeated.method == method)]
        runs = audit[(audit.gnn_seed == g) & (audit.method == method)]
        for cohort in ("all", "conflict"):
            view = part if cohort == "all" else part[part.conflict == True]
            tie_view = tie[tie.node_id.isin(view.node_id)]
            run_view = runs[runs.node_id.isin(view.node_id)]
            rho, n = correlation(view.one_hop_fraction, view.fid_drop)
            motif_rho, motif_n = correlation(view.roc_auc, view.fid_drop)
            t4 = view[view.roc_auc >= .75]
            result.append(dict(gnn_seed=g, method=method, cohort=cohort, n_nodes=len(view),
                               t1_mean_one_hop=view.one_hop_fraction.mean(), t1_spearman=rho, t1_n=n,
                               t1_high_drop_majority_one_hop=int(
                                   ((view.fid_drop >= .10) & (view.one_hop_fraction > .5)).sum()),
                               motif_fid_spearman=motif_rho, motif_fid_n=motif_n,
                               t4_high_auc_low_precision=int((t4.precision_at_5 < .5).sum()),
                               t4_high_auc_n=len(t4),
                               t4_zero_gap_nodes=tie_view.loc[tie_view.gap == 0, "node_id"].nunique(),
                               t4_gap_eligible_nodes=tie_view.loc[tie_view.gap.notna(), "node_id"].nunique(),
                               t4_mean_fid_sd=tie_view.fidelity_sd.mean(),
                               t4_max_fid_range=tie_view.fidelity_range.max(),
                               t5_negative_nodes=run_view.loc[run_view.negative, "node_id"].nunique(),
                               t5_negative_runs=int(run_view.negative.sum()), t5_total_runs=len(run_view)))
    if partner is not None:
        partner = partner.merge(conflicts[KEY + ["conflict"]], on=KEY, validate="many_to_one")
    return pd.DataFrame(result), repeated, audit[audit.negative], partner


def choose_cases(quadrants, fixed_seed):
    main = quadrants[(quadrants.fid_threshold == .10) & (quadrants.auc_threshold == .75) &
                     (quadrants.gnn_seed == fixed_seed) & (quadrants.method == "pgexplainer") &
                     (quadrants.quadrant != "undefined")].copy()
    for metric in MAIN:
        main[metric + "_distance"] = main[metric] - main.groupby("quadrant")[metric].transform("median")
    main["distance"] = np.hypot(main.fid_drop_distance, main.roc_auc_distance)
    main = main.sort_values(["distance", "node_id"])
    groups = {q: list(part.itertuples()) for q, part in main.groupby("quadrant")}
    # At most two per quadrant and seven overall, reserving an eighth for node 692.
    selected = [part[0] for part in groups.values()]
    covered = {row.true_label for row in selected}
    for label in sorted({1, 2, 3} - covered):
        available = [row for part in groups.values() for row in part if row.true_label == label
                     and sum(old.quadrant == row.quadrant for old in selected) < 2]
        if available:
            selected.append(min(available, key=lambda row: (row.distance, row.node_id)))
    for part in groups.values():
        if len(selected) < 7 and len(part) > 1 and sum(
                old.quadrant == part[0].quadrant for old in selected) < 2:
            selected.append(part[1])
    return [row.node_id for row in selected]


def case_plot(node_id, pairs, motif, top, curve, output):
    graph = nx.Graph()
    graph.add_node(node_id)
    graph.add_edges_from(pairs)
    layout = nx.spring_layout(graph, seed=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    palette = dict(zip(METHODS, ("#2563eb", "#7c3aed", "#00816c", "#97520c")))
    for ax, method in zip(axes.flat, METHODS):
        edges = top[top.method == method]
        chosen = list(zip(edges.u, edges.v))
        nx.draw_networkx_edges(graph, layout, ax=ax, edge_color="#cbd5e1", width=.5, alpha=.65)
        motif_edges = [pair for pair in pairs if pair in motif]
        nx.draw_networkx_edges(graph, layout, edgelist=motif_edges, ax=ax,
                               edge_color="#dc2626", width=1.6)
        nx.draw_networkx_nodes(graph, layout, ax=ax, node_size=11, node_color="#64748b")
        nx.draw_networkx_nodes(graph, layout, nodelist=[node_id], ax=ax, node_size=90,
                               node_color="#0f172a")
        for pair in chosen:
            nx.draw_networkx_edges(graph, layout, edgelist=[pair], ax=ax,
                                   edge_color="#dc2626" if pair in motif else palette[method], width=3)
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=dict(zip(chosen, edges["rank"])),
                                     ax=ax, font_size=8, rotate=False)
        ax.set_title(method)
        ax.set_axis_off()
        series = curve[curve.method == method]
        axes[1, 1].plot(series.k, series.probability, marker="o", label=method, color=palette[method])
    axes[1, 1].set(xlabel="Deleted edges", ylabel="Original class probability", xticks=range(6))
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(.04, .75, "Red edges: ground-truth motif\nNumbered edges: selected top five\n"
                             "Large node: explanation target", transform=axes[1, 2].transAxes,
                    va="top")
    fig.suptitle(f"BA-Shapes | node {node_id}")
    fig.savefig(output, dpi=145)
    plt.close(fig)


def run(args):
    cfg = protocol()
    torch.set_num_threads(cfg["torch_threads"])
    loaded = load_inputs(args, cfg)
    _, _, nodes, seeds, explainer_seeds, _, _, inputs, methods = loaded
    partner = None
    if not args.midterm:
        partner_manifest = args.t2_t3.with_name("manifest.json")
        manifest = json.loads(partner_manifest.read_text(encoding="utf-8"))
        if (manifest["outputs"][args.t2_t3.name] != sha256(args.t2_t3)
                or manifest["data_sha256"] != sha256(args.data)
                or manifest["nodes_sha256"] != sha256(args.nodes)):
            raise ValueError("T2/T3 provenance mismatch")
        partner = pd.read_csv(args.t2_t3)
        inputs.extend([args.t2_t3, partner_manifest])
    input_hashes = {str(path): sha256(path) for path in inputs}
    rows, edges, curves, candidates, motif, historical_t2_t3 = compute(args, cfg, loaded)
    stats, comparisons = configuration_tables(rows, seeds, explainer_seeds, cfg, methods)
    node, conflicts, quadrants, counts = node_tables(rows, cfg)
    if partner is not None and set(map(tuple, partner[KEY].to_numpy())) != set(
            map(tuple, conflicts[KEY].to_numpy())):
        raise ValueError("T2/T3 checkpoint/node keys differ from A3")
    mechanisms_table, ties, negative, partner = mechanisms(rows, node, conflicts, partner)
    case_seed = seeds[0] if args.midterm else cfg["fixed_gnn_seed"]
    historical = cfg["error_analysis"]["historical_node"]
    cases = [historical] if args.midterm else [] if args.saliency_scores else choose_cases(quadrants, case_seed)
    if args.saliency_scores:
        print("Partial comparison: final case selection pending GNNExplainer.")
    elif not cases:
        print("No case candidates with defined motif AUROC.")
    case_edges = edges[(edges.gnn_seed == case_seed) & edges.node_id.isin(cases) &
                       (edges.tie_seed == 0) & edges.explainer_seed.isin([0, 42])]
    case_curves = curves[(curves.gnn_seed == case_seed) & curves.node_id.isin(cases)]
    t5_edges = edges.merge(negative[KEY + ["method", "explainer_seed", "tie_seed"]],
                           on=KEY + ["method", "explainer_seed", "tie_seed"], validate="many_to_one")
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    tables = {"a3_0_metrics": stats, "a3_0_conflicts": comparisons, "a3_1_counts": counts,
              "a3_1_by_node": quadrants, "a3_1_node_conflicts": conflicts, "metrics_by_run": rows,
              "a3_2_mechanisms": mechanisms_table, "t4_tie_sensitivity": ties,
              "t5_negative_fidelity": negative, "t5_deleted_edges": t5_edges,
              "case_top_edges": case_edges, "case_deletion_curves": case_curves,
              "case_metrics": node[(node.gnn_seed == case_seed) & node.node_id.isin(cases)]}
    if partner is not None:
        tables["t2_t3_conflict_cohort"] = partner
        tables["case_t2_t3"] = partner[(partner.gnn_seed == case_seed) & partner.node_id.isin(cases)]
        tables["a3_2_t2_t3"] = partner.assign(
            cohort=np.where(partner.conflict == True, "conflict", "other_or_undefined")
        ).groupby(["gnn_seed", "intervention", "cohort"], as_index=False).agg(
            n_nodes=("node_id", "size"), mean_drop=("fid_drop_fixed", "mean"),
            mean_refreshed_drop=("fid_drop_refreshed", "mean"),
            mean_delta=("delta_refreshed_minus_fixed", "mean"))
    else:
        tables["case_t2_t3"] = historical_t2_t3
    outputs = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, na_rep="NaN", lineterminator="\n")
        outputs[path.name] = sha256(path)
    for case in cases:
        path = output / f"a3_node_{case}.png"
        case_plot(case, candidates[case], motif, case_edges[case_edges.node_id == case],
                  case_curves[case_curves.node_id == case], path)
        outputs[path.name] = sha256(path)
    write_json(output / "manifest.json", {
        **execution_metadata(), "mode": ("midterm_val" if args.midterm else
                 "partial_test" if args.saliency_scores else "final_test"),
        "methods": methods,
        "nodes": nodes, "gnn_seeds": seeds, "explainer_seeds": explainer_seeds,
        "tie_seeds": cfg["tie_sensitivity_seeds"], "cases": cases, "case_seed": case_seed,
        "case_explainer_seed": 42 if args.midterm else 0, "inputs": input_hashes,
        "outputs": outputs, "historical_node": historical,
        "case_rule": "Fixed GNN, PGExplainer seed-averaged metrics, quadrant median distance, class coverage",
        "winner_rule": "Winning sets are disjoint; ties within absolute tolerance 1e-12",
        "t4_precision_threshold": .5,
    })
    print(f"Wrote {output}: {len(nodes)} nodes, {len(seeds) * len(explainer_seeds)} configurations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bashapes"], default="bashapes")
    parser.add_argument("--data", type=Path, default=Path("results/data/bashapes/data.pt"))
    parser.add_argument("--nodes", type=Path, default=Path("data/bashapes/test_nodes.json"))
    parser.add_argument("--checkpoints", type=Path, default=Path("results/checkpoints/bashapes"))
    parser.add_argument("--scores", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--midterm", type=Path, help="Verified BA-Shapes T06 data directory")
    mode.add_argument("--saliency-scores", type=Path,
                      help="Partial test comparison with Saliency in a separate H2 directory")
    parser.add_argument("--t2-t3", type=Path,
                        default=Path("results/analysis/a3_t2_t3/t2_t3_by_node.csv"))
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
