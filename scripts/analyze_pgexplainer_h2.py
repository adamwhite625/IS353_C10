"""Generate preliminary BA-Shapes PGExplainer results directly from its H2 ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from sna.core.provenance import execution_metadata, protocol, sha256, verify_protocol_data, write_json
from sna.data import extract_candidate_edges
from sna.evaluation.core import fidelity_selected, ground_truth_metrics, rank_edges
from sna.evaluation.runner import seed_for
from train_gnn import load_run_model


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def run(archive, output, with_fidelity=False):
    cfg = protocol()
    data_path = Path("results/data/bashapes/data.pt")
    nodes_path = Path("data/bashapes/test_nodes.json")
    data_hash = verify_protocol_data("bashapes", data_path)
    raw_nodes = nodes_path.read_bytes()
    lf = raw_nodes.replace(b"\r\n", b"\n")
    acceptable_node_hashes = {digest(lf), digest(lf.replace(b"\n", b"\r\n"))}
    nodes = json.loads(raw_nodes)["nodes"]
    torch.set_num_threads(cfg["torch_threads"])
    graph = torch.load(data_path, map_location="cpu", weights_only=True)
    edges, motif_mask = graph["edge_index"], graph["gt_edge_mask"]
    if set(nodes) != set(graph["test_mask"].nonzero().flatten().tolist()):
        raise ValueError("The node list differs from the BA-Shapes test split")
    motif = {tuple(sorted(pair)) for pair, hit in zip(edges.T.tolist(), motif_mask) if hit}
    candidates = {node: sorted(extract_candidate_edges(edges, node, num_nodes=len(graph["x"])).candidate_pairs)
                  for node in nodes}
    seeds = cfg["gnn_seeds"]
    explainer_seeds = cfg["explainer_seeds"]
    rows, source_hashes, checkpoints = [], {}, {}
    with ZipFile(archive) as bundle:
        checksums = dict(reversed(line.split(maxsplit=1))
                         for line in bundle.read("SHA256SUMS.txt").decode().splitlines() if line.strip())

        def read_verified(path):
            blob = bundle.read(path)
            if digest(blob) != checksums[path]:
                raise ValueError(f"ZIP checksum mismatch: {path}")
            source_hashes[path] = checksums[path]
            return blob

        for g in seeds:
            folder = f"results/scores/bashapes/gcn_s{g}/"
            log = json.loads(read_verified(folder + "run_log.json"))
            checkpoint = Path(f"results/checkpoints/bashapes/gcn_s{g}.json")
            checkpoint_hash = json.loads(checkpoint.read_text())["checkpoint_sha256"]
            if (log["data_sha256"] != data_hash or log["node_list_sha256"] not in acceptable_node_hashes
                    or log["node_ids"] != nodes or log["checkpoint_sha256"] != checkpoint_hash):
                raise ValueError(f"Run log does not match BA-Shapes inputs: {folder}")
            if with_fidelity:
                weights = checkpoint.with_suffix(".pt")
                if sha256(weights) != checkpoint_hash:
                    raise ValueError(f"Checkpoint file hash differs: {weights}")
                model, predictions, _ = load_run_model(weights, graph, data_path)
                with torch.no_grad():
                    original = model(graph["x"], edges).softmax(-1)
                checkpoints[weights.name] = checkpoint_hash
                cache = {}
            for e in explainer_seeds:
                run_id = f"bashapes_gcn_g{g}_pgexplainer_e{e}_test"
                path = folder + f"pgexplainer_e{e}.csv"
                blob = read_verified(path)
                if source_hashes[path] != log["runs_by_id"][run_id]["csv_sha256"]:
                    raise ValueError(f"Run log CSV checksum mismatch: {path}")
                frame = pd.read_csv(io.BytesIO(blob), float_precision="round_trip")
                identity = {"run_id": run_id, "dataset": "bashapes", "split": "test", "arch": "gcn",
                            "method": "pgexplainer", "status": "success", "gnn_seed": g,
                            "explainer_seed": e, "checkpoint_sha256": checkpoint_hash}
                if (any(set(frame[key]) != {value} for key, value in identity.items())
                        or set(frame.node_id) != set(nodes)):
                    raise ValueError(f"Run identity or nodes differ: {path}")
                for node, group in frame.groupby("node_id"):
                    group = group.sort_values(["u", "v"])
                    pairs = list(zip(group.u, group.v))
                    if (pairs != candidates[node] or group.edge_id.duplicated().any()
                            or group.target_class.nunique() != 1):
                        raise ValueError(f"Invalid candidate edges or target at {run_id}/{node}")
                    target = int(group.target_class.iloc[0])
                    if with_fidelity and target != int(predictions[node]):
                        raise ValueError(f"Target differs from checkpoint at {run_id}/{node}")
                    scores = group.importance_score.to_numpy()
                    gt = ground_truth_metrics(pairs, scores, edges, motif_mask, [])
                    hits = np.array([pair in motif for pair in pairs])
                    direct = np.array([node in pair for pair in pairs])
                    k = cfg["main_budget"]
                    cut = np.sort(scores)[::-1]
                    gap = cut[k - 1] - cut[k] if len(cut) > k else np.nan
                    counts = group.importance_score.value_counts()
                    for tie in cfg["tie_sensitivity_seeds"]:
                        order = rank_edges(scores, seed_for("bashapes", node, "pgexplainer", tie))[:k]
                        fidelity = {}
                        if with_fidelity:
                            key = (node, tuple(sorted(order)))
                            if key not in cache:
                                cache[key] = fidelity_selected(model, graph, node, target, pairs,
                                                               order, float(original[node, target]))
                            fidelity = cache[key]
                        rows.append(dict(gnn_seed=g, explainer_seed=e, node_id=node,
                                         checkpoint_sha256=checkpoint_hash, tie_seed=tie,
                                         roc_auc=gt["roc_auc"], average_precision=gt["average_precision"],
                                         auc_reason=gt["auc_reason"], precision_at_5=hits[order].sum() / k,
                                         one_hop_fraction=direct[order].mean(), gap_5_6=gap,
                                         tied_fraction=counts[counts > 1].sum() / len(scores),
                                         target_class=target,
                                         top_pairs=json.dumps([pairs[i] for i in order]), **fidelity))
    detail = pd.DataFrame(rows)
    keys = ["gnn_seed", "explainer_seed", "node_id"]
    per_node = detail.groupby(keys, as_index=False).agg(
        roc_auc=("roc_auc", "first"), average_precision=("average_precision", "first"),
        precision_at_5=("precision_at_5", "mean"), one_hop_fraction=("one_hop_fraction", "mean"),
        gap_5_6=("gap_5_6", "first"), tied_fraction=("tied_fraction", "first"),
        precision_range=("precision_at_5", lambda v: v.max() - v.min()))
    per_node["high_auc_low_precision"] = (per_node.roc_auc >= .75) & (per_node.precision_at_5 < .5)
    per_node["zero_gap"] = per_node.gap_5_6 == 0
    per_node["precision_changes"] = per_node.precision_range > 0
    summary = per_node.groupby(keys[:2], as_index=False).agg(
        n_nodes=("node_id", "size"), n_auc=("roc_auc", "count"), auc_mean=("roc_auc", "mean"),
        ap_mean=("average_precision", "mean"), precision_mean=("precision_at_5", "mean"),
        mean_one_hop_fraction=("one_hop_fraction", "mean"), zero_gap=("zero_gap", "sum"),
        gap_eligible=("gap_5_6", "count"), precision_changes=("precision_changes", "sum"),
        high_auc_low_precision=("high_auc_low_precision", "sum"))
    tables = {"by_tie_seed.csv": detail, "by_node.csv": per_node, "by_configuration.csv": summary}
    if with_fidelity:
        fidelity_nodes = detail.groupby(keys, as_index=False).agg(
            fid_drop=("fid_drop", "mean"), fid_inv=("fid_inv", "mean"),
            drop_min=("fid_drop", "min"), drop_max=("fid_drop", "max"),
            negative_trials=("fid_drop", lambda values: (values < 0).sum()))
        fidelity_nodes["drop_range"] = fidelity_nodes.drop_max - fidelity_nodes.drop_min
        fidelity_nodes["changes_over_1e_6"] = fidelity_nodes.drop_range > 1e-6
        per_node = per_node.merge(fidelity_nodes, on=keys, validate="one_to_one")
        comparison = []
        for (g, e), group in per_node.groupby(keys[:2]):
            x, y = group.one_hop_fraction, group.fid_drop
            rho = spearmanr(x, y).statistic if min(x.nunique(), y.nunique()) > 1 else np.nan
            comparison.append(dict(gnn_seed=g, explainer_seed=e, fid_drop=y.mean(),
                                   fid_inv=group.fid_inv.mean(), one_hop_fid_spearman=rho,
                                   negative_node_count=(group.negative_trials > 0).sum(),
                                   negative_trials=group.negative_trials.sum(),
                                   fid_tie_changes=group.changes_over_1e_6.sum(),
                                   max_tie_drop_range=group.drop_range.max()))
        tables["by_node.csv"] = per_node
        tables["fidelity_by_configuration.csv"] = pd.DataFrame(comparison)
        tables["negative_fidelity.csv"] = detail[detail.fid_drop < 0]
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, table in tables.items():
        path = output / name
        table.to_csv(path, index=False, na_rep="NaN", lineterminator="\n")
        outputs[name] = sha256(path)
    write_json(output / "manifest.json", {**execution_metadata(), "archive": archive.name,
               "archive_sha256": sha256(archive), "data_sha256": data_hash,
               "nodes_sha256": sha256(nodes_path), "accepted_node_hashes": sorted(acceptable_node_hashes),
               "score_and_log_sha256": source_hashes, "outputs": outputs,
               "checkpoint_sha256": checkpoints,
               "mode": "scores_and_fidelity" if with_fidelity else "scores_only",
               "high_auc_threshold": .75, "low_precision_threshold": .5})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/a3_pgexplainer_preliminary"))
    parser.add_argument("--fidelity", action="store_true", help="Use verified H1 checkpoints")
    args = parser.parse_args()
    run(args.archive, args.out, args.fidelity)
