"""FB-3 Error Analysis mechanisms T2 (motif vs 1-hop deletion) and T3 (recomputed degree features)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from sna.data.candidate_edges import extract_candidate_edges


def _build_pair_gt_map(edge_index: torch.Tensor, gt_edge_mask: torch.Tensor) -> Dict[Tuple[int, int], bool]:
    e = edge_index.detach().cpu().numpy()
    mask = torch.as_tensor(gt_edge_mask).detach().cpu().numpy().astype(bool)
    if mask.ndim != 1 or len(mask) != e.shape[1]:
        raise ValueError("gt_edge_mask must align with columns of edge_index")
    labels_by_pair: Dict[Tuple[int, int], bool] = {}
    for j, (u, v) in enumerate(e.T):
        pair = (min(int(u), int(v)), max(int(u), int(v)))
        labels_by_pair[pair] = bool(mask[j])
    return labels_by_pair


def _filter_edges(edge_index: torch.Tensor, remove_pairs: Set[Tuple[int, int]]) -> torch.Tensor:
    if not remove_pairs:
        return edge_index
    endpoints = edge_index.detach().cpu().t().tolist()
    keep_mask = torch.tensor(
        [(min(u, v), max(u, v)) not in remove_pairs for u, v in endpoints],
        device=edge_index.device,
        dtype=torch.bool,
    )
    return edge_index[:, keep_mask]


def recompute_degree_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Recompute static degree features x[:, 0] = deg' and x[:, 1] = sqrt(deg') on perturbed graph."""
    x_new = x.clone()
    num_nodes = x.shape[0]
    non_loops = edge_index[:, edge_index[0] != edge_index[1]]
    deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
    if non_loops.numel() > 0:
        # Count unique undirected neighbors or out-degree (for symmetric undirected edge_index, row=0 gives degree)
        deg.index_add_(0, non_loops[0], torch.ones(non_loops.shape[1], dtype=x.dtype, device=x.device))
    x_new[:, 0] = deg
    if x_new.shape[1] > 1:
        x_new[:, 1] = torch.sqrt(deg)
    return x_new


def _target_prob(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor, node_id: int, target_class: int) -> float:
    with torch.no_grad():
        logits = model(x, edge_index)
        return float(logits[node_id].softmax(dim=-1)[target_class].cpu())


def evaluate_node_t2_t3(
    model: torch.nn.Module,
    data: Dict[str, Any],
    node_id: int,
    target_class: int,
    gt_pair_map: Optional[Dict[Tuple[int, int], bool]] = None,
) -> Dict[str, Any]:
    """Run T2 (Fid_motif vs Fid_1hop) and T3 (recomputed x degree features) on a single node."""
    device = next(model.parameters()).device
    model.eval()
    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)

    if gt_pair_map is None:
        gt_mask = data.get("edge_mask", data.get("gt_edge_mask"))
        if gt_mask is None:
            raise ValueError("Dataset does not contain ground truth edge_mask for T2/T3.")
        gt_pair_map = _build_pair_gt_map(edge_index, gt_mask)

    cand = extract_candidate_edges(edge_index, node_id, num_hops=2)
    pairs = cand.candidate_pairs

    motif_pairs = {p for p in pairs if gt_pair_map.get(p, False)}
    one_hop_pairs = {p for p in pairs if p[0] == node_id or p[1] == node_id}

    orig_prob = _target_prob(model, x, edge_index, node_id, target_class)

    # T2: delete all 2-hop motif edges (fixed x)
    edges_no_motif = _filter_edges(edge_index, motif_pairs)
    prob_no_motif_fixed_x = _target_prob(model, x, edges_no_motif, node_id, target_class)
    fid_motif = orig_prob - prob_no_motif_fixed_x

    # T2: delete all 1-hop edges (fixed x)
    edges_no_1hop = _filter_edges(edge_index, one_hop_pairs)
    prob_no_1hop_fixed_x = _target_prob(model, x, edges_no_1hop, node_id, target_class)
    fid_1hop = orig_prob - prob_no_1hop_fixed_x

    # T3: recompute [deg', sqrt(deg')] after deletion
    x_no_motif = recompute_degree_features(x, edges_no_motif)
    prob_no_motif_recomp_x = _target_prob(model, x_no_motif, edges_no_motif, node_id, target_class)
    fid_motif_recomp_x = orig_prob - prob_no_motif_recomp_x

    x_no_1hop = recompute_degree_features(x, edges_no_1hop)
    prob_no_1hop_recomp_x = _target_prob(model, x_no_1hop, edges_no_1hop, node_id, target_class)
    fid_1hop_recomp_x = orig_prob - prob_no_1hop_recomp_x

    return {
        "node_id": int(node_id),
        "target_class": int(target_class),
        "num_cand_edges": len(pairs),
        "num_motif_edges_2hop": len(motif_pairs),
        "num_1hop_edges": len(one_hop_pairs),
        "orig_prob": orig_prob,
        "fid_motif": fid_motif,
        "fid_1hop": fid_1hop,
        "fid_motif_recomp_x": fid_motif_recomp_x,
        "fid_1hop_recomp_x": fid_1hop_recomp_x,
        "t3_motif_delta": fid_motif_recomp_x - fid_motif,
        "t3_1hop_delta": fid_1hop_recomp_x - fid_1hop,
    }


def evaluate_dataset_t2_t3(
    model: torch.nn.Module,
    data: Dict[str, Any],
    node_ids: List[int],
    pred: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Run T2 and T3 across all specified nodes."""
    gt_mask = data.get("edge_mask", data.get("gt_edge_mask"))
    if gt_mask is None:
        raise ValueError("Dataset does not contain ground truth edge_mask for T2/T3.")
    gt_pair_map = _build_pair_gt_map(data["edge_index"], gt_mask)

    rows: List[Dict[str, Any]] = []
    for nid in node_ids:
        row = evaluate_node_t2_t3(
            model=model,
            data=data,
            node_id=int(nid),
            target_class=int(pred[int(nid)]),
            gt_pair_map=gt_pair_map,
        )
        rows.append(row)
    return rows

