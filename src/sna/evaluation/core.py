"""Shared undirected edge selection, baseline and metric conventions."""
from collections import deque
import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score


def validate_scores(scores, size=None):
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or (size is not None and len(a) != size) or not np.isfinite(a).all():
        raise ValueError('Scores must be finite, one-dimensional and match candidate edges')
    return a


def rank_edges(scores, seed=0):
    """Raw scores stay unchanged; seeded uniform permutation breaks only ties."""
    a = validate_scores(scores)
    permutation = np.random.default_rng(seed).permutation(len(a))
    return permutation[np.argsort(-a[permutation], kind='stable')]


def baseline_scores(pairs, node_id, method):
    if method == 'random':
        return np.zeros(len(pairs), dtype=float)
    if method != 'nearest_first':
        raise ValueError(method)
    adjacency = {}
    for u, v in pairs:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    distance = {node_id: 0}
    queue = deque([node_id])
    while queue:
        u = queue.popleft()
        for v in adjacency.get(u, []):
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    if any(u not in distance or v not in distance for u, v in pairs):
        raise ValueError('Candidate graph contains edges disconnected from target')
    return np.array([-min(distance[u], distance[v]) for u, v in pairs], dtype=float)


def top_indices(order, k):
    if int(k) != k or k < 0:
        raise ValueError('Budget must be a non-negative integer')
    return np.asarray(order, dtype=int)[:min(int(k), len(order))]


def kappa_at_k(selected_a, selected_b, k, m):
    """κ@k hiệu chỉnh theo mức trùng ngẫu nhiên (Phụ lục A của kế hoạch chốt).

    κ@k = (|A∩B| − k²/m) / (k − k²/m), với k = k_eff = min(k, m) và m là số cạnh ứng viên.
    Bằng 1 khi hai tập trùng hoàn toàn, 0 ở mức ngẫu nhiên, âm khi kém ngẫu nhiên.
    Trả về dict {'kappa', 'kappa_reason'}; giá trị không xác định là NaN kèm lý do:
    'no_candidates' (m = 0), 'zero_budget' (k_eff = 0), 'selection_equals_all_candidates'
    (k_eff = m, hai tập luôn trùng nên không có thông tin).
    selected_a/selected_b là chỉ số cạnh đã chọn; chỉ lấy k_eff phần tử đầu mỗi bên.
    """
    if int(k) != k or k < 0 or int(m) != m or m < 0:
        raise ValueError('k và m phải là số nguyên không âm')
    k_eff = min(int(k), int(m))
    nan = float('nan')
    if m == 0:
        return {'kappa': nan, 'kappa_reason': 'no_candidates'}
    if k_eff == 0:
        return {'kappa': nan, 'kappa_reason': 'zero_budget'}
    if k_eff == m:
        return {'kappa': nan, 'kappa_reason': 'selection_equals_all_candidates'}
    a = set(np.asarray(selected_a, dtype=int)[:k_eff].tolist())
    b = set(np.asarray(selected_b, dtype=int)[:k_eff].tolist())
    if len(a) != k_eff or len(b) != k_eff or any(not 0 <= i < m for i in a | b):
        raise ValueError('Tập chọn phải gồm đúng k_eff chỉ số khác nhau trong [0, m)')
    expected = k_eff * k_eff / m
    return {'kappa': float((len(a & b) - expected) / (k_eff - expected)), 'kappa_reason': ''}


def agreement(scores_a, scores_b, k, seed=0, order_a=None, order_b=None):
    a = validate_scores(scores_a)
    b = validate_scores(scores_b, len(a))
    oa = rank_edges(a, seed) if order_a is None else order_a
    ob = rank_edges(b, seed) if order_b is None else order_b
    sa, sb = set(top_indices(oa, k)), set(top_indices(ob, k))
    jac = len(sa & sb) / len(sa | sb) if sa | sb else float('nan')
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        tau, reason = float('nan'), 'constant_scores_or_too_few_edges'
    else:
        tau = float(kendalltau(a, b, variant='b').statistic)
        reason = '' if np.isfinite(tau) else 'undefined_rank_correlation'
    return {'jaccard': jac, 'kendall_tau_b': tau, 'tau_reason': reason,
            'jaccard_reason': '' if sa | sb else 'empty_selected_union',
            **kappa_at_k(top_indices(oa, k), top_indices(ob, k), k, len(a))}


def ground_truth_metrics(pairs, scores, edge_index, gt_edge_mask, selected):
    out = {key: float('nan') for key in ('roc_auc', 'average_precision', 'precision', 'recall')}
    if gt_edge_mask is None:
        return {**out, 'gt_reason': 'no_ground_truth', 'gt_positives': None,
                'auc_reason': 'no_ground_truth', 'precision_reason': 'no_ground_truth',
                'recall_reason': 'no_ground_truth'}
    e = edge_index.detach().cpu().numpy()
    mask = torch.as_tensor(gt_edge_mask).detach().cpu().numpy().astype(bool)
    if mask.ndim != 1 or len(mask) != e.shape[1]:
        raise ValueError('gt_edge_mask must align with columns of edge_index')
    # Vector hóa (kết quả giống hệt vòng lặp từng cạnh; chạy nhanh hơn trên đồ thị lớn).
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    base = max(int(e.max()) if e.size else 0, int(pair_arr.max()) if pair_arr.size else 0) + 2
    edge_keys = np.minimum(e[0], e[1]).astype(np.int64) * base + np.maximum(e[0], e[1])
    unique_keys, inverse = np.unique(edge_keys, return_inverse=True)
    positive = np.bincount(inverse, weights=mask.astype(float), minlength=len(unique_keys))
    total = np.bincount(inverse, minlength=len(unique_keys))
    if ((positive != 0) & (positive != total)).any():
        raise ValueError('Ground truth disagrees between directions of an edge')
    pair_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    pair_keys = np.minimum(pair_arr[:, 0], pair_arr[:, 1]) * base + np.maximum(pair_arr[:, 0], pair_arr[:, 1])
    where = np.searchsorted(unique_keys, pair_keys)
    where_clipped = np.minimum(where, max(len(unique_keys) - 1, 0))
    missing = (len(unique_keys) == 0) | (where >= len(unique_keys)) | (unique_keys[where_clipped] != pair_keys) \
        if len(pair_keys) else np.zeros(0, dtype=bool)
    if np.any(missing):
        raise KeyError(tuple(pair_arr[int(np.argmax(missing))]))
    labels = (positive[where_clipped] > 0).astype(int) if len(pair_keys) else np.zeros(0, dtype=int)
    a = validate_scores(scores, len(labels))
    pos = int(labels.sum())
    chosen = np.asarray(selected, dtype=int)
    hits = int(labels[chosen].sum())
    out['precision'] = hits / len(chosen) if len(chosen) else float('nan')
    out['recall'] = hits / pos if pos else float('nan')
    both = 0 < pos < len(labels)
    if both:
        out['roc_auc'] = float(roc_auc_score(labels, a))
        out['average_precision'] = float(average_precision_score(labels, a))
    return {**out, 'gt_positives': pos, 'gt_reason': '',
            'auc_reason': '' if both else 'single_class_or_empty_candidates',
            'precision_reason': '' if len(chosen) else 'empty_selection',
            'recall_reason': '' if pos else 'no_positive_candidate'}


def fidelity_selected(model, data, node_id, target_class, pairs, selected, orig_prob=None):
    """Deletion: remove selected pairs. Keep: retain only selected original pairs.

    Features stay fixed. Self-loops originally present stay in both graphs;
    GCNConv's implicit self-loops also remain active, even for empty selection.
    """
    device = next(model.parameters()).device
    x, edges = data['x'].to(device), data['edge_index'].to(device)
    chosen = {tuple(pairs[i]) for i in selected}
    endpoints = edges.detach().cpu().numpy()
    top = max([int(endpoints.max()) if endpoints.size else 0] + [max(p) for p in chosen] + [0])
    base = top + 2
    edge_keys = np.minimum(endpoints[0], endpoints[1]).astype(np.int64) * base + np.maximum(endpoints[0], endpoints[1])
    chosen_keys = np.array([u * base + v for u, v in chosen if u <= v], dtype=np.int64)
    selected_mask = torch.from_numpy(np.isin(edge_keys, chosen_keys)).to(device=device, dtype=torch.bool)
    loops = edges[0] == edges[1]
    def probability(e):
        with torch.no_grad():
            return float(model(x, e)[node_id].softmax(-1)[target_class].cpu())
    model.eval()
    original = probability(edges) if orig_prob is None else orig_prob
    deleted = probability(edges[:, ~selected_mask])
    kept = probability(edges[:, selected_mask | loops])
    return {'orig_prob': original, 'delete_prob': deleted, 'keep_prob': kept,
            'fid_drop': original - deleted, 'fid_inv': original - kept}
