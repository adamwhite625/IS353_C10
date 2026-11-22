"""GNNExplainer (Khôi phụ trách).
Tối ưu continuous edge mask bằng Cross-Entropy + Sparsity L1 + Entropy (BG06).
"""

import math
import numbers
import operator
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


def _validate_finite_real(
    value: Any,
    name: str,
    *,
    strictly_positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name}={value!r} must be a finite real number.")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name}={value!r} must be finite.")
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"{name}={fval} must be positive.")
    if non_negative and fval < 0.0:
        raise ValueError(f"{name}={fval} must be non-negative.")
    return fval


class GNNExplainer(BaseExplainer):
    def explain(
        self,
        model: torch.nn.Module,
        data: Dict[str, Any],
        node_id: int,
        target_class: int,
        candidate_edges: CandidateEdgesResult,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 0
    ) -> ExplainerResult:
        start_time = time.time()
        try:
            if isinstance(seed, (bool, np.bool_)):
                raise ValueError(f"seed={seed!r} must be an integer.")
            seed = operator.index(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            cfg = config or {}
            epochs_raw = cfg.get('epochs', 200)
            if isinstance(epochs_raw, (bool, np.bool_)):
                raise ValueError(f"epochs={epochs_raw!r} must be a non-negative integer.")
            try:
                epochs = operator.index(epochs_raw)
            except TypeError as exc:
                raise ValueError(f"epochs={epochs_raw!r} must be a non-negative integer.") from exc
            if epochs < 0:
                raise ValueError(f"epochs={epochs!r} must be a non-negative integer.")
            lr = _validate_finite_real(cfg.get('lr', 0.01), 'lr', strictly_positive=True)
            coeff_size = _validate_finite_real(cfg.get('coeff_size', 0.005), 'coeff_size', non_negative=True)
            coeff_ent = _validate_finite_real(cfg.get('coeff_ent', 0.1), 'coeff_ent', non_negative=True)
            size_loss_mode = cfg.get('size_loss_mode', 'mean')
            if size_loss_mode not in ('mean', 'sum'):
                raise ValueError(
                    f"Invalid size_loss_mode: '{size_loss_mode}'. Expected 'mean' or 'sum'."
                )

            x = data['x']
            edge_index = data['edge_index']
            if isinstance(node_id, (bool, np.bool_)):
                raise ValueError(f"node_id={node_id!r} must be an integer.")
            node_id = operator.index(node_id)
            if isinstance(target_class, (bool, np.bool_)):
                raise ValueError(f"target_class={target_class!r} must be an integer.")
            target_class = operator.index(target_class)
            if node_id < 0 or node_id >= x.size(0):
                raise ValueError(f"node_id={node_id} out of range [0, {x.size(0) - 1}].")
            if target_class < 0:
                raise ValueError(f"target_class={target_class} must be non-negative.")

            num_candidates = candidate_edges.num_candidate_edges
            if num_candidates == 0 or len(candidate_edges.candidate_pairs) == 0:
                return ExplainerResult(
                    edge_ids=[],
                    scores=[],
                    runtime_sec=time.time() - start_time,
                    status="empty_candidates",
                )
            if (
                len(candidate_edges.edge_ids) != num_candidates
                or len(candidate_edges.candidate_pairs) != num_candidates
            ):
                raise ValueError(
                    f"Inconsistent candidate_edges lengths: num_candidate_edges={num_candidates}, "
                    f"edge_ids={len(candidate_edges.edge_ids)}, pairs={len(candidate_edges.candidate_pairs)}."
                )

            first_param = next(model.parameters(), None)
            device = first_param.device if first_param is not None else x.device
            model.eval()

            sub_edge_index = candidate_edges.subgraph_edge_index.to(device)
            edge_to_pair_idx = candidate_edges.edge_to_pair_idx.to(device)

            mask_logits = torch.nn.Parameter(torch.randn(num_candidates, device=device) * 0.1 + 1.0)
            optimizer = torch.optim.Adam([mask_logits], lr=lr)

            num_total_edges = edge_index.size(1)
            base_edge_weight = torch.ones(num_total_edges, device=device)

            src_sub = sub_edge_index[0].cpu().numpy()
            dst_sub = sub_edge_index[1].cpu().numpy()
            src_all = edge_index[0].cpu().numpy()
            dst_all = edge_index[1].cpu().numpy()
            all_edge_map = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src_all, dst_all))}

            sub_to_orig_indices = torch.tensor(
                [all_edge_map[(int(u), int(v))] for u, v in zip(src_sub, dst_sub)],
                dtype=torch.long, device=device
            )

            x_dev = x.to(device)
            edge_index_dev = edge_index.to(device)

            # Temporarily freeze model parameters so loss.backward() never pollutes model.grad
            saved_req_grad = [(p, p.requires_grad) for p in model.parameters()]
            for p, _ in saved_req_grad:
                p.requires_grad_(False)

            try:
                if epochs == 0:
                    with torch.no_grad():
                        out = model(x_dev, edge_index_dev, edge_weight=base_edge_weight)
                    if out.ndim < 2 or target_class >= out.size(-1):
                        n_cls = out.size(-1) if out.ndim >= 2 else 0
                        raise ValueError(
                            f"target_class={target_class} out of range [0, {n_cls - 1}]."
                        )
                    if not torch.isfinite(out[node_id]).all().item():
                        raise ValueError("Model output logits contain non-finite values.")
                else:
                    with torch.enable_grad():
                        for epoch in range(epochs):
                            optimizer.zero_grad()
                            pair_probs = torch.sigmoid(mask_logits)
                            sub_probs = pair_probs[edge_to_pair_idx]

                            current_weight = base_edge_weight.clone()
                            current_weight[sub_to_orig_indices] = sub_probs

                            out = model(x_dev, edge_index_dev, edge_weight=current_weight)
                            if epoch == 0 and (out.ndim < 2 or target_class >= out.size(-1)):
                                n_cls = out.size(-1) if out.ndim >= 2 else 0
                                raise ValueError(
                                    f"target_class={target_class} out of range [0, {n_cls - 1}]."
                                )
                            if not torch.isfinite(out[node_id]).all().item():
                                raise ValueError("Model output logits contain non-finite values.")

                            log_probs = F.log_softmax(out[node_id:node_id+1], dim=-1)
                            loss_pred = -log_probs[0, target_class]
                            if size_loss_mode == 'sum':
                                loss_size = coeff_size * torch.sum(pair_probs)
                            else:
                                loss_size = coeff_size * torch.mean(pair_probs)

                            eps = 1e-12
                            ent = -pair_probs * torch.log(pair_probs + eps) - (1 - pair_probs) * torch.log(1 - pair_probs + eps)
                            loss_ent = coeff_ent * torch.mean(ent)

                            loss = loss_pred + loss_size + loss_ent
                            if not torch.isfinite(loss).item():
                                raise ValueError("GNNExplainer loss became non-finite.")
                            loss.backward()
                            optimizer.step()
            finally:
                for p, req in saved_req_grad:
                    p.requires_grad_(req)

            with torch.no_grad():
                final_scores = [
                    float(s) for s in torch.sigmoid(mask_logits).detach().cpu().numpy().tolist()
                ]
            if len(final_scores) != num_candidates or not all(
                math.isfinite(s) for s in final_scores
            ):
                raise ValueError("GNNExplainer produced non-finite importance scores.")

            return ExplainerResult(
                edge_ids=list(candidate_edges.edge_ids),
                scores=final_scores,
                runtime_sec=time.time() - start_time,
                status="success"
            )

        except Exception as e:
            return ExplainerResult(
                edge_ids=[],
                scores=[],
                runtime_sec=time.time() - start_time,
                status="failed",
                error_message=str(e)
            )
