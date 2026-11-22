"""Saliency Explainer (Khôi phụ trách).
Tính gradient tuyệt đối của logit mục tiêu theo cạnh (|grad|), gộp max 2 chiều (BG06).
"""

import math
import operator
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


class SaliencyExplainer(BaseExplainer):
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

            if candidate_edges.num_candidate_edges == 0 or len(candidate_edges.candidate_pairs) == 0:
                return ExplainerResult(
                    edge_ids=[],
                    scores=[],
                    runtime_sec=time.time() - start_time,
                    status="empty_candidates",
                )
            if (
                len(candidate_edges.edge_ids) != candidate_edges.num_candidate_edges
                or len(candidate_edges.candidate_pairs) != candidate_edges.num_candidate_edges
            ):
                raise ValueError(
                    f"Inconsistent candidate_edges lengths: num_candidate_edges={candidate_edges.num_candidate_edges}, "
                    f"edge_ids={len(candidate_edges.edge_ids)}, pairs={len(candidate_edges.candidate_pairs)}."
                )

            first_param = next(model.parameters(), None)
            device = first_param.device if first_param is not None else x.device
            model.eval()

            num_edges = edge_index.size(1)
            x_dev = x.to(device)
            edge_index_dev = edge_index.to(device)

            with torch.enable_grad():
                edge_weight = torch.ones(num_edges, device=device, requires_grad=True)
                out = model(x_dev, edge_index_dev, edge_weight=edge_weight)
                if out.ndim < 2 or target_class >= out.size(-1):
                    n_cls = out.size(-1) if out.ndim >= 2 else 0
                    raise ValueError(
                        f"target_class={target_class} out of range [0, {n_cls - 1}]."
                    )
                target_score = out[node_id, target_class]
                if not torch.isfinite(target_score).item():
                    raise ValueError("Model output logit is non-finite.")
                grad = torch.autograd.grad(target_score, edge_weight, allow_unused=True)[0]
            if grad is None:
                raise ValueError("Không tính được gradient theo edge_weight.")

            abs_grad = torch.abs(grad).detach().cpu()

            # Mapping cạnh
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            edge_lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src, dst))}

            scores = []
            for (u, v) in candidate_edges.candidate_pairs:
                idx_uv = edge_lookup.get((u, v))
                idx_vu = edge_lookup.get((v, u))

                val_uv = float(abs_grad[idx_uv].item()) if idx_uv is not None else 0.0
                val_vu = float(abs_grad[idx_vu].item()) if idx_vu is not None else 0.0
                if not math.isfinite(val_uv) or not math.isfinite(val_vu):
                    raise ValueError("Saliency produced non-finite edge gradient.")

                score = max(val_uv, val_vu)
                scores.append(float(score))

            if len(scores) != candidate_edges.num_candidate_edges or not all(
                math.isfinite(s) for s in scores
            ):
                raise ValueError("Saliency produced non-finite importance scores.")

            return ExplainerResult(
                edge_ids=list(candidate_edges.edge_ids),
                scores=scores,
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
