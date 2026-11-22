"""Pipeline chạy explainer hàng loạt, xuất CSV điểm cạnh theo schema v2 (F2).

Entry point chính thức cho lưới 5x5 cuối kỳ là ``scripts/run_explainers.py``. Hàm
``run_batch_explaining`` giữ chữ ký cũ (T05) để notebook/test cũ vẫn chạy, nhưng
xuất cùng 17 cột với ``run_explainers.py`` để bộ đánh giá đọc được trực tiếp.

Thay đổi so với giữa kỳ:
- Cột ``src, dst, score`` đổi thành ``u, v, importance_score``; thêm ``target_class``,
  ``seed``/``explainer_seed``, ``gnn_seed``, ``checkpoint_sha256``, ``runtime_sec``, ``status``.
- Không tính Fidelity trong batch_runner nữa: metric chỉ tính ở ``sna.evaluation`` (F4).
  ``df_evals`` được trả về rỗng để giữ tương thích chữ ký cũ.
- PGExplainer khởi tạo với ``seed`` trước khi tạo MLP (F1).
"""

import hashlib
import math
import pathlib
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union

import pandas as pd
import torch

from sna.data.candidate_edges import extract_candidate_edges
from sna.explainers.registry import get_explainer
from sna.pipelines.schema import SCHEMA_V2_COLUMNS, canonical_method, make_run_id


def _sha256_file(path: Union[str, pathlib.Path]) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_batch_explaining(
    model: torch.nn.Module,
    data: Dict[str, Any],
    dataset_name: str,
    node_ids: List[int],
    methods: List[str] = ['saliency', 'gnnexplainer', 'pgexplainer'],
    output_dir: Optional[pathlib.Path] = None,
    seed: int = 0,
    *,
    split: str = 'val',
    arch: str = 'gcn',
    gnn_seed: int = 1000,
    checkpoint_sha256: str = '',
    checkpoint_path: Optional[Union[str, pathlib.Path]] = None,
    pg_config: Optional[Dict[str, Any]] = None,
    explainer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy các explainer trên ``node_ids`` và trả về ``(df_edge_scores, df_evals)``.

    ``seed`` là seed explainer (ghi vào cả ``seed`` và ``explainer_seed``).
    ``checkpoint_sha256`` được tính từ ``checkpoint_path`` nếu truyền đường dẫn; nếu thiếu cả hai,
    cột để trống và phát cảnh báo (``check_scores.py`` sẽ không cho bàn giao CSV như vậy).
    """
    model.eval()
    device = next(model.parameters()).device
    num_nodes = int(data['x'].size(0))

    if checkpoint_path is not None:
        computed = _sha256_file(checkpoint_path)
        if checkpoint_sha256 and checkpoint_sha256 != computed:
            raise ValueError('checkpoint_sha256 không khớp với file checkpoint_path.')
        checkpoint_sha256 = computed
    if not checkpoint_sha256:
        warnings.warn('Thiếu checkpoint_sha256: CSV này không đạt hợp đồng bàn giao H2.', stacklevel=2)

    pg_cfg = {'epochs': 40, 'lr': 0.005, 'temp_init': 5.0, 'temp_min': 0.5}
    pg_cfg.update(pg_config or {})
    method_cfgs = explainer_configs or {}

    # PGExplainer: huấn luyện một lần cho mọi node, seed đặt trước khi tạo MLP (F1).
    pg_instance = None
    pg_train_sec = 0.0
    if any(canonical_method(m) == 'pgexplainer' for m in methods):
        from sna.explainers.pgexplainer import PGExplainerModel
        pg_instance = PGExplainerModel(model=model, device=device, seed=seed)
        pg_train_sec = pg_instance.train_on_graph(
            data=data,
            epochs=int(pg_cfg['epochs']),
            lr=float(pg_cfg['lr']),
            temp_init=float(pg_cfg['temp_init']),
            temp_min=float(pg_cfg['temp_min']),
            coeff_size=pg_cfg.get('coeff_size'),
            coeff_ent=pg_cfg.get('coeff_ent'),
            seed=seed,
            **({'forward_hops': int(pg_cfg['forward_hops'])} if 'forward_hops' in pg_cfg else {}),
        )
        print(f"PGExplainer đã sẵn sàng ({pg_train_sec:.2f}s huấn luyện).")

    print(f"Bắt đầu chạy batch: {len(node_ids)} nodes x {len(methods)} methods = {len(node_ids) * len(methods)} lượt...")

    with torch.no_grad():
        all_preds = model(data['x'].to(device), data['edge_index'].to(device)).argmax(dim=-1).cpu()

    rows: List[Dict[str, Any]] = []
    candidates = {
        int(n): extract_candidate_edges(data['edge_index'], int(n), num_hops=2, num_nodes=num_nodes)
        for n in node_ids
    }

    for method in methods:
        method_name = canonical_method(method)
        run_id = make_run_id(dataset_name, arch, gnn_seed, method_name, seed, split)
        kwargs = {'trained_model': pg_instance} if method_name == 'pgexplainer' else {}
        explainer = get_explainer(method_name, **kwargs)
        cfg = dict(pg_cfg) if method_name == 'pgexplainer' else dict(method_cfgs.get(method_name, {}))

        for node_id in node_ids:
            node_id = int(node_id)
            target_class = int(all_preds[node_id])
            cand = candidates[node_id]
            base = {
                'run_id': run_id, 'dataset': dataset_name, 'split': split, 'arch': arch,
                'gnn_seed': int(gnn_seed), 'checkpoint_sha256': checkpoint_sha256,
                'method': method_name, 'explainer_seed': int(seed), 'seed': int(seed),
                'node_id': node_id, 'target_class': target_class,
            }
            start = time.perf_counter()
            res = explainer.explain(
                model=model, data=data, node_id=node_id, target_class=target_class,
                candidate_edges=cand, config=cfg, seed=seed,
            )
            runtime = res.runtime_sec if res.runtime_sec is not None else time.perf_counter() - start
            runtime = float(runtime) if math.isfinite(float(runtime)) and runtime >= 0 else 0.0

            status = res.status
            if cand.num_candidate_edges == 0 and status == 'success':
                status = 'empty_candidates'
            if status == 'success':
                ok = (len(res.scores) == len(cand.candidate_pairs) == len(res.edge_ids)
                      and all(math.isfinite(float(s)) for s in res.scores))
                if not ok:
                    status = 'failed'
            if status == 'success':
                for eid, (u, v), s in zip(res.edge_ids, cand.candidate_pairs, res.scores):
                    rows.append({**base, 'edge_id': int(eid), 'u': int(u), 'v': int(v),
                                 'importance_score': float(s), 'runtime_sec': runtime, 'status': 'success'})
            else:
                # Một dòng đánh dấu cho node rỗng/lỗi, cùng quy ước với run_explainers.py.
                rows.append({**base, 'edge_id': -1, 'u': -1, 'v': -1,
                             'importance_score': float('nan'), 'runtime_sec': runtime,
                             'status': status if status in ('empty_candidates', 'failed') else 'failed'})

    df_scores = pd.DataFrame(rows, columns=SCHEMA_V2_COLUMNS)
    df_evals = pd.DataFrame()

    if output_dir is not None:
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df_scores.to_csv(out_path / "edge_scores.csv", index=False, encoding='utf-8')
        print(f"Đã lưu kết quả tại: {out_path}")

    return df_scores, df_evals
