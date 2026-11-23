"""Bộ đánh giá v2 (F3): đọc nhiều (g, e) một lượt, kiểm tra hash, tính metric và độ đồng thuận.

Bố cục đầu vào: ``{scores_root}/{ds}/gcn_s{g}/{method}_e{e}.csv`` (+ ``run_log.json``) theo schema v2
(Mục 6.4 kế hoạch chốt). Checkpoint ``{checkpoints}/gcn_s{g}.pt``.

Quy trình hai bước, để chia việc theo seed GNN chạy song song:

* ``run_part``: với MỘT seed GNN g, nạp checkpoint (so hash với CSV/JSON/run_log, lệch thì dừng),
  tính Fid-/Fid+/AUROC/AP/precision/recall cho explainer và hai baseline, và độ đồng thuận
  within / between / between_avg / mốc tham chiếu (chỉ cần dữ liệu của g đó).
  Ghi vào ``{out}/parts/g{g}/``.
* ``aggregate``: gộp các phần, tính cross-GNN (cần nhiều g; đọc lại CSV điểm, kiểm sha256 so với
  bản đã ghi trong phần) rồi ghi bảng tổng hợp và ``manifest.json``.

Quy ước (đều ghi vào manifest): top-k_eff theo thứ hạng phá hòa bằng hoán vị ngẫu nhiên có seed
(một thứ tự dùng chung mọi budget); NaN kèm cột ``*_reason``; Fidelity âm được giữ nguyên;
SD tính với ddof=1 giữa các node, trung bình theo node trước (repeat/seed không phải node mới).
"""
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_geometric.utils import k_hop_subgraph

from sna.core.config import PROJECT_ROOT
from sna.core.provenance import execution_metadata, protocol, verify_protocol_data, write_json
from sna.data import extract_candidate_edges
from sna.models import GCN
from .core import (agreement, baseline_scores, fidelity_selected, ground_truth_metrics,
                   rank_edges, top_indices)
from .runner import BASELINES, METHODS, seed_for, sha256, summary, validate_score_frame

EXPLAINERS = METHODS[:3]
DETERMINISTIC = {'saliency'}
V2_COLUMNS = ['gnn_seed', 'checkpoint_sha256', 'explainer_seed', 'status']
METRIC_COLS = ['fid_drop', 'fid_inv', 'roc_auc', 'average_precision', 'precision', 'recall']
AGREEMENT_COLS = ['kappa', 'jaccard', 'kendall_tau_b']
LEGACY_G = -1  # nhãn "seed GNN" cho checkpoint giữa kỳ (không có g)
KINDS = ('within', 'cross_gnn', 'cross_gnn_diff_e', 'between', 'between_avg', 'ref_nearest_first', 'ref_random')


class EvaluationError(ValueError):
    """Đầu vào không hợp lệ; bộ đánh giá dừng thay vì tính số trên dữ liệu sai."""


class ProvenanceError(EvaluationError):
    """Hash không khớp (checkpoint, dữ liệu hoặc file node)."""


@dataclass
class EvalConfig:
    dataset: str
    data_path: Path
    nodes_path: Path
    out: Path
    scores_root: Path = None
    checkpoints: Path = None            # thư mục chứa gcn_s{g}.pt (hoặc đường dẫn file, chế độ legacy)
    nodes_limit: int = None
    split: str = 'test'
    budgets: tuple = (1, 3, 5, 10)
    main_budget: int = 5
    gnn_seeds: tuple = (1000, 1001, 1002, 1003, 1004)
    explainer_seeds: tuple = (0, 1, 2, 3, 4)
    methods: tuple = tuple(EXPLAINERS)
    baseline_repeats: tuple = (0, 1, 2, 3, 4)
    saliency_seeds: tuple = (0,)
    saliency_fixed_gnn_seeds: tuple = (0, 1)
    fixed_gnn_seed: int = 1000
    device: str = 'cpu'
    legacy_scores: Path = None          # đặt khác None = chế độ --legacy-midterm
    data_sha256_override: str = None    # chỉ cho test dữ liệu tổng hợp; CLI không có cờ này
    local_fidelity: bool = True         # fidelity trên đồ thị con 3-hop (chính xác cho GCN 2 lớp, nhanh hơn)
    argv: list = field(default_factory=list)

    @property
    def legacy(self):
        return self.legacy_scores is not None

    def checkpoint_path(self, g):
        return Path(self.checkpoints) if self.legacy else Path(self.checkpoints) / f'gcn_s{g}.pt'

    def score_dir(self, g):
        return Path(self.scores_root) / self.dataset / f'gcn_s{g}'

    def expected_seeds(self, method, g):
        """(bắt buộc, tùy chọn). Saliency tất định: chỉ e0, thêm e1 trên GNN cố định (tùy chọn)."""
        if method in DETERMINISTIC:
            optional = self.saliency_fixed_gnn_seeds if g == self.fixed_gnn_seed else ()
            return list(self.saliency_seeds), [e for e in optional if e not in self.saliency_seeds]
        return list(self.explainer_seeds), []


def config_from_protocol(dataset, **overrides):
    """Giá trị mặc định lấy từ configs/protocol_final.yaml."""
    p = protocol()
    defaults = dict(
        dataset=dataset, data_path=PROJECT_ROOT / 'data' / dataset / 'data.pt',
        nodes_path=PROJECT_ROOT / 'data' / dataset / 'test_nodes.json',
        out=PROJECT_ROOT / 'results' / 'eval' / f'{dataset}_test',
        checkpoints=PROJECT_ROOT / 'results' / 'checkpoints' / dataset,
        budgets=tuple(p['budgets']), main_budget=p['main_budget'],
        gnn_seeds=tuple(p['gnn_seeds']), explainer_seeds=tuple(p['explainer_seeds']),
        methods=tuple(p['methods']), baseline_repeats=tuple(p['baseline_repeat_seeds']),
        saliency_seeds=tuple(p['saliency_seeds']),
        saliency_fixed_gnn_seeds=tuple(p['saliency_fixed_gnn_seeds']),
        fixed_gnn_seed=p['fixed_gnn_seed'], device=p.get('device', 'cpu'))
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return EvalConfig(**defaults)


# ---------------------------------------------------------------------------
# Nạp đầu vào và kiểm tra hash
# ---------------------------------------------------------------------------

def load_nodes(path, limit=None):
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    nodes = raw['nodes'] if isinstance(raw, dict) else raw
    nodes = [int(n) for n in nodes]
    if len(set(nodes)) != len(nodes) or not nodes:
        raise EvaluationError(f'{path}: danh sách node rỗng hoặc có node trùng')
    if limit is not None:
        nodes = nodes[:limit]
    return nodes, sha256(path)


def line_ending_variants(path):
    """sha256 của file ở dạng nguyên bản, LF và CRLF.

    Checkout Windows (autocrlf) đổi kết thúc dòng của file JSON nhỏ nên sha256 nguyên bản có thể
    khác bản đã ghi trong run_log.json dù nội dung giống hệt; chấp nhận cả ba dạng cho file node.
    """
    import hashlib
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return {hashlib.sha256(b).hexdigest() for b in (raw, lf, lf.replace(b'\n', b'\r\n'))}


def load_data(cfg):
    if cfg.data_sha256_override is not None:
        digest = sha256(cfg.data_path)
        if digest != cfg.data_sha256_override:
            raise ProvenanceError(f'{cfg.data_path}: sha256 khác giá trị mong đợi')
    else:
        try:
            digest = verify_protocol_data(cfg.dataset, cfg.data_path)
        except ValueError as exc:
            raise ProvenanceError(f'Hash data.pt không khớp protocol_final.yaml: {exc}') from exc
    data = torch.load(cfg.data_path, map_location='cpu', weights_only=False)
    return data, digest


def read_json_if_exists(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8-sig'))


def expected_checkpoint_hashes(cfg, g, csv_hashes, nodes_sha, nodes_hash_variants=None):
    """Các nguồn khai báo hash checkpoint của g: CSV điểm, JSON cạnh checkpoint, run_log.json."""
    sources = [(f'CSV:{name}', h) for name, hs in csv_hashes.items() for h in sorted(hs)]
    ckpt = cfg.checkpoint_path(g)
    meta = read_json_if_exists(ckpt.with_suffix('.json'))
    if meta:
        for key in ('checkpoint_sha256', 'sha256'):
            if meta.get(key):
                sources.append((f'{ckpt.with_suffix(".json").name}:{key}', meta[key]))
    if not cfg.legacy:
        log = read_json_if_exists(cfg.score_dir(g) / 'run_log.json')
        if log:
            if log.get('checkpoint_sha256'):
                sources.append(('run_log.json:checkpoint_sha256', log['checkpoint_sha256']))
            if log.get('node_list_sha256') and log['node_list_sha256'] not in (nodes_hash_variants or {nodes_sha}):
                raise ProvenanceError(
                    f'g={g}: sha256 file node ({nodes_sha}) khác node_list_sha256 trong run_log.json '
                    f'({log["node_list_sha256"]}); node đánh giá không phải danh sách đã chạy explainer')
    return sources


def verify_checkpoint_hash(cfg, g, csv_hashes, nodes_sha):
    variants = line_ending_variants(cfg.nodes_path)
    ckpt = cfg.checkpoint_path(g)
    if not ckpt.is_file():
        raise EvaluationError(f'Không thấy checkpoint: {ckpt}')
    actual = sha256(ckpt)
    for source, declared in expected_checkpoint_hashes(cfg, g, csv_hashes, nodes_sha, variants):
        if declared != actual:
            raise ProvenanceError(
                f'Hash checkpoint LỆCH (g={g}): file {ckpt} có sha256 {actual}, '
                f'nhưng {source} khai báo {declared}. Dừng đánh giá.')
    return actual


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    hidden = ck.get('hidden', ck.get('hidden_dim'))
    classes = ck.get('n_classes', ck.get('out_dim', ck.get('num_classes')))
    model = GCN(int(ck['in_dim']), int(hidden), int(classes), float(ck.get('dropout', 0.0)))
    model.load_state_dict(ck['state_dict'])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


# ---------------------------------------------------------------------------
# Đọc CSV điểm
# ---------------------------------------------------------------------------

@dataclass
class ScoreFile:
    path: Path
    sha: str
    checkpoint_hashes: set
    nodes: dict        # node -> (pairs đã sắp (u, v), điểm theo cùng thứ tự, target_class)
    empty: set
    failed: set
    n_rows: int = 0


def parse_score_frame(df, path, sha, *, dataset, method=None, e=None, g=None, legacy=False):
    """Kiểm tra schema/định danh rồi gom theo node. Dùng lại ``validate_score_frame`` của T06."""
    if not legacy:
        absent = set(V2_COLUMNS) - set(df)
        if absent:
            raise EvaluationError(f'{path}: thiếu cột schema v2: {sorted(absent)}')
        for col, want in (('dataset', dataset), ('method', method), ('gnn_seed', g),
                          ('explainer_seed', e)):
            if set(df[col].astype(type(want))) != {want}:
                raise EvaluationError(f'{path}: cột {col} không khớp {want!r}')
        status = df.status.astype(str)
        unknown = set(status) - {'success', 'empty_candidates', 'failed'}
        if unknown:
            raise EvaluationError(f'{path}: status không hợp lệ: {sorted(unknown)}')
        hashes = set(df.checkpoint_sha256.astype(str))
    else:
        status = pd.Series(['success'] * len(df), index=df.index)
        hashes = set()
    ok = df[status == 'success'].copy()
    nodes = {}
    if not ok.empty:
        ok = validate_score_frame(ok)
        for node, part in ok.groupby('node_id'):
            if part.duplicated('edge_id').any() or part.duplicated(['u', 'v']).any():
                raise EvaluationError(f'{path}: node {node} có cạnh/edge_id trùng')
            if part.target_class.nunique() != 1:
                raise EvaluationError(f'{path}: node {node} có nhiều target_class')
            part = part.sort_values(['u', 'v'])   # thứ tự chuẩn: không phụ thuộc thứ tự dòng
            nodes[int(node)] = (list(zip(part.u.tolist(), part.v.tolist())),
                                part.importance_score.to_numpy(float), int(part.target_class.iloc[0]))
    empty = {int(n) for n in df.loc[status == 'empty_candidates', 'node_id']}
    failed = {int(n) for n in df.loc[status == 'failed', 'node_id']}
    if empty & set(nodes) or failed & set(nodes):
        raise EvaluationError(f'{path}: cùng một node vừa success vừa empty/failed')
    return ScoreFile(Path(path), sha, hashes, nodes, empty, failed, len(df))


def load_score_files(cfg, g):
    """{(method, e): ScoreFile} + lỗi file thiếu. Legacy: một CSV giữa kỳ, e = cột seed."""
    files, errors = {}, []
    if cfg.legacy:
        raw = pd.read_csv(cfg.legacy_scores)
        sha = sha256(cfg.legacy_scores)
        raw = raw[(raw.dataset == cfg.dataset) & ~raw.method.isin(BASELINES)]
        for (method, seed), part in raw.groupby(['method', 'seed']):
            if method in cfg.methods:
                files[(method, int(seed))] = parse_score_frame(part, cfg.legacy_scores, sha,
                                                               dataset=cfg.dataset, legacy=True)
        for method in cfg.methods:
            if not any(m == method for m, _ in files):
                errors.append(dict(g=g, method=method, reason='missing_method_in_legacy_csv'))
        return files, errors
    for method in cfg.methods:
        required, optional = cfg.expected_seeds(method, g)
        for e in required + optional:
            path = cfg.score_dir(g) / f'{method}_e{e}.csv'
            if not path.is_file():
                if e in required:
                    errors.append(dict(g=g, method=method, explainer_seed=e, reason=f'missing_csv:{path.name}'))
                continue
            files[(method, e)] = parse_score_frame(pd.read_csv(path), path, sha256(path),
                                                   dataset=cfg.dataset, method=method, e=e, g=g)
    return files, errors


def tie_seed(cfg, node, method, rep, g):
    """Seed phá hòa. Legacy dùng đúng ``seed_for`` của T06; v2 thêm g để cross-GNN không bị
    thổi phồng do cùng một hoán vị phá hòa."""
    key = method if cfg.legacy else f'{method}:g{g}'
    return seed_for(cfg.dataset, node, key, rep)


class GScores:
    """Điểm đã kiểm tra nhất quán của một seed GNN g, đưa về thứ tự cạnh chuẩn."""

    def __init__(self, cfg, g, files, nodes, pred, graph_pairs=None):
        self.cfg, self.g, self.files = cfg, g, files
        self.pairs, self.scores, self.empty_nodes, self.errors = {}, {}, set(), []
        self.invalid_nodes = set()
        for node in nodes:
            self._build_node(node, pred.get(node), graph_pairs(node) if graph_pairs else None)

    def _err(self, node, method, e, reason):
        self.errors.append(dict(g=self.g, explainer_seed=e, method=method, node_id=node, reason=reason))

    def _build_node(self, node, target, graph):
        reference = None
        usable = {}
        for (method, e), f in self.files.items():
            if node in f.failed:
                self._err(node, method, e, 'explainer_failed')
            elif node in f.empty:
                usable[(method, e)] = ([], np.zeros(0), None)
            elif node in f.nodes:
                usable[(method, e)] = f.nodes[node]
            else:
                self._err(node, method, e, 'missing_node_in_csv')
        pair_sets = {tuple(v[0]) for v in usable.values()}
        if len(pair_sets) > 1:
            self.invalid_nodes.add(node)
            self._err(node, 'all', None, 'methods_disagree_on_candidate_edges')
            return
        if pair_sets:
            reference = list(next(iter(pair_sets)))
            if graph is not None and set(reference) != graph:
                self.invalid_nodes.add(node)
                self._err(node, 'all', None, 'csv_candidates_differ_from_graph_2hop')
                return
        elif graph is not None:
            reference = sorted(graph)
        if reference is None:
            return
        self.pairs[node] = reference
        if not reference:
            self.empty_nodes.add(node)
        for key, (_, scores, tc) in usable.items():
            if tc is not None and target is not None and tc != target:
                self._err(node, key[0], key[1], 'target_class_differs_from_checkpoint')
                continue
            self.scores.setdefault(key, {})[node] = scores

    def seeds_of(self, method):
        return sorted(e for (m, e) in self.scores if m == method)

    def resolve(self, method, e):
        """Điểm của (method, e); phương pháp tất định thiếu e thì dùng seed nhỏ nhất có sẵn."""
        if (method, e) in self.scores:
            return e
        if method in DETERMINISTIC and self.seeds_of(method):
            return self.seeds_of(method)[0]
        return None


class OrderBook:
    """Điểm + thứ hạng phá hòa có cache cho explainer (method, e) và baseline (method, repeat)."""

    def __init__(self, cfg, gs):
        self.cfg, self.gs, self.cache = cfg, gs, {}

    def get(self, method, rep, node):
        key = (method, rep, node)
        if key not in self.cache:
            if method in BASELINES:
                scores = baseline_scores(self.gs.pairs[node], node, method)
            else:
                scores = self.gs.scores[(method, rep)][node]
            seed = tie_seed(self.cfg, node, method, rep, self.gs.g)
            self.cache[key] = (scores, rank_edges(scores, seed), seed)
        return self.cache[key]

    def has(self, method, rep, node):
        if method in BASELINES:
            return node in self.gs.pairs
        return node in self.gs.scores.get((method, rep), {})


def mean_rank_scores(cfg, book, gs, method, node):
    """Điểm = −(thứ hạng trung bình qua các seed explainer); thứ hạng 0 là tốt nhất."""
    seeds = [e for e in gs.seeds_of(method) if book.has(method, e, node)]
    if not seeds:
        return None
    ranks = []
    for e in seeds:
        order = book.get(method, e, node)[1]
        r = np.empty(len(order))
        r[order] = np.arange(len(order))
        ranks.append(r)
    return -np.mean(ranks, axis=0)


# ---------------------------------------------------------------------------
# Hàng kết quả
# ---------------------------------------------------------------------------

NAN = float('nan')


def agreement_rows(base, node, sa, sb, oa, ob, m, budgets):
    rows = []
    for budget in budgets:
        rows.append({**base, 'node_id': node, 'budget': budget, 'k_eff': min(budget, m),
                     'n_candidates': m, **agreement(sa, sb, budget, order_a=oa, order_b=ob)})
    return rows


def pair_base(cfg, kind, g_a, g_b, e_a, e_b, method_a, method_b, pred_same=True):
    return dict(dataset=cfg.dataset, kind=kind, g_a=g_a, g_b=g_b, e_a=e_a, e_b=e_b,
                method_a=method_a, method_b=method_b, pred_same=pred_same)


def ordered(methods):
    return sorted(methods, key=METHODS.index)


def within_between_rows(cfg, g, gs, book, nodes):
    """Độ đồng thuận chỉ cần dữ liệu của một g: within, between, between_avg, mốc tham chiếu."""
    rows = []
    for node in nodes:
        if node not in gs.pairs:
            continue
        m = len(gs.pairs[node])
        # (a) WITHIN
        for method in cfg.methods:
            for e1, e2 in combinations([e for e in gs.seeds_of(method) if book.has(method, e, node)], 2):
                sa, oa, _ = book.get(method, e1, node)
                sb, ob, _ = book.get(method, e2, node)
                rows += agreement_rows(pair_base(cfg, 'within', g, g, e1, e2, method, method),
                                       node, sa, sb, oa, ob, m, cfg.budgets)
        # (c) BETWEEN: cùng (g, e); phương pháp tất định thiếu e dùng seed có sẵn của nó
        for ma, mb in combinations(ordered(cfg.methods), 2):
            grid = sorted({e for meth in (ma, mb) if meth not in DETERMINISTIC
                           for e in gs.seeds_of(meth)} or
                          {e for meth in (ma, mb) for e in gs.seeds_of(meth)})
            for e in grid:
                ea, eb = gs.resolve(ma, e), gs.resolve(mb, e)
                if ea is None or eb is None or not book.has(ma, ea, node) or not book.has(mb, eb, node):
                    continue
                sa, oa, _ = book.get(ma, ea, node)
                sb, ob, _ = book.get(mb, eb, node)
                rows += agreement_rows(pair_base(cfg, 'between', g, g, ea, eb, ma, mb),
                                       node, sa, sb, oa, ob, m, cfg.budgets)
        # (d) BETWEEN_AVG: thứ hạng trung bình qua seed explainer, rồi tính Agr giữa hai thứ hạng
        avg = {meth: mean_rank_scores(cfg, book, gs, meth, node) for meth in cfg.methods}
        for ma, mb in combinations(ordered(cfg.methods), 2):
            if avg[ma] is None or avg[mb] is None:
                continue
            oa = rank_edges(avg[ma], tie_seed(cfg, node, f'{ma}:avgrank', 0, g))
            ob = rank_edges(avg[mb], tie_seed(cfg, node, f'{mb}:avgrank', 0, g))
            rows += agreement_rows(pair_base(cfg, 'between_avg', g, g, -1, -1, ma, mb),
                                   node, avg[ma], avg[mb], oa, ob, m, cfg.budgets)
        # (e) mốc tham chiếu
        for method in cfg.methods:
            for e in gs.seeds_of(method):
                if not book.has(method, e, node):
                    continue
                n_rep = len(cfg.baseline_repeats)
                repeats = cfg.baseline_repeats if cfg.legacy else [cfg.baseline_repeats[e % n_rep]]
                for r in repeats:
                    sa, oa, _ = book.get(method, e, node)
                    sb, ob, _ = book.get('nearest_first', r, node)
                    rows += agreement_rows(pair_base(cfg, 'ref_nearest_first', g, g, e, r, method, 'nearest_first'),
                                           node, sa, sb, oa, ob, m, cfg.budgets)
        for r1, r2 in combinations(cfg.baseline_repeats, 2):
            sa, oa, _ = book.get('random', r1, node)
            sb, ob, _ = book.get('random', r2, node)
            rows += agreement_rows(pair_base(cfg, 'ref_random', g, g, r1, r2, 'random', 'random'),
                                   node, sa, sb, oa, ob, m, cfg.budgets)
    return rows


class LocalGraph:
    """Đồ thị con 3-hop quanh node đích, đã đánh lại chỉ số.

    Với GCN 2 lớp, đầu ra của node đích chỉ phụ thuộc các cạnh liên thuộc node cách ≤ 2 bước
    (kể cả chuẩn hóa theo bậc), nên đồ thị con cảm sinh trên node cách ≤ 3 bước cho cùng xác suất
    (khác biệt chỉ ở sai số làm tròn float32 ~1e-7). Dùng để tránh forward toàn đồ thị Cora.
    """

    def __init__(self, data, node):
        subset, edge_index, mapping, _ = k_hop_subgraph(node, 3, data['edge_index'], relabel_nodes=True,
                                                         num_nodes=len(data['x']))
        self.data = {'x': data['x'][subset], 'edge_index': edge_index}
        self.node = int(mapping)
        self.index = {int(n): i for i, n in enumerate(subset.tolist())}

    def pairs(self, pairs):
        return [(self.index[u], self.index[v]) for u, v in pairs]


def metric_rows(cfg, g, gs, book, model, data, pred, nodes, errors, orig_cache, local_cache=None):
    rows, fid_cache = [], {}
    local_cache = {} if local_cache is None else local_cache
    x_gt = data.get('gt_edge_mask') if cfg.dataset == 'bashapes' else None
    plans = [(meth, e, 'explainer') for meth in cfg.methods for e in gs.seeds_of(meth)]
    plans += [(meth, r, 'baseline') for meth in sorted(BASELINES) for r in cfg.baseline_repeats]
    for node in nodes:
        if node not in gs.pairs:
            continue
        pairs, target = gs.pairs[node], int(pred[node])
        if node not in orig_cache:
            with torch.no_grad():
                orig_cache[node] = float(model(data['x'], data['edge_index'])[node].softmax(-1)[target])
        for method, rep, kind in plans:
            if not book.has(method, rep, node):
                continue
            scores, order, seed = book.get(method, rep, node)
            for budget in cfg.budgets:
                row = dict(dataset=cfg.dataset, g=g, kind=kind, method=method, explainer_seed=rep,
                           node_id=node, target_class=target, budget=budget, k_eff=min(budget, len(pairs)),
                           n_candidates=len(pairs), tie_seed=seed, status='success', reason='')
                if not pairs:
                    row.update(status='empty_candidates', reason='empty_candidates',
                               **{c: NAN for c in METRIC_COLS})
                    rows.append(row)
                    continue
                try:
                    selected = top_indices(order, budget)
                    key = (node, tuple(sorted(int(i) for i in selected)))
                    if key not in fid_cache:
                        if cfg.local_fidelity:
                            if node not in local_cache:
                                local_cache[node] = LocalGraph(data, node)
                            lg = local_cache[node]
                            fid_cache[key] = fidelity_selected(model, lg.data, lg.node, target, lg.pairs(pairs),
                                                               selected, orig_cache[node])
                        else:
                            fid_cache[key] = fidelity_selected(model, data, node, target, pairs, selected,
                                                               orig_cache[node])
                    row.update(fid_cache[key])
                    row.update(ground_truth_metrics(pairs, scores, data['edge_index'], x_gt, selected))
                except Exception as exc:  # không giữ metric dở dang
                    row.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                               **{c: NAN for c in METRIC_COLS})
                    errors.append(dict(g=g, explainer_seed=rep, method=method, node_id=node,
                                       reason=row['reason']))
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Bước 1: một phần theo seed GNN
# ---------------------------------------------------------------------------

def part_dir(cfg, g):
    return Path(cfg.out) / 'parts' / (f'g{g}' if not cfg.legacy else 'legacy')


def validate_budgets(cfg):
    if not cfg.budgets or any(int(b) != b or b < 1 for b in cfg.budgets) or len(set(cfg.budgets)) != len(cfg.budgets):
        raise EvaluationError('budgets phải là các số nguyên dương, không trùng')


def run_part(cfg, g, data, data_sha, nodes, nodes_sha, graph_cache=None, local_cache=None):
    """Tính mọi thứ cần dữ liệu của MỘT seed GNN; ghi vào ``{out}/parts/g{g}``."""
    started = time.perf_counter()
    validate_budgets(cfg)
    out = part_dir(cfg, g)
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    mask = data.get(f'{cfg.split}_mask')
    if mask is None or not all(bool(mask[n]) for n in nodes):
        raise EvaluationError(f'Có node không thuộc split {cfg.split!r} trong danh sách node đánh giá')
    files, file_errors = load_score_files(cfg, g)
    errors += file_errors
    # 1) hash checkpoint — trước khi nạp trọng số
    csv_hashes = {f.path.name if not cfg.legacy else 'legacy': f.checkpoint_hashes for f in files.values()}
    ckpt_sha = verify_checkpoint_hash(cfg, g, csv_hashes, nodes_sha)
    model, ck = load_model(cfg.checkpoint_path(g), cfg.device)
    x, edges = data['x'].to(cfg.device), data['edge_index'].to(cfg.device)
    with torch.no_grad():
        logits = model(x, edges)
        pred_all = logits.argmax(-1).cpu()
        probs_all = logits.softmax(-1).cpu()
    if 'pred' in ck and not torch.equal(pred_all, torch.as_tensor(ck['pred']).cpu()):
        raise EvaluationError(f'g={g}: dự đoán tính lại khác ck["pred"] đã lưu')
    pred = {n: int(pred_all[n]) for n in nodes}
    orig_cache = {n: float(probs_all[n, pred[n]]) for n in nodes}
    data_dev = {**data, 'x': x, 'edge_index': edges}

    graph_cache = {} if graph_cache is None else graph_cache
    local_cache = {} if local_cache is None else local_cache

    def graph_pairs(node):
        if node not in graph_cache:
            cand = extract_candidate_edges(data['edge_index'], node, num_hops=2,
                                           num_nodes=len(data['x']))
            graph_cache[node] = {tuple(p) for p in cand.candidate_pairs}
        return graph_cache[node]

    gs = GScores(cfg, g, files, nodes, pred, graph_pairs)
    errors += gs.errors
    book = OrderBook(cfg, gs)
    t_metrics = time.perf_counter()
    metrics = metric_rows(cfg, g, gs, book, model, data_dev, pred, nodes, errors, orig_cache, local_cache)
    t_agree = time.perf_counter()
    agree = within_between_rows(cfg, g, gs, book, nodes)
    done = time.perf_counter()
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(agree).to_csv(out / 'agreement.csv', index=False)
    pd.DataFrame([dict(g=g, node_id=n, pred=pred[n]) for n in nodes]).to_csv(out / 'preds.csv', index=False)
    pd.DataFrame(errors, columns=['g', 'explainer_seed', 'method', 'node_id', 'reason']).to_csv(
        out / 'errors.csv', index=False)
    info = dict(
        g=g, dataset=cfg.dataset, status='completed_with_errors' if errors else 'completed',
        data_sha256=data_sha, nodes_sha256=nodes_sha, n_nodes=len(nodes),
        checkpoint_path=str(cfg.checkpoint_path(g)), checkpoint_sha256=ckpt_sha,
        score_csv_sha256={f'{m}_e{e}': f.sha for (m, e), f in sorted(files.items())},
        n_metric_rows=len(metrics), n_agreement_rows=len(agree), n_errors=len(errors),
        seconds=dict(load=round(t_metrics - started, 3), metrics=round(t_agree - t_metrics, 3),
                     agreement=round(done - t_agree, 3), total=round(done - started, 3)))
    write_json(out / 'part_manifest.json', info)
    return info


# ---------------------------------------------------------------------------
# Bước 2: gộp, cross-GNN, tổng hợp, manifest
# ---------------------------------------------------------------------------

def cross_gnn_rows(cfg, parts_info, nodes):
    """(b) CROSS-GNN: cùng (phương pháp, e), mọi cặp g<g', chỉ node hai GNN dự đoán cùng lớp.

    Đọc lại CSV điểm (kiểm sha256 so với bản ghi trong phần) và lấy dự đoán từ ``preds.csv``.
    Node đổi dự đoán vẫn có dòng, kappa/jaccard/tau là NaN với lý do ``pred_changed``.
    """
    gs_by_g, preds = {}, {}
    for g in sorted(parts_info):
        info = parts_info[g]
        files, _ = load_score_files(cfg, g)
        for (m, e), f in files.items():
            if info['score_csv_sha256'].get(f'{m}_e{e}') != f.sha:
                raise ProvenanceError(f'g={g}: CSV {f.path.name} đã đổi kể từ lúc tính phần (sha256 lệch)')
        p = pd.read_csv(part_dir(cfg, g) / 'preds.csv')
        preds[g] = dict(zip(p.node_id.astype(int), p.pred.astype(int)))
        gs_by_g[g] = GScores(cfg, g, files, nodes, preds[g])
    books = {g: OrderBook(cfg, gs) for g, gs in gs_by_g.items()}
    rows, change = [], []
    for ga, gb in combinations(sorted(gs_by_g), 2):
        changed = [n for n in nodes if preds[ga].get(n) != preds[gb].get(n)]
        change.append(dict(dataset=cfg.dataset, g_a=ga, g_b=gb, n_nodes=len(nodes),
                           n_pred_changed=len(changed), pred_change_rate=len(changed) / len(nodes)))
        for method in cfg.methods:
            seeds = sorted(set(gs_by_g[ga].seeds_of(method)) & set(gs_by_g[gb].seeds_of(method)))
            # 'cross_gnn' (Mục 7.3): cùng seed explainer e ở hai GNN. Với phương pháp có khởi tạo ngẫu nhiên
            # theo seed, hai lượt cùng e dùng CÙNG khởi tạo nên đo này có thể bị thổi phồng. Vì vậy thêm
            # 'cross_gnn_diff_e' (đối chứng): ghép (g, e) với (g', e'), e' là seed kế tiếp (vòng tròn), e' != e.
            plans = [('cross_gnn', e, e) for e in seeds]
            if len(seeds) > 1:
                plans += [('cross_gnn_diff_e', e, seeds[(i + 1) % len(seeds)]) for i, e in enumerate(seeds)]
            for kind, e_a, e_b in plans:
                for node in nodes:
                    if node not in gs_by_g[ga].pairs or node not in gs_by_g[gb].pairs:
                        continue
                    if not (books[ga].has(method, e_a, node) and books[gb].has(method, e_b, node)):
                        continue
                    m = len(gs_by_g[ga].pairs[node])
                    base = pair_base(cfg, kind, ga, gb, e_a, e_b, method, method,
                                     pred_same=preds[ga][node] == preds[gb][node])
                    if not base['pred_same']:
                        rows += [{**base, 'node_id': node, 'budget': b, 'k_eff': min(b, m), 'n_candidates': m,
                                  'jaccard': NAN, 'kendall_tau_b': NAN, 'kappa': NAN,
                                  'tau_reason': 'pred_changed', 'jaccard_reason': 'pred_changed',
                                  'kappa_reason': 'pred_changed'} for b in cfg.budgets]
                        continue
                    sa, oa, _ = books[ga].get(method, e_a, node)
                    sb, ob, _ = books[gb].get(method, e_b, node)
                    rows += agreement_rows(base, node, sa, sb, oa, ob, m, cfg.budgets)
    return rows, change


def summarize(cfg, metrics, agree):
    outputs = {}
    if len(metrics):
        outputs['summary.csv'] = summary(metrics, ['dataset', 'method', 'budget'], METRIC_COLS)
        outputs['summary_by_g.csv'] = summary(metrics, ['dataset', 'g', 'method', 'budget'], METRIC_COLS)
    if len(agree):
        outputs['agreement_summary.csv'] = summary(
            agree, ['dataset', 'kind', 'method_a', 'method_b', 'budget'], AGREEMENT_COLS)
        outputs['agreement_summary_by_g.csv'] = summary(
            agree, ['dataset', 'kind', 'g_a', 'g_b', 'method_a', 'method_b', 'budget'], AGREEMENT_COLS)
    return outputs


def aggregate(cfg, data_sha, nodes, nodes_sha, run_seconds=None):
    started = time.perf_counter()
    validate_budgets(cfg)
    out = Path(cfg.out)
    gs_list = list(cfg.gnn_seeds) if not cfg.legacy else [LEGACY_G]
    parts_info, metrics, agree, errors, preds = {}, [], [], [], []
    for g in gs_list:
        pdir = part_dir(cfg, g)
        if not (pdir / 'part_manifest.json').is_file():
            raise EvaluationError(f'Thiếu phần cho g={g}: {pdir}. Chạy phần đó trước (--gnn-seeds {g}).')
        info = json.loads((pdir / 'part_manifest.json').read_text(encoding='utf-8'))
        if info['data_sha256'] != data_sha or info['nodes_sha256'] != nodes_sha or info['n_nodes'] != len(nodes):
            raise ProvenanceError(f'g={g}: phần đã tính với data/danh sách node khác bản hiện tại')
        parts_info[g] = info
        metrics.append(pd.read_csv(pdir / 'metrics.csv'))
        agree.append(pd.read_csv(pdir / 'agreement.csv'))
        errors.append(pd.read_csv(pdir / 'errors.csv'))
        preds.append(pd.read_csv(pdir / 'preds.csv'))
    metrics_df = pd.concat([d for d in metrics if len(d)], ignore_index=True) if any(len(d) for d in metrics) else pd.DataFrame()
    agree_df = pd.concat([d for d in agree if len(d)], ignore_index=True) if any(len(d) for d in agree) else pd.DataFrame()
    cross, change = ([], [])
    if not cfg.legacy and len(gs_list) > 1:
        cross, change = cross_gnn_rows(cfg, parts_info, nodes)
        agree_df = pd.concat([agree_df, pd.DataFrame(cross)], ignore_index=True)
    errors_df = pd.concat(errors, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out / 'metrics.csv', index=False)
    agree_df.to_csv(out / 'agreement.csv', index=False)
    errors_df.to_csv(out / 'errors.csv', index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / 'preds.csv', index=False)
    pd.DataFrame(change, columns=['dataset', 'g_a', 'g_b', 'n_nodes', 'n_pred_changed',
                                  'pred_change_rate']).to_csv(out / 'pred_change.csv', index=False)
    for name, table in summarize(cfg, metrics_df, agree_df).items():
        table.to_csv(out / name, index=False)
    meta = execution_metadata()
    meta['argv'] = list(cfg.argv) or meta['argv']
    manifest = {
        **meta,
        'schema': 'evaluator_v2', 'dataset': cfg.dataset, 'split': cfg.split,
        'mode': 'legacy_midterm' if cfg.legacy else 'final',
        'device': cfg.device, 'status': 'completed_with_errors' if len(errors_df) else 'completed',
        'error_count': int(len(errors_df)),
        'data_path': str(cfg.data_path), 'data_sha256': data_sha,
        'nodes_path': str(cfg.nodes_path), 'nodes_sha256': nodes_sha, 'n_nodes': len(nodes),
        'node_ids': nodes,
        'checkpoints': {str(g): {'path': i['checkpoint_path'], 'sha256': i['checkpoint_sha256']}
                        for g, i in parts_info.items()},
        'score_csv_sha256': {f'g{g}/{k}': v for g, i in parts_info.items()
                             for k, v in i['score_csv_sha256'].items()},
        'protocol': {
            'budgets': list(cfg.budgets), 'main_budget': cfg.main_budget,
            'gnn_seeds': list(gs_list), 'explainer_seeds': list(cfg.explainer_seeds),
            'methods': list(cfg.methods), 'baselines': sorted(BASELINES),
            'baseline_repeat_seeds': list(cfg.baseline_repeats),
            'k_eff': 'min(k, m)',
            'tie_policy': 'seeded_uniform_permutation_shared_across_budgets; '
                          'seed = seed_for(dataset, node, f"{method}:g{g}", repeat) (legacy: không có g)',
            'nan_policy': 'nan_with_reason; Fidelity âm được giữ nguyên (keep_negative_fidelity)',
            'kappa': 'κ@k=(|A∩B|−k²/m)/(k−k²/m); NaN khi k_eff=m hoặc m=0',
            'sd': 'sample SD (ddof=1) giữa node; trung bình theo node trước khi tổng hợp',
            'cross_gnn': 'chỉ node hai GNN dự đoán cùng lớp (pred_same); node khác lớp là NaN lý do pred_changed',
            'cross_gnn_diff_e': "đối chứng cho cross_gnn: ghép (g, e) với (g', e' khác e), loại ảnh hưởng của khởi tạo chung theo seed",
            'between_avg': 'điểm = −(thứ hạng trung bình qua seed explainer), phá hòa bằng seed',
            'ref_nearest_first': 'explainer (g, e) với nearest_first repeat r=e (legacy: mọi repeat)',
            'fidelity_graph': 'đồ thị con 3-hop (tương đương toàn đồ thị với GCN 2 lớp, sai khác ~1e-7)'
                              if cfg.local_fidelity else 'toàn đồ thị',
            'saliency': 'tất định: chỉ e0 (và e1 trên GNN cố định); between dùng e0 cho mọi e'},
        'row_counts': {'metrics': int(len(metrics_df)), 'agreement': int(len(agree_df)),
                       'cross_gnn': len(cross)},
        'part_seconds': {str(g): i['seconds'] for g, i in parts_info.items()},
        'aggregate_seconds': round(time.perf_counter() - started, 3),
    }
    write_json(out / 'manifest.json', manifest)
    return manifest


# ---------------------------------------------------------------------------
# Điều phối
# ---------------------------------------------------------------------------

def run(cfg, mode='all'):
    """mode: 'all' (phần + gộp), 'parts' (chỉ phần), 'aggregate' (chỉ gộp)."""
    nodes, nodes_sha = load_nodes(cfg.nodes_path, cfg.nodes_limit)
    data, data_sha = load_data(cfg)
    if mode in ('all', 'parts'):
        cache, local_cache = {}, {}
        for g in ([LEGACY_G] if cfg.legacy else cfg.gnn_seeds):
            info = run_part(cfg, g, data, data_sha, nodes, nodes_sha, cache, local_cache)
            print(f"g={g}: {info['status']} metrics={info['n_metric_rows']} "
                  f"agreement={info['n_agreement_rows']} errors={info['n_errors']} "
                  f"{info['seconds']['total']}s", flush=True)
    if mode in ('all', 'aggregate'):
        manifest = aggregate(cfg, data_sha, nodes, nodes_sha)
        print(f"{manifest['status']}: {cfg.out}", flush=True)
        return manifest
    return None


def build_parser():
    p = argparse.ArgumentParser(description='Bộ đánh giá v2: metric + độ đồng thuận (κ@k, Jaccard, τ_b).')
    p.add_argument('--dataset', required=True, choices=['bashapes', 'cora'])
    p.add_argument('--scores-root', type=Path, default=PROJECT_ROOT / 'results' / 'scores')
    p.add_argument('--data', type=Path, help='đường dẫn data.pt (mặc định data/{ds}/data.pt)')
    p.add_argument('--checkpoints', type=Path, help='thư mục gcn_s{g}.pt (mặc định results/checkpoints/{ds})')
    p.add_argument('--nodes', type=Path, help='JSON danh sách node (mặc định data/{ds}/test_nodes.json)')
    p.add_argument('--nodes-limit', type=int, help='chỉ lấy N node đầu danh sách (thử nhanh)')
    p.add_argument('--out', type=Path, help='mặc định results/eval/{ds}_test')
    p.add_argument('--budgets', type=int, nargs='+')
    p.add_argument('--gnn-seeds', type=int, nargs='+')
    p.add_argument('--explainer-seeds', type=int, nargs='+')
    p.add_argument('--methods', nargs='+', choices=EXPLAINERS)
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--full-graph-fidelity', action='store_true',
                   help='tính fidelity trên toàn đồ thị (chậm hơn, chính xác bit-by-bit với core); '
                        'mặc định dùng đồ thị con 3-hop')
    p.add_argument('--device', choices=['cpu', 'cuda'])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--parts-only', action='store_true',
                      help='chỉ tính phần theo từng seed GNN (chạy song song), không gộp')
    mode.add_argument('--aggregate', action='store_true',
                      help='chỉ gộp các phần đã có (cần đủ phần cho mọi --gnn-seeds) và tính cross-GNN')
    p.add_argument('--legacy-midterm', action='store_true',
                   help='chạy trên CSV điểm giữa kỳ (--scores) với checkpoint giữa kỳ; không có hash CSV')
    p.add_argument('--scores', type=Path, help='CSV giữa kỳ (chỉ dùng với --legacy-midterm)')
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    ds = a.dataset
    over = dict(scores_root=a.scores_root, data_path=a.data, checkpoints=a.checkpoints,
                nodes_path=a.nodes, out=a.out, nodes_limit=a.nodes_limit, split=a.split,
                budgets=tuple(a.budgets) if a.budgets else None,
                gnn_seeds=tuple(a.gnn_seeds) if a.gnn_seeds else None,
                explainer_seeds=tuple(a.explainer_seeds) if a.explainer_seeds else None,
                methods=tuple(a.methods) if a.methods else None, device=a.device,
                local_fidelity=False if (a.full_graph_fidelity or a.legacy_midterm) else None,
                argv=['evaluate.py', *argv])
    if a.legacy_midterm:
        if a.scores is None:
            build_parser().error('--legacy-midterm cần --scores')
        d = PROJECT_ROOT / 'data' / ds
        over.update(legacy_scores=a.scores,
                    checkpoints=a.checkpoints or d / 'gcn_ckpt.pt',
                    nodes_path=a.nodes or d / 'val_nodes.json',
                    nodes_limit=a.nodes_limit or 20, split='val',
                    out=a.out or PROJECT_ROOT / 'results' / 'eval' / f'{ds}_legacy_midterm',
                    saliency_seeds=(42,), saliency_fixed_gnn_seeds=())
    cfg = config_from_protocol(ds, **over)
    mode = 'parts' if a.parts_only else 'aggregate' if a.aggregate else 'all'
    try:
        manifest = run(cfg, mode)
    except EvaluationError as exc:
        print(f'LỖI: {exc}', file=sys.stderr)
        return 2
    return 1 if manifest and manifest['error_count'] else 0
