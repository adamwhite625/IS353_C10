"""Kiểm thử D1 Seed-Ensemble (scripts/analyze_ensemble.py)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import analyze_ensemble as ae  # noqa: E402
from sna.evaluation.core import rank_edges  # noqa: E402
from sna.evaluation.runner_v2 import mean_rank_scores  # noqa: E402


class FakeBook:
    """Giả OrderBook: điểm theo (method, e, node); thứ tự phá hòa bằng seed cố định theo e."""

    def __init__(self, scores):
        self.scores = scores

    def has(self, method, e, node):
        return (method, e, node) in self.scores

    def get(self, method, e, node):
        s = self.scores[(method, e, node)]
        return s, rank_edges(s, 100 + e), 100 + e


def test_mean_rank_by_hand():
    # seed A: cạnh 2 > 0 > 1 ; seed B: 0 > 2 > 1 -> hạng A=[1,2,0], B=[0,2,1]; TB=[0.5,2,0.5]
    a = np.array([5.0, 1.0, 9.0])
    b = np.array([9.0, 1.0, 5.0])
    got = ae.ensemble_scores([rank_edges(a), rank_edges(b)])
    assert np.allclose(got, [-0.5, -2.0, -0.5])


def test_single_seed_ensemble_equals_itself():
    s = np.array([0.3, 0.9, 0.1, 0.5, 0.7])
    book = FakeBook({('gnnexplainer', 0, 7): s})
    ens = ae.ensemble_for(book, 'gnnexplainer', [0], 7)
    assert np.array_equal(rank_edges(ens, 3), rank_edges(s, 100))
    # có hòa: thứ tự phá hòa của seed đơn được giữ nguyên
    t = np.array([1.0, 1.0, 1.0, 0.0])
    book = FakeBook({('pgexplainer', 2, 1): t})
    ens = ae.ensemble_for(book, 'pgexplainer', [2], 1)
    assert np.array_equal(np.argsort(-ens, kind='stable'), book.get('pgexplainer', 2, 1)[1])


def test_deterministic_and_matches_runner_mean_rank():
    rng = np.random.default_rng(0)
    scores = {('gnnexplainer', e, 4): np.round(rng.random(12), 1) for e in range(5)}   # có hòa
    book = FakeBook(scores)
    first = ae.ensemble_for(book, 'gnnexplainer', range(5), 4)
    assert np.array_equal(first, ae.ensemble_for(FakeBook(scores), 'gnnexplainer', range(5), 4))
    gs = SimpleNamespace(seeds_of=lambda m: [0, 1, 2, 3, 4])
    assert np.array_equal(first, mean_rank_scores(None, book, gs, 'gnnexplainer', 4))
    assert np.array_equal(rank_edges(first, 9), rank_edges(first, 9))


def test_disjoint_splits_counts():
    seeds = range(5)
    assert len(ae.disjoint_splits(seeds, 1, 1)) == 10
    assert len(ae.disjoint_splits(seeds, 2, 2)) == 15
    assert len(ae.disjoint_splits(seeds, 2, 3)) == 10
    for a, b in ae.disjoint_splits(seeds, 2, 2):
        assert not set(a) & set(b)


def test_split_kappa_identical_groups_is_one():
    s = np.arange(10.0)
    k = ae.split_kappa(s, s.copy(), (0,), (1,), (1, 3, 5), lambda key: 0)
    assert k == {1: 1.0, 3: 1.0, 5: 1.0}


def test_empty_candidates_node_is_handled_by_helpers():
    assert ae.ensemble_scores([np.array([], dtype=int)]).shape == (0,)
    # k_eff = m -> κ không xác định (NaN), không ném lỗi
    k = ae.split_kappa(np.array([1.0, 2.0]), np.array([2.0, 1.0]), (0,), (1,), (5,), lambda key: 0)
    assert np.isnan(k[5])


def test_pick_extreme_seed_direction():
    means = {0: 0.1, 1: 0.5, 2: -0.2}
    assert ae.pick_extreme_seed(means, 'fid_drop', 'best') == 1
    assert ae.pick_extreme_seed(means, 'fid_drop', 'worst') == 2
    assert ae.pick_extreme_seed(means, 'fid_inv', 'best') == 2     # Inv thấp hơn tốt hơn
    assert ae.pick_extreme_seed({0: float('nan')}, 'fid_drop', 'best') is None
