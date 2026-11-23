"""F4: κ@k hiệu chỉnh theo mức trùng ngẫu nhiên, và tích hợp vào agreement()."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import math
import warnings
import numpy as np
import pytest
from sna.evaluation.core import kappa_at_k, agreement, rank_edges


def test_identical_sets_give_one():
    out = kappa_at_k([3, 1, 4], [4, 3, 1], 3, 10)
    assert out == {'kappa': 1.0, 'kappa_reason': ''}


def test_k_equals_m_is_nan_with_reason():
    out = kappa_at_k([0, 1, 2], [2, 1, 0], 3, 3)
    assert math.isnan(out['kappa']) and out['kappa_reason']
    # k > m: k_eff = m, cũng không xác định
    out = kappa_at_k([0, 1, 2], [2, 1, 0], 10, 3)
    assert math.isnan(out['kappa']) and out['kappa_reason'] == 'selection_equals_all_candidates'


def test_no_candidates_is_nan_with_reason():
    out = kappa_at_k([], [], 5, 0)
    assert math.isnan(out['kappa']) and out['kappa_reason'] == 'no_candidates'
    out = kappa_at_k([], [], 0, 4)
    assert math.isnan(out['kappa']) and out['kappa_reason'] == 'zero_budget'


def test_hand_computed_values_including_negative():
    # m = 4, k = 2: kỳ vọng ngẫu nhiên k²/m = 1; mẫu số = 2 - 1 = 1
    assert kappa_at_k([0, 1], [0, 1], 2, 4)['kappa'] == pytest.approx(1.0)
    assert kappa_at_k([0, 1], [1, 2], 2, 4)['kappa'] == pytest.approx(0.0)
    assert kappa_at_k([0, 1], [2, 3], 2, 4)['kappa'] == pytest.approx(-1.0)  # rời nhau: kém ngẫu nhiên
    # m = 10, k = 5: k²/m = 2.5, mẫu số 2.5; giao 1 -> (1 - 2.5)/2.5 = -0.6
    assert kappa_at_k(range(5), [4, 5, 6, 7, 8], 5, 10)['kappa'] == pytest.approx(-0.6)
    # m = 10, k = 3: k²/m = 0.9; giao 2 -> (2 - 0.9)/(3 - 0.9)
    assert kappa_at_k([0, 1, 2], [0, 1, 9], 3, 10)['kappa'] == pytest.approx(1.1 / 2.1)


def test_k_larger_than_m_uses_k_eff():
    # k = 7 > m = 4 nhưng chỉ có k_eff = 4 = m -> NaN; k = 3 < m = 4 thì xác định
    assert math.isnan(kappa_at_k([0, 1, 2, 3], [3, 2, 1, 0], 7, 4)['kappa'])
    # m = 6, k = 100 -> k_eff = 6 = m -> NaN
    assert math.isnan(kappa_at_k(range(6), range(6), 100, 6)['kappa'])
    # Dùng cắt k_eff phần tử đầu: lấy 2 phần tử đầu của danh sách dài hơn
    assert kappa_at_k([0, 1, 2, 3], [0, 1, 3, 2], 2, 5)['kappa'] == pytest.approx(1.0)


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        kappa_at_k([0, 0], [0, 1], 2, 5)  # trùng chỉ số
    with pytest.raises(ValueError):
        kappa_at_k([0, 7], [0, 1], 2, 5)  # ngoài [0, m)
    with pytest.raises(ValueError):
        kappa_at_k([0], [0], -1, 5)


def test_random_masks_average_kappa_near_zero():
    rng = np.random.default_rng(123)
    m, k = 200, 20
    values = [kappa_at_k(rng.permutation(m)[:k], rng.permutation(m)[:k], k, m)['kappa']
              for _ in range(4000)]
    assert abs(np.mean(values)) < 0.02
    # Jaccard của cùng cấu hình KHÔNG ≈ 0 (đó là lý do dùng κ)
    jac = []
    for _ in range(500):
        a, b = set(rng.permutation(m)[:k]), set(rng.permutation(m)[:k])
        jac.append(len(a & b) / len(a | b))
    assert np.mean(jac) > 0.03


def test_agreement_adds_kappa_without_changing_existing_keys():
    a = np.array([5., 4, 3, 2, 1, 0])
    b = np.array([5., 3, 4, 0, 1, 2])
    out = agreement(a, b, 3)
    assert {'jaccard', 'kendall_tau_b', 'tau_reason', 'jaccard_reason', 'kappa', 'kappa_reason'} <= set(out)
    # top-3 của a = {0,1,2}, của b = {0,1,2}: giao 3, k²/m = 1.5 -> κ = 1
    assert out['jaccard'] == 1.0 and out['kappa'] == pytest.approx(1.0)
    c = np.array([0., 1, 2, 3, 4, 5])
    out = agreement(a, c, 2)
    assert out['jaccard'] == 0.0
    assert out['kappa'] == pytest.approx((0 - 4 / 6) / (2 - 4 / 6))
    # k = m -> NaN kèm lý do, các khóa cũ vẫn có giá trị
    out = agreement(a, b, 6)
    assert math.isnan(out['kappa']) and out['kappa_reason'] and out['jaccard'] == 1.0
    # m = 0
    out = agreement([], [], 0)
    assert math.isnan(out['kappa']) and out['kappa_reason'] == 'no_candidates'
    assert math.isnan(out['jaccard']) and out['jaccard_reason'] == 'empty_selected_union'


def test_agreement_kappa_with_ties_uses_given_orders():
    s = np.zeros(8)
    oa, ob = rank_edges(s, 1), rank_edges(s, 1)
    assert agreement(s, s, 3, order_a=oa, order_b=ob)['kappa'] == pytest.approx(1.0)


def test_metrics_package_is_deprecated_but_still_works():
    import sna
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = sna.compute_jaccard_at_k([3, 2, 1], [3, 1, 2], k=2)
    assert value == pytest.approx(1 / 3)  # top-2: {0, 1} và {0, 2}
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
