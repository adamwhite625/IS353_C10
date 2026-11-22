"""F4: kappa@k checks from KE_HOACH_CUOI_KY_FINAL.md (Appendix A)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import math
import warnings
import numpy as np
import pytest
from sna.evaluation.core import agreement, kappa_at_k


def test_identical_selection_is_one():
    assert kappa_at_k({0, 1, 2}, {0, 1, 2}, 10) == (pytest.approx(1.0), '')
    scores = np.arange(10.0)
    assert agreement(scores, scores, 5)['kappa'] == pytest.approx(1.0)


def test_undefined_cases_are_nan_with_reason():
    kappa, reason = kappa_at_k(set(), set(), 0)
    assert math.isnan(kappa) and reason == 'empty_candidates'
    kappa, reason = kappa_at_k({0, 1, 2}, {0, 1, 2}, 3)
    assert math.isnan(kappa) and reason == 'k_eff_equals_m'
    # agreement() clips k to m, so k > m also yields k_eff = m.
    out = agreement([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], 5)
    assert math.isnan(out['kappa']) and out['kappa_reason'] == 'k_eff_equals_m'
    assert math.isnan(agreement([], [], 0)['kappa'])


def test_known_value_and_below_chance():
    # k=2, m=4: expected overlap 1, so disjoint -> -1, overlap 1 -> 0.
    assert kappa_at_k({0, 1}, {2, 3}, 4) == (pytest.approx(-1.0), '')
    assert kappa_at_k({0, 1}, {1, 2}, 4) == (pytest.approx(0.0), '')
    with pytest.raises(ValueError):
        kappa_at_k({0, 1}, {0}, 4)


def test_random_masks_average_near_zero():
    rng = np.random.default_rng(0)
    values = [agreement(rng.random(40), rng.random(40), 5)['kappa'] for _ in range(4000)]
    assert abs(np.mean(values)) < 0.02


def test_legacy_metrics_warn_deprecated():
    import sna.metrics as legacy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        legacy.compute_jaccard_at_k([1.0, 0.0], [1.0, 0.0], 1)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
