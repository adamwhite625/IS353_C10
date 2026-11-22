"""Shared statistics helpers (bootstrap, Wilcoxon, Holm)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import math
import numpy as np
import pytest
from scipy.stats import wilcoxon
from sna.evaluation.stats import (bootstrap_ci, summarize, paired_bootstrap_ci,
                                  wilcoxon_paired, holm, paired_compare, ci_excludes_zero)


def test_bootstrap_is_seeded_and_brackets_mean():
    x = np.random.default_rng(1).normal(0.3, 1.0, 140)
    assert bootstrap_ci(x, seed=7) == bootstrap_ci(x, seed=7)
    assert bootstrap_ci(x, seed=7) != bootstrap_ci(x, seed=8)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    # Roughly the normal-theory 95 % interval for n = 140.
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    assert hi - lo == pytest.approx(2 * half, rel=0.15)


def test_constant_and_degenerate_inputs():
    assert bootstrap_ci([0.5] * 10) == (pytest.approx(0.5), pytest.approx(0.5))
    assert all(math.isnan(v) for v in bootstrap_ci([1.0]))
    s = summarize([1.0, float('nan'), 3.0])
    assert (s['mean'], s['n'], s['n_nan']) == (2.0, 2, 1)
    assert s['sd'] == pytest.approx(math.sqrt(2))  # ddof = 1
    empty = summarize([float('nan')])
    assert empty['n'] == 0 and math.isnan(empty['mean']) and math.isnan(empty['sd'])


def test_paired_drops_nan_pairs():
    a = [1.0, 2.0, float('nan'), 4.0]
    b = [0.0, float('nan'), 1.0, 1.0]
    lo, hi = paired_bootstrap_ci(a, b)       # valid diffs: 1, 3
    assert 1.0 <= lo <= hi <= 3.0
    assert wilcoxon_paired(a, b)['n'] == 2
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_wilcoxon_matches_scipy_and_edge_cases():
    rng = np.random.default_rng(3)
    a, b = rng.random(50), rng.random(50)
    assert wilcoxon_paired(a, b)['p'] == pytest.approx(wilcoxon(a, b).pvalue)
    same = wilcoxon_paired([1.0, 2.0], [1.0, 2.0])
    assert same['p'] == 1.0 and same['reason'] == 'all_differences_zero'
    none = wilcoxon_paired([float('nan')], [1.0])
    assert math.isnan(none['p']) and none['reason'] == 'no_valid_pairs'


def test_holm_hand_example_and_nan():
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    out = holm([0.02, float('nan'), 0.6])     # m = 2
    assert out[0] == pytest.approx(0.04) and math.isnan(out[1]) and out[2] == pytest.approx(0.6)
    assert holm([0.9, 0.8]).max() <= 1.0
    with pytest.raises(ValueError):
        holm([1.5])


def test_defaults_match_locked_protocol():
    from sna.core.provenance import protocol
    from sna.evaluation import stats
    spec = protocol()['statistics']
    assert spec['bootstrap'] == {'unit': 'node', 'repeats': stats.BOOTSTRAP_REPEATS,
                                 'confidence': stats.CONFIDENCE}
    assert spec['sample_sd_ddof'] == 1
    assert (spec['paired_test'], spec['adjustment']) == ('wilcoxon_signed_rank', 'holm')


def test_paired_compare_detects_shift():
    rng = np.random.default_rng(0)
    b = rng.random(140)
    r = paired_compare(b + 0.1 + rng.normal(0, 0.02, 140), b)
    assert r['delta'] == pytest.approx(0.1, abs=0.01)
    assert ci_excludes_zero(r['ci_lo'], r['ci_hi']) and r['p'] < 1e-6
    assert not ci_excludes_zero(float('nan'), 1.0)
