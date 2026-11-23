import numpy as np
import pytest
from scipy import stats as st

from sna.evaluation.stats import (bootstrap_ci, compare_many, compare_paired, holm_adjust,
                                  paired_bootstrap_diff_ci, sample_sd, spearman_with_n,
                                  summarize, wilcoxon_signed_rank)


def test_sample_sd():
    assert sample_sd([1, 2, 3, 4]) == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert sample_sd([1, np.nan, 3]) == pytest.approx(np.sqrt(2))
    assert np.isnan(sample_sd([5.0])) and np.isnan(sample_sd([np.nan, 1.0]))


def test_holm_hand_example():
    out = holm_adjust([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(out, [0.03, 0.06, 0.06, 0.02])


def test_holm_nan_clip_monotone():
    out = holm_adjust([0.5, np.nan, 0.4, 0.9])
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2, 3]], [1.0, 1.0, 1.0])
    assert holm_adjust([np.nan]).shape == (1,)


def test_bootstrap_deterministic_and_nan():
    x = np.random.default_rng(1).normal(size=200)
    a = bootstrap_ci(x, seed=3)
    assert a == bootstrap_ci(x, seed=3)
    assert a != bootstrap_ci(x, seed=4)
    xn = np.append(x, [np.nan, np.nan])
    assert bootstrap_ci(xn, seed=3) == a
    assert a['n'] == 200 and a['ci_low'] < a['estimate'] < a['ci_high']
    assert np.isnan(bootstrap_ci([1.0], seed=0)['ci_low'])


def test_bootstrap_coverage():
    rng = np.random.default_rng(0)
    hit = 0
    reps = 150
    for i in range(reps):
        ci = bootstrap_ci(rng.normal(2.0, 1.0, 80), n_boot=300, seed=i)
        hit += ci['ci_low'] <= 2.0 <= ci['ci_high']
    assert 0.88 <= hit / reps <= 0.99


def test_paired_diff_nan_and_same_indices():
    rng = np.random.default_rng(2)
    b = rng.normal(size=100)
    a = b + 0.5 + rng.normal(scale=0.1, size=100)
    a2, b2 = a.copy(), b.copy()
    a2[3] = np.nan
    b2[7] = np.nan
    r = paired_bootstrap_diff_ci(a2, b2, seed=0)
    assert r['n'] == 98
    keep = np.ones(100, bool)
    keep[[3, 7]] = False
    assert r == paired_bootstrap_diff_ci(a[keep], b[keep], seed=0)
    # lấy mẫu cặp: CI hẹp (lấy mẫu độc lập sẽ rộng hơn nhiều)
    assert r['ci_high'] - r['ci_low'] < 0.1
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([1, 2], [1], seed=0)


def test_wilcoxon_matches_scipy_and_degenerate():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=40), rng.normal(size=40) - 0.3
    w = wilcoxon_signed_rank(a, b)
    ref = st.wilcoxon(a, b)
    assert w['p'] == pytest.approx(ref.pvalue)
    assert w['statistic'] == pytest.approx(ref.statistic)
    z = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert np.isnan(z['p']) and z['reason'] == 'mọi hiệu bằng 0'
    assert np.isnan(wilcoxon_signed_rank([np.nan], [1.0])['p'])
    assert np.isnan(wilcoxon_signed_rank([1, 2], [1, 3])['p'])  # chỉ 1 hiệu khác 0
    a0, b0 = np.append(a, [0.0, 1.0]), np.append(b, [0.0, 1.0])
    assert wilcoxon_signed_rank(a0, b0)['p'] == pytest.approx(
        st.wilcoxon(a0, b0, zero_method='wilcox').pvalue)


def test_compare_paired_and_many():
    rng = np.random.default_rng(7)
    b = rng.uniform(size=60)
    a = b + 0.2 + rng.normal(scale=0.05, size=60)
    r = compare_paired(a, b, seed=1)
    assert r['n'] == 60 and r['mean_diff'] == pytest.approx(r['mean_a'] - r['mean_b'])
    assert r['ci_low'] > 0 and r['p_raw'] < 1e-6 and r['rank_biserial'] == pytest.approx(1.0)
    assert r['effect'] == pytest.approx(np.median(a - b))
    table = {'AB': (a, b), 'AA': (a, a), 'BC': (b, b + rng.normal(size=60))}
    df = compare_many(table, seed=1)
    assert list(df['name']) == ['AB', 'AA', 'BC']
    np.testing.assert_allclose(df['p_holm'], holm_adjust(df['p_raw']), equal_nan=True)
    assert np.isnan(df.loc[1, 'p_holm']) and df.loc[1, 'wilcoxon_reason'] == 'mọi hiệu bằng 0'
    assert compare_many({}, seed=0).empty


def test_permutation_invariance():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=50), rng.normal(size=50)
    perm = rng.permutation(50)
    r1, r2 = compare_paired(a, b, seed=0), compare_paired(a[perm], b[perm], seed=0)
    for k in ('n', 'mean_a', 'mean_diff', 'p_raw', 'effect', 'rank_biserial'):
        assert r1[k] == pytest.approx(r2[k])
    # CI bootstrap: cùng phân phối, thực hiện cụ thể khác -> gần nhau
    assert r1['ci_low'] == pytest.approx(r2['ci_low'], abs=0.1)


def test_spearman_and_summarize():
    x = np.arange(10.0)
    rho, p, n = spearman_with_n(x, x ** 2)
    assert rho == pytest.approx(1.0) and n == 10
    y = x.copy()
    y[2] = np.nan
    assert spearman_with_n(x, y)[2] == 9
    assert np.isnan(spearman_with_n([1, 2], [1, 2])[0])
    assert np.isnan(spearman_with_n(x, np.ones(10))[0])
    s = summarize([1, 2, 3, np.nan])
    assert s == {'mean': 2.0, 'sd': 1.0, 'median': 2.0, 'n': 3, 'n_nan': 1}
    assert summarize([np.nan])['n'] == 0
