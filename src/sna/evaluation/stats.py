"""Shared statistics for the final protocol (FB-2 / FB-3).

Conventions (configs/protocol_final.yaml, ``statistics``):
- the unit of resampling is the node; NaN values are dropped (pairwise for
  paired tests) and the number of valid nodes is always reported;
- sample SD uses ddof=1;
- bootstrap CI is the percentile interval of the mean, B = 2000, 95 %;
- paired comparisons use Wilcoxon signed-rank, then Holm within one table.
Every function is deterministic given ``seed``.
"""
import math
import numpy as np
from scipy.stats import wilcoxon

BOOTSTRAP_REPEATS = 2000
CONFIDENCE = 0.95


def _finite(x):
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _paired(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError('Paired inputs must have the same length (one value per node)')
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def bootstrap_ci(x, repeats=BOOTSTRAP_REPEATS, confidence=CONFIDENCE, seed=0):
    """Percentile bootstrap CI of the mean over nodes. Returns (lo, hi).

    NaN when fewer than two valid values (no sampling variability to estimate).
    """
    a = _finite(x)
    if len(a) < 2:
        return float('nan'), float('nan')
    if not 0 < confidence < 1 or repeats < 1:
        raise ValueError('confidence must be in (0, 1) and repeats >= 1')
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(int(repeats), len(a)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def summarize(x, repeats=BOOTSTRAP_REPEATS, confidence=CONFIDENCE, seed=0):
    """mean, sample SD (ddof=1), valid n, NaN count and bootstrap CI of the mean."""
    raw = np.asarray(x, dtype=float).ravel()
    a = raw[np.isfinite(raw)]
    lo, hi = bootstrap_ci(a, repeats, confidence, seed)
    return {'mean': float(a.mean()) if len(a) else float('nan'),
            'sd': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
            'n': int(len(a)), 'n_nan': int(len(raw) - len(a)),
            'ci_lo': lo, 'ci_hi': hi}


def paired_bootstrap_ci(a, b, repeats=BOOTSTRAP_REPEATS, confidence=CONFIDENCE, seed=0):
    """CI of mean(a - b), resampling nodes; pairs with any NaN are dropped."""
    a, b = _paired(a, b)
    return bootstrap_ci(a - b, repeats, confidence, seed)


def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank on per-node pairs.

    Zero differences are dropped (``zero_method='wilcox'``). Returns statistic,
    p, n (valid pairs), n_nonzero and a reason when the test is undefined.
    """
    a, b = _paired(a, b)
    d = a - b
    nonzero = int(np.count_nonzero(d))
    out = {'statistic': float('nan'), 'p': float('nan'), 'n': int(len(d)),
           'n_nonzero': nonzero, 'reason': ''}
    if len(d) == 0:
        out['reason'] = 'no_valid_pairs'
    elif nonzero == 0:
        out.update(p=1.0, reason='all_differences_zero')
    else:
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        out.update(statistic=float(res.statistic), p=float(res.pvalue))
    return out


def holm(pvalues):
    """Holm step-down adjusted p-values, in input order.

    NaN p-values stay NaN and do not count towards the number of tests.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if np.any((p[idx] < 0) | (p[idx] > 1)):
        raise ValueError('p-values must lie in [0, 1]')
    m = len(idx)
    order = idx[np.argsort(p[idx], kind='stable')]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def paired_compare(a, b, repeats=BOOTSTRAP_REPEATS, confidence=CONFIDENCE, seed=0):
    """Effect size and test for one row of a paired table: Δ = mean(a - b).

    Holm adjustment is applied afterwards across the rows of a table with
    :func:`holm`.
    """
    pa, pb = _paired(a, b)
    lo, hi = bootstrap_ci(pa - pb, repeats, confidence, seed)
    test = wilcoxon_paired(pa, pb)
    return {'delta': float((pa - pb).mean()) if len(pa) else float('nan'),
            'ci_lo': lo, 'ci_hi': hi, 'p': test['p'], 'n': test['n'],
            'n_nonzero': test['n_nonzero'], 'reason': test['reason']}


def ci_excludes_zero(lo, hi):
    """True when a CI lies strictly on one side of 0 (NaN-safe)."""
    return bool(math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0))
