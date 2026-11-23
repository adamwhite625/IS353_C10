"""Thống kê dùng chung cho FB-2/FB-3 (Mục 7.3-7.5, Phụ lục A).

Quy ước: độ lệch chuẩn mẫu ddof=1; bootstrap lấy mẫu lại trên đơn vị node
(B=2000, tin cậy 0,95); kiểm định cặp Wilcoxon signed-rank hai phía; hiệu chỉnh
Holm trong từng bảng. NaN được loại bỏ tường minh và số lượng còn lại luôn được
báo cáo. Mọi hàm xác định theo seed.
"""
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats as _st

__all__ = ['sample_sd', 'bootstrap_ci', 'paired_bootstrap_diff_ci',
           'wilcoxon_signed_rank', 'holm_adjust', 'compare_paired',
           'compare_many', 'spearman_with_n', 'summarize']

_NAN = float('nan')


def _vec(x, name='x'):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f'{name} phải là vector một chiều')
    return a


def _pair(a, b):
    a, b = _vec(a, 'a'), _vec(b, 'b')
    if len(a) != len(b):
        raise ValueError(f'a và b phải cùng độ dài node (nhận {len(a)} và {len(b)})')
    ok = ~(np.isnan(a) | np.isnan(b))
    return a[ok], b[ok]


def sample_sd(x):
    """Độ lệch chuẩn mẫu (ddof=1), bỏ NaN; NaN nếu còn dưới 2 giá trị."""
    a = _vec(x)
    a = a[~np.isnan(a)]
    return float(np.std(a, ddof=1)) if len(a) >= 2 else _NAN


def _boot_stats(v, statistic, n_boot, seed):
    """Thống kê trên n_boot mẫu lặp lại (chỉ số node rút có hoàn lại)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    samples = v[idx]
    if statistic is np.mean:
        return samples.mean(axis=1)
    if statistic is np.median:
        return np.median(samples, axis=1)
    return np.array([statistic(s) for s in samples], dtype=float)


def _ci_from(v, statistic, n_boot, confidence, seed):
    if not 0 < confidence < 1:
        raise ValueError('confidence phải nằm trong (0, 1)')
    if len(v) < 2:
        return {'estimate': float(statistic(v)) if len(v) else _NAN,
                'ci_low': _NAN, 'ci_high': _NAN, 'n': int(len(v))}
    dist = _boot_stats(v, statistic, n_boot, seed)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.nanpercentile(dist, [100 * alpha, 100 * (1 - alpha)])
    return {'estimate': float(statistic(v)), 'ci_low': float(lo),
            'ci_high': float(hi), 'n': int(len(v))}


def bootstrap_ci(values, statistic=np.mean, n_boot=2000, confidence=0.95, seed=0):
    """CI bootstrap phân vị trên đơn vị node. Bỏ NaN trước khi lấy mẫu.

    Trả dict: estimate, ci_low, ci_high, n (số node hợp lệ). n < 2 -> CI NaN.
    """
    a = _vec(values, 'values')
    return _ci_from(a[~np.isnan(a)], statistic, n_boot, confidence, seed)


def paired_bootstrap_diff_ci(a, b, statistic=np.mean, n_boot=2000, confidence=0.95, seed=0):
    """CI bootstrap cho thống kê của Δ(v)=a(v)-b(v); lấy lại CÙNG chỉ số node.

    Node có NaN ở a hoặc b bị loại; n là số node còn lại.
    """
    a, b = _pair(a, b)
    return _ci_from(a - b, statistic, n_boot, confidence, seed)


def wilcoxon_signed_rank(a, b, zero_method='wilcox', min_nonzero=2):
    """Wilcoxon signed-rank hai phía trên các cặp không NaN.

    zero_method mặc định 'wilcox' (Wilcoxon 1945): bỏ cặp có hiệu bằng 0 trước khi
    xếp hạng, vì cặp hoà không mang thông tin về hướng; khác 'pratt' (giữ hạng của
    số 0) và 'zsplit'. Phương pháp tính p để scipy tự chọn (exact khi n nhỏ, không
    hoà, không có 0; ngược lại xấp xỉ chuẩn có hiệu chỉnh hoà).
    Trả dict: statistic, p, n (cặp hợp lệ), n_nonzero, reason (None nếu tính được).
    Nếu mọi hiệu = 0 hoặc n_nonzero < min_nonzero thì p = NaN kèm reason, không ném lỗi.
    """
    a, b = _pair(a, b)
    d = a - b
    n_nz = int(np.count_nonzero(d))
    out = {'statistic': _NAN, 'p': _NAN, 'n': int(len(d)), 'n_nonzero': n_nz, 'reason': None}
    if len(d) == 0:
        out['reason'] = 'không có cặp hợp lệ (toàn NaN)'
    elif n_nz == 0:
        out['reason'] = 'mọi hiệu bằng 0'
    elif n_nz < min_nonzero:
        out['reason'] = f'quá ít hiệu khác 0 ({n_nz} < {min_nonzero})'
    if out['reason'] is None:
        res = _st.wilcoxon(a, b, zero_method=zero_method, alternative='two-sided')
        out['statistic'], out['p'] = float(res.statistic), float(res.pvalue)
    return out


def _rank_biserial(d):
    """(W+ - W-)/(W+ + W-) trên hiệu khác 0, thuộc [-1, 1]; NaN nếu không có."""
    d = d[d != 0]
    if len(d) == 0:
        return _NAN
    r = _st.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def holm_adjust(pvalues):
    """p đã hiệu chỉnh Holm step-down (đơn điệu, cắt tại 1), giữ thứ tự đầu vào.

    NaN bị bỏ qua (không tính vào m) và ra NaN.
    """
    p = _vec(pvalues, 'pvalues')
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok], kind='stable')]
    m = len(order)
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def compare_paired(a, b, *, seed, n_boot=2000):
    """So sánh cặp theo node cho Bảng A2-2 (a - b).

    Khóa: n, mean_a, mean_b, mean_diff, ci_low, ci_high (bootstrap của hiệu trung
    bình), p_raw (Wilcoxon), effect (hiệu trung vị), rank_biserial, wilcoxon_reason.
    mean_a/mean_b tính trên cùng tập node hợp lệ của cả hai.
    """
    a, b = _pair(a, b)
    ci = _ci_from(a - b, np.mean, n_boot, 0.95, seed)
    w = wilcoxon_signed_rank(a, b)
    return {'n': int(len(a)),
            'mean_a': float(a.mean()) if len(a) else _NAN,
            'mean_b': float(b.mean()) if len(b) else _NAN,
            'mean_diff': ci['estimate'], 'ci_low': ci['ci_low'], 'ci_high': ci['ci_high'],
            'p_raw': w['p'],
            'effect': float(np.median(a - b)) if len(a) else _NAN,
            'rank_biserial': _rank_biserial(a - b),
            'wilcoxon_reason': w['reason']}


def compare_many(table, *, seed, n_boot=2000):
    """Nhiều phép so sánh {tên: (a, b)} -> DataFrame (cột 'name', compare_paired, 'p_holm').

    Mọi phép dùng chung seed (kết quả mỗi phép không phụ thuộc thứ tự bảng);
    Holm áp trên toàn bộ các phép trong bảng này.
    """
    items = list(table.items()) if isinstance(table, Mapping) else list(table)
    rows = []
    for name, (a, b) in items:
        rows.append({'name': name, **compare_paired(a, b, seed=seed, n_boot=n_boot)})
    df = pd.DataFrame(rows)
    if df.empty:
        df['p_holm'] = []
        return df
    df['p_holm'] = holm_adjust(df['p_raw'].to_numpy())
    return df


def spearman_with_n(x, y):
    """Spearman bỏ cặp NaN, trả (rho, p, n); NaN khi n < 3 hoặc một vector hằng."""
    x, y = _pair(x, y)
    n = int(len(x))
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return _NAN, _NAN, n
    res = _st.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue), n


def summarize(values):
    """mean, sd (ddof=1), median, n (không NaN), n_nan."""
    a = _vec(values, 'values')
    nan = np.isnan(a)
    v = a[~nan]
    return {'mean': float(v.mean()) if len(v) else _NAN, 'sd': sample_sd(v),
            'median': float(np.median(v)) if len(v) else _NAN,
            'n': int(len(v)), 'n_nan': int(nan.sum())}
