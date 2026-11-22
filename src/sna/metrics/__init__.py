"""DEPRECATED legacy metrics (Fidelity, Agreement, Ground Truth).

Kept only so midterm notebooks and ``tests/test_package.py`` still run.
These helpers differ from the final protocol: agreement returns 0.0 instead
of NaN, ground truth uses trapezoidal PR-AUC instead of average precision,
and fidelity treats an empty keep set differently. All final-term results
must use ``sna.evaluation.core`` (see KE_HOACH_CUOI_KY_FINAL.md, F4).
"""

import functools
import warnings

from . import fidelity as _fidelity
from . import agreement as _agreement
from . import ground_truth as _ground_truth


def _deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f'sna.metrics.{func.__name__} is deprecated; use sna.evaluation.core',
                      DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper


compute_fidelity = _deprecated(_fidelity.compute_fidelity)
compute_jaccard_at_k = _deprecated(_agreement.compute_jaccard_at_k)
compute_kendall_tau = _deprecated(_agreement.compute_kendall_tau)
compute_ground_truth_auc = _deprecated(_ground_truth.compute_ground_truth_auc)

__all__ = [
    'compute_fidelity',
    'compute_jaccard_at_k',
    'compute_kendall_tau',
    'compute_ground_truth_auc',
]
