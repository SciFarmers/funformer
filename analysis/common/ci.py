"""Confidence interval on a mean, using Student t rather than 1.96.

Every interval in this project is a mean over a small sample -- three
directions in the chicken bin, six among mammals, fourteen across the fish
split. 1.96 is the normal quantile and is only correct as n grows; at n=3 the
right multiplier is t(2) = 4.30, which is more than twice as wide.

Using it everywhere, including where n is large enough that the difference is
invisible, means the choice never has to be revisited per call site.
"""
import numpy as np
from scipy import stats


def mean_ci(x, conf=0.95):
    """(mean, lo, hi) for a 1-D sample, two-sided, Student t with n-1 df.

    Returns (mean, nan, nan) for n < 2, where the interval is undefined.
    """
    a = np.asarray([v for v in np.ravel(np.asarray(x, dtype=float))
                    if np.isfinite(v)])
    n = a.size
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(a.mean())
    if n < 2:
        return m, np.nan, np.nan
    se = a.std(ddof=1) / np.sqrt(n)
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1)) * se
    return m, m - h, m + h
