"""Fast exact prediction limits used throughout (paper conventions).

Lower limit: smallest j with F(j) >  gamma/2   (exact inversion of the one-sided test)
Upper limit: smallest j with F(j) >= 1-gamma/2 (quantile)
for the Beta-Binomial distributions stated in the text."""
import numpy as np
from scipy.stats import betabinom

def _cdf(m, a, b):
    return np.cumsum(betabinom.pmf(np.arange(m + 1), m, a, b))

def lower(m, a, b, g):
    return int(np.searchsorted(_cdf(m, a, b), g / 2, side="right"))

def upper(m, a, b, g):
    c = _cdf(m, a, b); return int(min(np.searchsorted(c, 1 - g / 2 - 1e-12, side="left"), m))

def thatcher(k, n, m, g):
    """future count of an event seen k times among n exchangeable compounds"""
    lo = 0 if k == 0 else lower(m, k, n - k + 1, g)
    hi = m if k == n else upper(m, k + 1, n - k, g)
    return lo, hi

def rank_law(k, n, m, g):
    """number of m future own-class scores at or below the k-th of n calibration scores"""
    return lower(m, k, n + 1 - k, g), upper(m, k, n + 1 - k, g)
