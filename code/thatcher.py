"""Numerical check of the identities behind the exact prediction limits.
Given K + K_m = T, K ~ Hypergeometric(n+m, T, n). For all 0<=k<=n, 0<=j<=m:
  lower tail: P(K >= k | T = k+j) = P(B <= j),  B ~ BetaBinomial(m; k, n-k+1)   (k >= 1)
  upper tail: P(K <= k | T = k+j) = P(B' >= j), B' ~ BetaBinomial(m; k+1, n-k)  (k <= n-1)"""
from scipy.stats import hypergeom, betabinom
import numpy as np
err_lo = err_hi = 0.0
for n in [20, 57, 200]:
    for m in [10, 100, 1000]:
        j = np.arange(m + 1)
        for k in range(n + 1):
            T = k + j
            if k >= 1:
                err_lo = max(err_lo, np.max(np.abs(hypergeom.sf(k - 1, n + m, T, n) - betabinom.cdf(j, m, k, n - k + 1))))
            if k <= n - 1:
                err_hi = max(err_hi, np.max(np.abs(hypergeom.cdf(k, n + m, T, n) - betabinom.sf(j - 1, m, k + 1, n - k))))
print(f"max error lower-tail identity: {err_lo:.2e}")
print(f"max error upper-tail identity: {err_hi:.2e}")
