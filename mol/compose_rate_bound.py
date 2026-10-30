"""Compose a rate bound with a binomial tail, and compare it with the exact inversion.

A referee's natural objection to "a rate bound does not give you a count" is that you can
always compose one: bound the rate from the audit at level g/2, then take a binomial upper
quantile at that rate at level g/2, and union.  That is valid.  It also pays twice.  This
measures what it costs against conditioning on the audit-plus-batch total and paying once.

Both sides are one-sided upper bounds at total level g = 0.05, and the rate bound is given the
most generous instrument available: the exact Clopper-Pearson upper limit, not the
Hoeffding-Bentkus or betting bounds that risk-controlling prediction sets actually use, which
are looser at these sample sizes.
"""
import sys
sys.path.insert(0, "/home/claude/conformal-screening/mol")
sys.path.insert(0, "/home/claude/conformal-screening/code")
from scipy.stats import beta, binom
from intervals import thatcher

G = 0.05


def exact_upper(k, n, m, g=G):
    """upper end of the exact interval with the whole of g in the upper tail"""
    return thatcher(k, n, m, 2 * g)[1]


def composed_upper(k, n, m, g=G):
    """Clopper-Pearson upper on the rate at g/2, then a binomial upper quantile at g/2"""
    pu = beta.ppf(1 - g / 2, k + 1, n - k) if k < n else 1.0
    return int(binom.ppf(1 - g / 2, m, pu))


rows = [
    ("primary screen, plate",        384, 96,  8),
    ("primary screen, per 1000",     384, 1000, 8),
    ("solubility audit, per 1000",   384, 1000, 89),
    ("plate priced by a quarter slice", 24, 72, 2),
    ("electrolyte, one plate",       364, 96,  30),
    ("small audit, large batch",      49, 1000, 1),
    ("audit the size of the batch",  400, 400, 80),
]
print(f"{'configuration':34}{'n':>5}{'m':>6}{'k':>5}"
      f"{'exact':>8}{'composed':>10}{'ratio':>8}")
for name, n, m, k in rows:
    e, c = exact_upper(k, n, m), composed_upper(k, n, m)
    pt = m * k / n
    print(f"{name:34}{n:>5}{m:>6}{k:>5}{e:>8}{c:>10}{c/max(e,1):>8.2f}")
print()
print("excess of the composed bound over the exact one, as a share of the point estimate:")
for name, n, m, k in rows:
    e, c = exact_upper(k, n, m), composed_upper(k, n, m)
    pt = m * k / n
    print(f"  {name:34} point {pt:7.1f}   exact +{e-pt:6.1f}   composed +{c-pt:6.1f}")
