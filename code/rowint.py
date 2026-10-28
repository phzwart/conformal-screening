"""Calibration-only and test-set prediction intervals for outcome-table cells.

Calibration-only (class-conditional, given m_y future compounds of class y):
  A_y: own-class coverage event   -> exact rank law  BetaBinomial(m_y; k_y, n_y+1-k_y)
  B_y: cross-class labelling event -> Thatcher interval from c_y of n_y calibration compounds
Both intervals at level gamma/2, so they hold simultaneously with prob >= 1-gamma, and every
cell of the row is bounded by arithmetic on |A_y| and |B_y|, whichever regime applies:
  two-label regime (A^c subset of B): other_single = A^c, own_single = B^c, both = rest
  empty regime     (A and B disjoint): own_single = A,   other_single = B,   empty = rest
Mixed batch of m: union over m_1 in its interval (level gamma/2), each of the four events
(A_0, B_0, A_1, B_1) at gamma/8, so all cells of the mixed batch hold jointly at 1 - gamma."""
import numpy as np
from intervals import thatcher, rank_law

def row_intervals(n, k, c, m, two_label, g):
    aL, aU = rank_law(k, n, m, g / 2); bL, bU = thatcher(c, n, m, g / 2)
    if two_label:
        return {"own_single": (m - bU, m - bL), "other_single": (m - aU, m - aL),
                "middle": (max(0, aL + bL - m), min(m, aU + bU - m))}
    return {"own_single": (aL, aU), "other_single": (bL, bU), "middle": (max(0, m - aU - bU), m - aL - bL)}

def cell_names(y, two_label):
    own, other = f"{y}_s{y}", f"{y}_s{1 - y}"
    return {"own_single": own, "other_single": other, "middle": f"{y}_both" if two_label else f"{y}_empty"}

def calonly_class(rec_m, y, n, m, g=0.05):
    k, c = (rec_m["k0"], rec_m["c0"]) if y == 0 else (rec_m["k1"], rec_m["c1"])
    iv = row_intervals(n, k, c, m, rec_m["two_label"], g); nm = cell_names(y, rec_m["two_label"])
    return {nm[key]: iv[key] for key in iv}

def calonly_mixed(rec_m, n0, n1, m, g=0.05):
    m1L, m1U = thatcher(n1, n0 + n1, m, g / 2); out = {}
    for y in (0, 1):
        res = None
        for m1 in range(m1L, m1U + 1):
            my = m1 if y == 1 else m - m1
            iv = calonly_class(rec_m, y, n1 if y == 1 else n0, my, g / 4)   # gamma/8 per event: 4 events + m1 = gamma
            res = iv if res is None else {k: (min(res[k][0], v[0]), max(res[k][1], v[1])) for k, v in iv.items()}
        out.update(res)
    return out

def testset_mixed(test_counts, n_test, m, g=0.05):
    return {k: thatcher(v, n_test, m, g) for k, v in test_counts.items()}
