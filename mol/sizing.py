"""How many candidates does a round need for a target number of hits?

The audit intervals answer what a batch of a given size will contain.  A campaign asks the
question the other way round: a project needs a certain number of hits to proceed, and wants
to know how large the batch has to be.  Inverting the interval answers it.  The smallest
batch whose lower bound on confident hits reaches the target is the batch size that delivers
the target with the stated probability, and it is found by searching over batch size.

Estimates of campaign hit counts are already available by summing calibrated per-compound
probabilities, and they are expectations: the number of hits a batch will contain on
average.  A plan that has to hold needs the bound instead, and the two separate most where
screens operate.  At the in-class rates of a primary screen, a few percent, the expectation
divides the target by the rate and the bound does not, because at those rates the rate
itself is the uncertain quantity.

The audit enters the answer directly.  A larger audit pins the rate down and shrinks the
required batch, so audit measurements and synthesis are exchangeable against one another at a
rate this computes.  The exchange has a limit: the interval's width carries one term in the
audit size and one in the batch size, the first the uncertainty in the rate and the second
which candidates fall in this batch, so once the audit exceeds the batch the second term
dominates and further measurement buys little.

Looking more than once is optional continuation, which the fixed-sample interval does not
cover, so a campaign that re-estimates between rounds spends its confidence level across the
planned number of looks.  The price of that is computed here too.

Nothing in this is specific to a dataset: it is a property of the interval construction, and
the only inputs are the target, the observed in-class rate, the audit size and the
confidence level.

Usage: python3 -I sizing.py OUT.json
"""
import sys, os, json, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intervals

OUT = sys.argv[1]
G = 0.05
TARGETS = (20, 50)
RATES = (0.02, 0.05, 0.10, 0.25, 0.50)
AUDITS = (49, 100, 384, 1000)
LOOKS = (1, 2, 3, 6)
MMAX = 400000


def size_for(target, k, n, g):
    """smallest batch whose guaranteed hit count reaches the target"""
    if intervals.thatcher(k, n, MMAX, g)[0] < target:
        return None
    lo, hi = 1, MMAX
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals.thatcher(k, n, mid, g)[0] >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    res = {"confidence": 1 - G, "targets": list(TARGETS), "rates": list(RATES),
           "audits": list(AUDITS), "looks": list(LOOKS), "required": {}, "looks_cost": {},
           "variance_share": {}}

    for target in TARGETS:
        rows = []
        for rate in RATES:
            o = {"rate": rate, "expectation": int(round(target / rate))}
            for n in AUDITS:
                o[f"n{n}"] = size_for(target, int(round(rate * n)), n, G)
            rows.append(o)
        res["required"][str(target)] = rows
        print(f"--- target {target} hits, {100*(1-G):.0f}% confidence")
        print(f"{'rate':>6} {'expectation':>12} " + " ".join(f"{'n=' + str(n):>9}" for n in AUDITS))
        for o in rows:
            cells = " ".join(f"{(o[f'n{n}'] if o[f'n{n}'] else '>400k'):>9}" for n in AUDITS)
            print(f"{o['rate']:>6.2f} {o['expectation']:>12} " + cells)
        print()

    # price of re-estimating between rounds, at a declared number of looks
    for target in TARGETS:
        rows = []
        for rate in RATES:
            o = {"rate": rate}
            for j in LOOKS:
                o[f"looks{j}"] = size_for(target, int(round(rate * 384)), 384, G / j)
            rows.append(o)
        res["looks_cost"][str(target)] = rows
    print("--- price of looking more than once (target 20, audit 384)")
    print(f"{'rate':>6} " + " ".join(f"{str(j) + ' look(s)':>10}" for j in LOOKS))
    for o in res["looks_cost"]["20"]:
        print(f"{o['rate']:>6.2f} " + " ".join(f"{o[f'looks{j}']:>10}" for j in LOOKS))
    print()

    # where the interval's width comes from, as a share of its variance
    for m in (50, 500):
        rows = []
        for n in (25, 50, 100, 200, 400, 2000):
            e, a = 1.0 / n, 1.0 / m
            rows.append({"audit": n, "epistemic_share": e / (e + a),
                         "aleatoric_share": a / (e + a),
                         "half_width_p10": 2 * m * math.sqrt(0.10 * 0.90 * (e + a))})
        res["variance_share"][str(m)] = rows
    print("--- share of interval variance from the audit, outcome frequency 0.10")
    print(f"{'batch':>7} {'audit':>7} {'from audit':>11} {'from batch':>11}")
    for m, rows in res["variance_share"].items():
        for o in rows:
            print(f"{m:>7} {o['audit']:>7} {100*o['epistemic_share']:>10.0f}% "
                  f"{100*o['aleatoric_share']:>10.0f}%")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
