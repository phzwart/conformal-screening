"""Is the coverage failure a shift in prevalence, or in the class-conditional rates?

When a batch introduces chemistry the audit does not hold, two things can move at once.
The share of in-class candidates can change, and the rule's behaviour within each class can
change.  The first is a prevalence shift and is in principle correctable: the counts the
audit implies are rescaled to the batch's own class sizes.  The second is not, because it
means the mapping from score to outcome has moved and no rescaling of totals repairs it.

This separates them.  Each outcome cell belongs to one class: confident hits and missed
binders are in-class cells, false leads and correct rejections out-class ones.  The audit
gives the rate of each cell within its class, and the published interval carries that rate
to the batch through the batch's total size.  The rescaled interval carries it through the
batch's own class size instead.  That second calculation uses the realized class counts, so
it is an oracle and is not a method: it cannot be run before the batch is assayed, and it is
computed here only to ask how much of the failure prevalence accounts for.

If the oracle restores coverage, the failure is a prevalence shift and the remedy is to
estimate the batch's class sizes rather than to measure its chemistry.  If it does not, the
rule's behaviour within a class has changed and the pilot conclusion stands.

Usage: python3 -I rescale_check.py RUNS.jsonl [OUT.json]
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intervals

RUNS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
G = 0.05
# outcome cell -> which class its members belong to
CELLS = (("confident hit", 1), ("missed binder", 1),
         ("false lead", 0), ("correct rejection", 0))
DESIGNS = ("random", "block3", "block2", "block1")


def main():
    rows = [json.loads(l) for l in open(RUNS)]
    rows = [r for r in rows if not r.get("skipped") and r["label"] == "activity"
            and r["score"] == "model"]
    res = {"confidence": 1 - G, "designs": {}}
    print("coverage of the four class-attributable outcome lines")
    print(f"{'design':8} {'reps':>5} {'as published':>13} {'rescaled (oracle)':>19} "
          f"{'prevalence ratio':>17}")
    for d in DESIGNS:
        v = [r for r in rows if r["design"] == d]
        if not v:
            continue
        pub_in = pub_n = res_in = res_n = 0
        ratios = []
        for r in v:
            na0, na1 = r["n_audit"]; nf0, nf1 = r["n_future"]
            ratios.append((na1 / (na0 + na1)) / (nf1 / (nf0 + nf1)))
            for name, grp in CELLS:
                a = r["audit_intervals"][name]
                pub_n += 1; pub_in += bool(a["inside"])
                n_aud = na1 if grp == 1 else na0
                n_bat = nf1 if grp == 1 else nf0
                lo, hi = intervals.thatcher(a["audit_count"], n_aud, n_bat, G)
                res_n += 1; res_in += bool(lo <= a["future_count"] <= hi)
        pub = 100.0 * pub_in / pub_n; resc = 100.0 * res_in / res_n
        mean_ratio = sum(ratios) / len(ratios)
        res["designs"][d] = {"reps": len(v), "published": pub, "rescaled": resc,
                             "mean_prevalence_ratio": mean_ratio,
                             "published_counts": [pub_in, pub_n],
                             "rescaled_counts": [res_in, res_n]}
        print(f"{d:8} {len(v):>5} {pub:>12.0f}% {resc:>18.0f}% {mean_ratio:>17.2f}")
    if OUT:
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
