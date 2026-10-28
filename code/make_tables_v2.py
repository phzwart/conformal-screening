"""Generates the LaTeX tables of the revision from stored results (no hand-typed numbers).
Usage: python3 -I make_tables_v2.py ROOT   (ROOT = paper directory; writes ROOT/tables/*.tex)
Tables whose inputs are missing are skipped with a message."""
import json, os, sys, re
ROOT = sys.argv[1]; T = os.path.join(ROOT, "tables")
def load(p):
    p = os.path.join(ROOT, p)
    if not os.path.exists(p): print("missing", p); return None
    return json.load(open(p))
def relabel(t):
    """display names: in-class (label 1, the property screened for) and out-class (label 0)"""
    for a, b in (("missed active", "missed in-class"), ("deferred inactive", "deferred out-class"), ("deferred active", "deferred in-class"),
                 ("inactives", "out-class"), ("actives", "in-class"), ("inactive", "out-class")):
        t = t.replace(a, b)
    return re.sub(r"(?<![a-z_-])active(?![a-z_])", "in-class", t)
def write(name, text): open(os.path.join(T, name), "w").write(relabel(text)); print("wrote", name)
f3 = lambda v: f"{v:.3f}"; f1 = lambda v: f"{v:.1f}"; iv = lambda a: f"[{a[0]}, {a[1]}]"

# ---- ESI: tied scores ----
d = load("code/ties_sim.json")
if d:
    rows = []
    for o in d:
        k, r = o["ties kept"], o["randomized tie-breaking"]
        rows.append(f"{100 * o['tied_fraction']:.0f} & {f3(k['cells']['0_s0'])} & {f3(r['cells']['0_s0'])} & {f3(k['cells']['0_s1'])} & {f3(r['cells']['0_s1'])} & {f3(k['row_joint']['0'])} & {f3(r['row_joint']['0'])} & {f3(k['row_joint']['1'])} & {f3(r['row_joint']['1'])} \\\\")
    R = d[0]["ties kept"]["cell_n"]["0_s0"]
    write("tab_ties.tex", r"""\begin{table}[ht]
\centering
\caption{Tied scores and calibration-only intervals ($n=1000$ calibration and $m=1000$ future compounds, prevalence 0.2, standard thresholds at $\alpha=0.10$; %d replicates per row). A fraction of inactive compounds share one score at the 0.92 quantile of the inactive distribution, as for a cluster of identical fingerprints. Entries: coverage of the nominal 95\%% class-conditional intervals, with ties kept (kept) and with randomized tie-breaking (rand.). Row: all cells of the class row covered together.}
\label{tab:ties}
\small
\begin{tabular}{rrrrrrrrr}
\toprule
Tied & \multicolumn{2}{c}{correct rejection} & \multicolumn{2}{c}{false lead} & \multicolumn{2}{c}{inactive row} & \multicolumn{2}{c}{active row} \\
inactives (\%%) & kept & rand. & kept & rand. & kept & rand. & kept & rand. \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (R, "\n".join(rows)))

# ---- ESI: calibration-only simulation ----
d = load("code/calib_only.json")
if d:
    rows = []
    for o in d:
        first = True
        for r in o["rows"]:
            rows.append(f"{o['n_cal'] if first else ''} & {r['cell'].replace('{', '$\\{').replace('}', '\\}$')} & {f3(r['coverage_class_conditional'])} & {f3(r['coverage_mixed_batch'])} & {f1(r['width_calibration_only_mixed'])} & {f1(r['width_split_budget'])} & {f3(r['coverage_split_budget'])} \\\\")
            first = False
        rows.append(r"\midrule")
    rows = rows[:-1]
    info = "; ".join(f"$n={o['n_cal']}$: {o['reps']} replicates ({o['skipped_fewer_than_20_per_class']} skipped with fewer than 20 calibration compounds in a class), all mixed-batch cells covered together in {100 * o['mixed_joint_all_cells']:.1f}\\%" for o in d)
    write("tab_calonly.tex", r"""\begin{table*}[ht]
\centering
\caption{Calibration-only intervals (nominal 95\%%), future batch of $m=1000$. Class-conditional: coverage of the row intervals given the number of future compounds of the class. Mixed: coverage for a batch of unknown composition (union over the interval for the number of actives; levels as in Section~\ref{sec:calonly}). Widths are for the mixed batch. Split budget: the same $n$ labelled compounds split equally into calibration and test sets, with test-set intervals. %s.}
\label{tab:calonly}
\small
\begin{tabular}{rlccrrc}
\toprule
$n$ & Cell & \multicolumn{2}{c}{Coverage, calibration only} & \multicolumn{2}{c}{Mean width} & Coverage \\
 & & class-conditional & mixed & calibration only & split budget & split budget \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (info, "\n".join(rows)))

# ---- AqSolDB: main Table 5 rows, ESI Chow table, ESI per-split and capacity table ----
S = load("mol/v2/aqsol_results.json")
if S:
    A, B = S["setting_A_fewest_missed"], S["setting_B_fewest_deferred"]
    names = [("missed", "missed soluble"), ("false_lead", "false lead"), ("confirmed", "soluble confirmed"), ("rejected", "insoluble rejected"),
             ("deferred_insoluble", "deferred, insoluble"), ("deferred_soluble", "deferred, soluble")]
    rows = [f"{lab} & {A[k]['test_per1000']} & {iv(A[k]['testset_interval'])} & {iv(A[k]['testset_interval_bonferroni'])} & {B[k]['test_per1000']} & {iv(B[k]['testset_interval'])} & {iv(B[k]['testset_interval_bonferroni'])} \\\\" for k, lab in names]
    write("tab_solubility_rows.tex", "\n".join(rows) + "\n")
    lab = {"5_0.3": (5, 0.3), "10_0.5": (10, 0.5), "20_0.8": (20, 0.8), "50_0.5": (50, 0.5)}
    rows = []
    for k, (l, r) in lab.items():
        c = S["chow"][k]; d = lambda b: f"{c['paired_diff_vs_' + b]['mean']:+.3f} ({c['paired_diff_vs_' + b]['se']:.3f})"
        rows.append(f"{l} & {r} & {f3(c['chow_platt'])} & {f3(c['chow_isotonic'])} & {f3(c['erm_two_thresholds'])} & {f3(c['best_candidate_incl_trivial'])} & {f3(c['defer_all'])} & {d('chow_platt')} & {d('chow_isotonic')} \\\\")
    write("tab_chow.tex", r"""\begin{table*}[ht]
\centering
\caption{AqSolDB lipophilic screen: mean test-set cost per compound (units of a wasted follow-up) over %d splits. All baselines were fitted on the full calibration set and applied to the test set. Chow: cost-optimal action per compound on probabilities recalibrated by Platt scaling (logistic regression on the logit of $u$) or isotonic regression. Two thresholds: the two thresholds on $u$ minimizing empirical calibration cost. Set rule: cheapest candidate on calibration estimates, among the swept rules and the trivial rules (defer all, label all inactive, label all soluble). Last two columns: paired difference, set rule minus Chow, mean (standard error over splits).}
\label{tab:chow}
\small
\begin{tabular}{rrccccccc}
\toprule
$\lambda$ & $\rho$ & Chow (Platt) & Chow (isotonic) & two thresholds & set rule & defer all & set $-$ Platt & set $-$ isotonic \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (S["reps"], "\n".join(rows)))
    P = S["per_split"]; rows = []
    for x, c in zip(P, S["capacity"]):
        b = c["bonferroni"]
        cap = lambda z: f"{z['upper_deferred_per1000']} / {z['test_deferred_per1000']} / {z['test_errors_per1000']}" if z else "--"
        rows.append(f"{x['rep']} & {x['distinct_rules']} & {x['pareto']} & {x['supported']} & {100 * x['region_point']:.1f} & {100 * x['region_robust']:.1f} & {cap(b)} \\\\")
    write("tab_aqsol_splits.tex", r"""\begin{table*}[ht]
\centering
\caption{AqSolDB, per split. Rules: distinct rules from the sweep; Pareto: rules not dominated on estimated missed solubles, false leads and deferrals; cheapest: rules that are cheapest for some cost ratio on the grid of Fig.~\ref{main-fig:solubility}; regions: percentage of that grid where acting on single labels is cheapest for the cheapest rule, at the estimated within-set fractions (point) and for all fractions in simultaneous 95\%% test-set intervals (robust). Pareto, cheapest and regions use the audit-set rates of each rule. Capacity (calibration only): at most 500 deferrals per 1000; the rule with fewest estimated errors whose calibration-only upper limit for deferrals is within capacity, Bonferroni over the %d requests; entries are upper limit / realized audit-set deferrals / realized audit-set errors, per 1000.}
\label{tab:aqsolsplits}
\small
\begin{tabular}{rrrrrrc}
\toprule
Split & rules & Pareto & cheapest & \multicolumn{2}{c}{region (\%%)} & capacity rule \\
 & & & & point & robust & (calibration only) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (len(S["alphas"]) ** 2, "\n".join(rows)))

# ---- Tox21 (revision run): ESI endpoint table, monotonicity, pilots; main Table 4 rows ----
TX = load(os.environ.get("TOX21_RESULTS", "mol/v2/tox21_results_v2.json"))
if TX:
    import numpy as np
    try:
        F = np.load(os.path.join(ROOT, "mol/tox21_features.npz"), allow_pickle=True); NLAB = dict(zip(list(F["tasks"]), (~np.isnan(F["Y"])).sum(0).tolist()))
    except Exception: NLAB = {}
    tex = lambda t: t.replace("NR-PPAR-gamma", r"NR-PPAR-$\gamma$")
    rows = []
    for t, E in TX["endpoints"].items():
        r, sc = E["random"], E.get("scaffold", {}); b, st = r["batch"], r["standard"]; v = r["violations_batch"]
        rows.append(f"{tex(t)} & {NLAB.get(t, '')} & {100 * r['prevalence_test']:.1f} & {r['n_cal_actives_median']:.0f} & {r['auroc_median']:.2f} & {sc.get('auroc_median', float('nan')):.2f} & {100 * b['single_fraction']['median']:.1f} & {100 * st['single_fraction']['median']:.1f} & {100 * b['hit_rate_active_calls']:.0f} & {b['enrichment']:.1f} & {v['inactive']['observed']}/{v['inactive']['expected']:.1f} & {v['active']['observed']}/{v['active']['expected']:.1f} & {100 * r['calonly']['coverage_class_conditional']:.1f} & {100 * r['tied_fraction_median']:.1f} \\\\")
    nrep = next(iter(TX["endpoints"].values()))["random"]["reps"]
    write("tab_tox21.tex", r"""\begin{table*}[ht]
\centering
\caption{Tox21 endpoints and results over %d random splits with the batch correction ($\alpha=\delta=0.10$ per class) unless stated. Columns: labelled compounds; active (\%%); median calibration actives; median test AUROC (random, scaffold splits); median single-label fraction (\%%) with the batch correction and with the standard threshold; fraction of actives among single-label active calls (\%%) and enrichment over prevalence; per-class violations, i.e.\ splits in which test coverage of the class fell below 90\%%, observed/expected under the exact law (inactive, active); coverage of the class-conditional calibration-only intervals (\%%); median tied scores before tie-breaking (\%%).}
\label{tab:tox21data}
\small
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lrrrrrrrrrrrrr}
\toprule
Endpoint & $n$ & act. & cal. act. & AUC r & AUC s & single & single (std) & hit & enr. & viol. 0 & viol. 1 & cal.-only & ties \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (nrep, "\n".join(rows)))
    rows = [f"{tex(t)} & {' & '.join(f'{100 * x:.1f}' for x in m['active_fraction_by_bin'])} & {m['decreases']} & {m['largest_decrease_z']:.1f} \\\\" for t, m in TX["monotonicity"].items()]
    write("tab_tox21mono.tex", r"""\begin{table*}[ht]
\centering
\caption{Monotonicity diagnostic for the Tox21 models: percentage of actives in ten equal-count bins of the test-set score $u$, pooled over random splits, from the lowest to the highest bin; number of adjacent bins in which the fraction decreases; largest such decrease in standard errors ($z$). With ten bins and noisy low-prevalence bins, decreases with $z<2$ are expected by chance.}
\label{tab:tox21mono}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
Endpoint & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & decr. & max $z$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(rows))
    rows = []
    for npil, p in TX["pilots"].items():
        v, vf = p["violations"], p["violations_feasible_only"]
        rows.append(f"{npil} & {p['pilots']} & {p['too_few_actives_skipped']} & {p['median_actives']:.0f} & {100 * p['request_infeasible_fraction']:.0f} & {v['inactive']['observed']}/{v['inactive']['expected']:.1f} & {v['active']['observed']}/{v['active']['expected']:.1f} & {vf['active']['observed']}/{vf['active']['expected']:.1f} ({vf['active']['n']}) & {100 * p['calonly']['coverage_class_conditional']:.1f} & {100 * p['calonly']['coverage_mixed']:.1f} \\\\")
    so = TX["violations_scaffold_other"]; co = TX["calonly_scaffold_other"]
    rows.insert(0, f"other scaffolds & -- & -- & -- & -- & {so['inactive']['observed']}/{so['inactive']['expected']:.1f} & {so['active']['observed']}/{so['active']['expected']:.1f} & -- & {100 * co['coverage_class_conditional']:.1f} & {100 * co['coverage_mixed']:.1f} \\\\")
    write("tab_pilots.tex", r"""\begin{table*}[ht]
\centering
\caption{Tox21 scaffold splits: calibration on scaffolds other than those screened, or on a random pilot sample of the screened compounds (pilot sizes 200, 500, 1000; the rest of the test set is screened). Pilots with fewer than five actives were skipped. Infeasible: pilots for which the active-class request ($\alpha=\delta=0.10$ for the planned number of actives) cannot be met, so the most conservative threshold is used. Violations: splits in which test coverage of the class fell below 90\%%, observed/expected under the exact law; for feasible pilots only, with their number. Last columns: coverage (\%%) of calibration-only intervals, class-conditional and for the mixed screened batch.}
\label{tab:pilots}
\small
\begin{tabular}{lrrrrrrrrr}
\toprule
Calibration & pilots & skipped & median & infeasible & \multicolumn{2}{c}{violations} & active, feasible & \multicolumn{2}{c}{calibration-only} \\
 & & & actives & (\%%) & inactive & active & pilots only & class & mixed \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(rows))
    rep = TX["representative"]; ts = list(rep)
    order = ["correct rejection", "false lead", "deferred inactive", "confident hit", "missed active", "deferred active"]
    lab = {"deferred inactive": "deferred", "deferred active": "deferred"}
    rows = [f"{'inactive' if i < 3 else 'active'} & {lab.get(c, c)} & " + " & ".join(f"{rep[t]['batch'][c]['test_per1000']} {iv(rep[t]['batch'][c]['testset_interval'])}" for t in ts) + r" \\" for i, c in enumerate(order)]
    write("tab_tox21cells_rows.tex", "\n".join(rows) + "\n")
    json.dump({t: {"rep": rep[t]["rep"], "rank": rep[t]["rank"], "n_cal": rep[t]["n_cal"], "n_test": rep[t]["n_test"]} for t in ts}, open(os.path.join(T, "tab_tox21cells_meta.json"), "w"))

# ---- audit-set check on real data (main text Table) ----
AC = load("mol/v2/audit_check.json")
if AC:
    r, s_ = AC["tox21_random"], AC["tox21_scaffold"]
    order = ["correct rejection", "false lead", "deferred inactive", "confident hit", "missed active", "deferred active", "single label", "deferred"]
    rows = [f"{k} & {100 * r['coverage'][k]:.1f} & {100 * s_['coverage'][k]:.1f} \\\\" for k in order]
    write("tab_audit_rows.tex", "\n".join(rows) + "\n")
    json.dump({"random_checks": r["checks"], "scaffold_checks": s_["checks"], "aqsol": AC["aqsol"],
               "random_per_task_range": [min(r["per_task"].values()), max(r["per_task"].values())],
               "scaffold_per_task_range": [min(s_["per_task"].values()), max(s_["per_task"].values())]}, open(os.path.join(T, "tab_audit_meta.json"), "w"), indent=1)

# ---- ESI: interval validation (main text Fig. 2) ----
d = load("code/validation.json")
if d and "cpxm_coverage" in d[0]:
    NAMES = {"single-label": "single label", "single-label error": "single-label error", "both labels": "both labels",
             "empty set": "empty set", "miscoverage": "miscoverage"}
    rows = []
    for o in d:
        for i, kp in enumerate(o["kpis"]):
            c = lambda k: f3(o[k + "_coverage"][i]); w = lambda k: f1(o[k + "_mean_width"][i])
            rows.append(f"{o['n_test'] if i == 0 else ''} & {NAMES[kp]} & {c('exact')} & {w('exact')} & {c('bayes1')} & {w('bayes1')} & {c('cpxm')} & {w('cpxm')} & {c('plugin')} & {f3(o['clopper_pearson_longrun_coverage'][i])} \\\\")
    write("tab_validation.tex", r"""\begin{table*}[ht]
\centering
\caption{Coverage of 95\% prediction intervals for the count of each outcome in a future batch of $m=1000$, and mean interval width, from an audit set of $n$ compounds; @REPS@ replicates per audit size. Audit interval: the Beta--Binomial limits of Section~\ref{sec:intervals}. $m\times$CP: $m$ times the Clopper--Pearson interval for the rate, rounded outward. Plug-in: Binomial$(m,K/n)$ quantiles. Last column: Clopper--Pearson coverage of the long-run rate $p_g(Q)$, the quantity it is designed for.}
\label{tab:validation}
\small
\begin{tabular}{rlcccccccc}
\toprule
$n$ & Outcome & \multicolumn{2}{c}{Audit interval} & \multicolumn{2}{c}{Bayesian, uniform prior} & \multicolumn{2}{c}{$m\times$CP} & Plug-in & CP, rate \\
 & & coverage & width & coverage & width & coverage & width & coverage & coverage \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table*}
""".replace("@REPS@", str(d[0]["reps"])).replace("@ROWS@", "\n".join(rows)))
