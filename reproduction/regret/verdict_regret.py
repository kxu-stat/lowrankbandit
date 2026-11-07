#!/usr/bin/env python3
"""The verdict of the regret experiment: do the collected results meet the expectations stated in advance
at the declared thresholds?  Reads what collect_regret.py wrote; re-tunes, drops and re-weights nothing.

    python3 verdict_regret.py [--out DIR] [RUN_DIR ...]

RUN_DIR are output roots of run_regret.py (default: regret_gaussian and regret_binomial, those that exist).  Writes
OUT/verdict_regret.md and .tsv.  Statuses: PASS (met), WATCH (direction as predicted but not established), FAIL
(contradicted), INFO (reported, not judged), n/a (a run with fewer replications than simulation.PLAN: the debug pass
exercises the pipeline and makes no claim).
"""
import argparse, csv, datetime, os, re, sys
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np
from scipy import stats
import simulation as S

DESIGN_OF = {d[:3]: d for d in S.MASTER_DESIGNS}      # case-id prefix -> design

CRITERIA = [
    ("C1", "completeness: every planned cell finished; failures kept and listed", "0 missing (FAIL otherwise); 0 failed (WATCH otherwise)"),
    ("C3", "replications: every (case, configuration) carries its case's planned B", "B == the planned B everywhere (FAIL if below)"),
    ("C4", "solver health: finite error, rank and regret in every Proposed and matched cell", "all finite (PASS); else FAIL"),
    ("S1", "sensitivity: our regret over the c_tau grid, relative to the reported c_tau", "INFO: reported (supplement)"),
    ("S2", "(tier scale) regret at the horizon against n, every run tuned to its n: O's slope on log n (OLS over the "
           "horizons) and the log-log slopes of O, E*, F* between the two largest n", "INFO: reported"),
    ("E1", "P1  O < M: paired 95% CI of (O - M_k) below 0 for every T1 in every case", "all CIs < 0 PASS; any CI > 0 FAIL; else WATCH"),
    ("E2", "P2  O < every configuration of OFUL and ESTR: paired CI of (O - configuration) below 0 for all (intersection-union)", "as E1, over every configuration"),
    ("E3", "P3  (Gaussian) E* < F* when d/r >= 8 and F* < E* when d/r <= 4, E*, F* the best configurations; either order in between",
     "as E1 on the d/r >= 8 and d/r <= 4 cases; in between INFO; binary rewards INFO (reported, not predicted)"),
    ("E4", "P4  sign(E* - F*) invariant across K at fixed (design, d, r, noise)", "all groups with >= 2 K levels invariant PASS; else WATCH"),
    ("E5", "P5  (E* - O)/E* increases with d/r: OLS of the case means on log(d/r) and the noise coordinate; the d/r slope judged",
     "d/r slope > 0 and p < 0.05 PASS; > 0 WATCH; < 0 with p < 0.05 FAIL; one noise level: INFO"),
    ("E6", "precision: Monte-Carlo relative standard error of O, E* and F*", "max se/mean <= 0.05 PASS; <= 0.10 WATCH; else FAIL"),
    ("E8", "P6  (secondary) M_k < E_k at every T1: paired CI of (M_k - ESTR at that T1, its best multiplier) below 0", "as E1"),
    ("E9", "P7  O < ESTR at T1 = tau (our exploration length): paired CI of (O - each multiplier) below 0",
           "as E1; main and karm tiers"),
    ("G1", "(binary rewards, supplement) O against every configuration of the authors' 2y - 1 OFUL and ESTR", "INFO: reported"),
    ("B1", "ESTR's stage 1 (50 starts): the stage-1 recoveries whose subspace misses a dominant "
           "direction of the truth (bm_sin > 0.5) where most replications at that (case, T1) recover it (median <= 0.25): "
           "the isolated failures a spurious local minimum produces; and the (case, T1) where T1 is too short for the data", "INFO: reported"),
    ("A1", "(tier approx: approximately low-rank truths, singular values ~ j^-2 and j^-1.5) our estimation error and fitted rank at n; ESTR given "
           "rank 2 and rank 4 at their best; E* against F* (P3 is not declared for this truth)", "INFO: reported"),
    ("X1", "LM/GLM correspondence: O, M, E, F present in both families (runs with the matched arm only)", "present in both PASS; else FAIL"),
    ("X2", "LM/GLM correspondence: P1 and P2 in the representative case (16, 2), K = 16, sphere, both links", "hold in both PASS; in one WATCH; else FAIL"),
]
LEVEL = {"n/a": 0, "INFO": 0, "PASS": 1, "WATCH": 2, "FAIL": 3}


def worst(statuses):
    return max(statuses, key=lambda s: LEVEL[s]) if statuses else "n/a"


def read_tsv(path):
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", "True", "False") else (v == "True" if v else None)
            except ValueError:
                pass
    return rows


def fmt(v):
    return ("%.3f" % v) if isinstance(v, float) else str(v)


def md_table(cols, rows):
    if not rows:
        return ["(none)"]
    return (["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
            ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows])


def judge(comps, debug, flip=False):
    """PASS if every interval lies below 0 (above 0 with flip), FAIL if one lies on the other side, else WATCH."""
    if debug or not comps:
        return "n/a"
    below = [(z["ci_lo"] > 0) if flip else (z["ci_hi"] < 0) for z in comps]
    wrong = [(z["ci_hi"] < 0) if flip else (z["ci_lo"] > 0) for z in comps]
    return "FAIL" if any(wrong) else "PASS" if all(below) else "WATCH"


def count(comps, flip=False):
    return "%d / %d" % (sum((z["ci_lo"] > 0) if flip else (z["ci_hi"] < 0) for z in comps), len(comps))


def noise_coordinate(family, level):
    return np.log(0.5 / np.sqrt(level)) if family == "binomial" else np.log(level)


def ols(y, *xs):
    """OLS of y on (1, x1, ...): slopes, their standard errors and two-sided p-values."""
    X = np.column_stack([np.ones(len(y))] + list(xs))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    dof = len(y) - X.shape[1]
    s2 = np.sum((y - X @ beta) ** 2) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta[1:], se[1:], 2 * stats.t.sf(np.abs(beta[1:] / se[1:]), dof)


def p5(od, family):
    ok = [o for o in od if np.isfinite([o["E_best"], o["O"], o["d_over_r"]]).all() and o["E_best"]]
    adv = np.array([(o["E_best"] - o["O"]) / o["E_best"] for o in ok])
    log_dr = np.log([o["d_over_r"] for o in ok])
    n_dr, n_noise = len({o["d_over_r"] for o in ok}), len({o["noise_level"] for o in ok})
    if n_noise > 1 and n_dr > 1 and len(ok) >= 6:
        b, se, p = ols(adv, log_dr, [noise_coordinate(family, o["noise_level"]) for o in ok])
        status = "FAIL" if (b[0] < 0 and p[0] < 0.05) else "PASS" if (b[0] > 0 and p[0] < 0.05) else "WATCH"
        return (status, "slope on log(d/r) %.3f (se %.3f, p = %.3f); on the noise coordinate %.3f (se %.3f, p = %.3f), "
                "reported; %d cases" % (b[0], se[0], p[0], b[1], se[1], p[1], len(ok)), "the d/r slope is judged")
    if n_noise == 1 and n_dr > 1 and len(ok) >= 3:
        b, se, p = ols(adv, log_dr)
        return "INFO", "slope on log(d/r) %.3f (se %.3f, p = %.3f); one noise level; %d cases" % (b[0], se[0], p[0], len(ok)), ""
    return "n/a", "%d cases" % len(ok), "needs cases at several d/r"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    runs = a.runs or [d for d in ("regret_gaussian", "regret_binomial")
                      if os.path.isfile(os.path.join(d, "collected", "status.tsv"))]
    if not runs:
        sys.exit("no collected run directories")
    rows, md, evals = [], [], {}

    def add(run, cid, status, value="", note=""):
        rows.append(dict(run=run, id=cid, status=status, value=value, note=note))

    for run in runs:
        first = len(rows)                             # this run's rows start here
        cd = os.path.join(run, "collected")
        st, rp = read_tsv(os.path.join(cd, "status.tsv")), read_tsv(os.path.join(cd, "replicates.tsv"))
        if not st:
            sys.exit("no collected/status.tsv in %s: run collect_regret.py first" % run)
        family = "binomial" if "_nb" in st[0]["key"] else "gaussian"
        planned = {}                                  # case -> the replications its tasks carry
        for s_ in st:
            cid = re.match(r"(.+)_b\d+_", s_["key"]).group(1)
            planned.setdefault(cid, set()).add(int(re.search(r"_b(\d+)_", s_["key"]).group(1)))
        debug = any(len(v) < S.PLAN[DESIGN_OF[cid[:3]]][1] for cid, v in planned.items())
        tag = family
        md += ["## `%s`: %s rewards, %d cases, B = %s, %d cells" % (
            run, family, len(planned), "/".join(str(b) for b in sorted({len(v) for v in planned.values()})), len(st)), ""]
        missing = [s for s in st if s["status"] == "missing"]
        failed = [s for s in st if s["status"] in ("error", "stale")]
        add(tag, "C1", "FAIL" if missing else "WATCH" if failed else "PASS",
            "ok %d / failed %d / missing %d of %d" % (len(st) - len(missing) - len(failed), len(failed), len(missing), len(st)))
        if failed:
            md += ["### Failure log (cells kept, never dropped)", "", *md_table(["key", "status", "error"], failed), ""]
        reps = {}
        for r in rp:
            reps.setdefault((r["case_id"], r["config_id"]), set()).add(r["replication"])
        short = [k for k, v in reps.items() if len(v) < len(planned[k[0]])]
        add(tag, "C3", "FAIL" if short else "PASS", "%d (case, configuration) with fewer replications than planned" % len(short),
            "fewer replications than simulation.PLAN: statistical criteria are n/a (debug pass)" if debug else "")
        ours = [r for r in rp if r["method"] in ("Proposed", "Matched")]
        finite = all(np.isfinite([r["err_n"], r["rank_n"], r["regret"]]).all() for r in ours) if ours else True
        elapsed = [s["elapsed"] for s in st if isinstance(s.get("elapsed"), float)]
        add(tag, "C4", "n/a" if not ours else "PASS" if finite else "FAIL",
            "finite: %s; median cell %.0f s, max %.0f s" % (finite, np.median(elapsed) if elapsed else 0,
                                                            max(elapsed) if elapsed else 0))
        cp = read_tsv(os.path.join(cd, "comparisons.tsv"))
        sm = read_tsv(os.path.join(cd, "summary.tsv"))
        od = read_tsv(os.path.join(cd, "ordering.tsv"))
        if not od:
            continue
        kind = lambda *k: [z for z in cp if z["kind"] in k]
        sens = kind("sensitivity")
        ratio = [1 + z["mean_diff"] / next(s["regret"] for s in sm if s["case_id"] == z["case_id"] and s["config_id"] == z["right"])
                 for z in sens]
        add(tag, "S1", "INFO", "regret at the other c_tau / at C_TAU: %.2f-%.2f over %d (case, c_tau); CI above 0 in %s"
            % (min(ratio), max(ratio), len(ratio), count(sens, flip=True)) if ratio else "none")
        e1 = kind("O_vs_M")
        add(tag, "E1", judge(e1, debug), "CI < 0 in %s (case, T1)" % count(e1))
        e2 = kind("O_vs_config_OFUL", "O_vs_config_ESTR", "O_vs_config_ESTR_r4")
        add(tag, "E2", judge(e2, debug), "CI < 0 in %s (case, configuration); cases with all: %d / %d" % (
            count(e2), sum(bool(o["P2_all_configs"]) for o in od), len(od)))
        dr_of = {o["case_id"]: o["d_over_r"] for o in od}
        exact = {o["case_id"] for o in od if o.get("truth", "exact") in ("exact", None)}   # P3: exact low rank only
        e3 = [z for z in kind("Ebest_vs_Fbest") if z["case_id"] in exact]
        hi = [z for z in e3 if dr_of[z["case_id"]] >= S.DR_REF]
        lo = [z for z in e3 if dr_of[z["case_id"]] <= S.DR_LOW]
        declared = family == S.P3_FAMILY
        s3 = worst([judge(hi, debug), judge(lo, debug, flip=True)]) if declared else "INFO"
        add(tag, "E3", s3, "d/r >= 8: E* < F* (CI) in %s; d/r <= 4: F* < E* (CI) in %s" % (count(hi), count(lo, flip=True)),
            "" if declared else "binary rewards: reported, not predicted")
        groups = {}
        for o in od:
            groups.setdefault((o["design"], o["d"], o["rank"], o["noise_level"]), []).append(o)
        kk = [len({o["E_lt_F"] for o in g}) == 1 for g in groups.values() if len({o["K"] for o in g}) > 1]
        add(tag, "E4", "n/a" if (debug or not kk) else "PASS" if all(kk) else "WATCH",
            "K-invariant in %d / %d groups" % (sum(kk), len(kk)) if kk else "one K level only")
        s5, v5, n5 = p5(od, family)
        add(tag, "E5", "n/a" if debug else s5, v5, n5)
        # each case's own O, E* and F*
        main_cfg = {(o["case_id"], c) for o in od for c in ("A_t%d" % S.C_TAU_GRID.index(S.C_TAU), o["E_config"], o["F_config"])}
        rel = [(s["se"] / s["regret"], s) for s in sm if (s["case_id"], s["config_id"]) in main_cfg
               and isinstance(s["se"], float) and s["regret"]]
        if rel:
            rs, sw = max(rel, key=lambda z: z[0])
            add(tag, "E6", "n/a" if debug else "PASS" if rs <= 0.05 else "WATCH" if rs <= 0.10 else "FAIL",
                "max se/mean = %.3f (%s, %s)" % (rs, sw["case_id"], sw["config_id"]))
        e8 = kind("M_vs_ESTR_same_T1")
        add(tag, "E8", judge(e8, debug), "CI < 0 in %s (case, T1)" % count(e8))
        sc = read_tsv(os.path.join(cd, "scaling.tsv"))
        if sc:
            last = {}
            for r in sc:
                if r["case"] not in last or r["n"] > last[r["case"]]["n"]:
                    last[r["case"]] = r
            add(tag, "S2", "INFO", "; ".join("%s: O per log n %.1f; slopes at n = %d: O %.2f, E* %.2f, F* %.2f" % (
                c, r["O_per_log_n"], r["n"], r["O_slope"], r["E_slope"], r["F_slope"]) for c, r in sorted(last.items())))
        e9 = kind("O_vs_config_ESTR_tau")
        if e9:
            add(tag, "E9", judge(e9, debug), "CI < 0 in %s (case, multiplier); cases with all: %d / %d" % (
                count(e9), sum(bool(o.get("P7_all_configs")) for o in od), len(od)))
        g1 = kind("O_vs_config_OFUL_pm1", "O_vs_config_ESTR_pm1")
        if g1:
            add(tag, "G1", "INFO", "CI < 0 in %s (case, configuration of the 2y - 1 versions)" % count(g1))
        # B1: ESTR's stage 1, one recovery per (case, method, T1, replication) (the five multipliers share it).  At a
        # T1 too short for the data the subspace is missed in most replications (a statistical limit, reported as the
        # (case, T1) median); an isolated failure -- bm_sin > 0.5 where the (case, T1) median is at most 0.25 -- is
        # what a spurious local minimum looks like
        recov = {}
        for r in rp:
            if str(r["method"]).startswith("ESTR") and isinstance(r.get("bm_sin"), float):
                recov[(r["case_id"], r["method"], r["tau"], r["replication"])] = r["bm_sin"]
        if recov:
            by_t1 = {}
            for (cid, meth, t1, b), v in recov.items():
                by_t1.setdefault((cid, meth, t1), []).append((b, v))
            iso, short = [], []
            for key, vals in by_t1.items():
                med = float(np.median([v for _, v in vals]))
                if med > 0.5:
                    short.append(key)
                elif med <= 0.25:
                    iso += [(key, b, v) for b, v in vals if v > 0.5]
            add(tag, "B1", "INFO", "%d stage-1 recoveries in %d (case, T1); isolated failures (bm_sin > 0.5, median of its "
                "(case, T1) <= 0.25): %d%s; (case, T1) with median bm_sin > 0.5 (T1 too short for the data): %d of %d"
                % (len(recov), len(by_t1), len(iso),
                   " (" + ", ".join("%s %s T1 %d b%d %.2f" % (k[0], k[1], k[2], b, v) for k, b, v in iso[:8]) + ")" if iso else "",
                   len(short), len(by_t1)))
        apx = [o for o in od if o.get("truth", "exact") not in ("exact", None)]
        if apx:
            main_arm = "A_t%d" % S.C_TAU_GRID.index(S.C_TAU)
            errs = {s_["case_id"]: s_ for s_ in sm if s_["config_id"] == main_arm}
            add(tag, "A1", "INFO", "; ".join(
                "%s: O %.2f (error %.3f, rank %.1f), E* r = 2 %.2f, E* r = 4 %s, F* %.2f, E* < F* (CI) %s" % (
                    o["case_id"], o["O"], errs[o["case_id"]]["err_n"], errs[o["case_id"]]["rank_n"], o["E_best"],
                    "%.2f" % o["E_r4_best"] if isinstance(o.get("E_r4_best"), float) else "-", o["F_best"],
                    count([z for z in kind("Ebest_vs_Fbest") if z["case_id"] == o["case_id"]]))
                for o in apx))
        if sc:                                        # the tier scale: its comparisons are reported, not judged
            for r in rows[first:]:
                if r["id"] in ("E1", "E2", "E3", "E4", "E5", "E8", "E9"):
                    r.update(status="INFO", note="tier scale: reported; the predictions are judged on the main grid")
        table = []
        for o in sorted(od, key=lambda o: (o["design"], o["K"], o["rank"], o["d"], o["noise_level"])):
            table.append(dict(case_id=o["case_id"], d_over_r=o["d_over_r"], O=o["O"], E_best=o["E_best"], E_config=o["E_config"],
                              F_best=o["F_best"], F_config=o["F_config"], P1=o["P1_count"], P2=o["P2_count"],
                              E_lt_F=o["E_lt_F"], P6=o["P6_count"], E_tau=o.get("E_tau_best", ""), P7=o.get("P7_count", "")))
        evals[family] = dict(table=table, od=od, debug=debug, methods=sorted({s["method"] for s in sm}))
        cols = ["case_id", "d_over_r", "O", "E_best", "E_config", "F_best", "F_config", "P1", "P2", "E_lt_F", "P6"]
        md += ["### Per-case outcomes (mean regret; P1, P2, P6, P7: intervals below 0 / comparisons%s)" % (
                   "; B below the plan, indicative only" if debug else ""), "",
               *md_table(cols + (["E_tau", "P7"] if e9 else []), table), ""]
    if "gaussian" in evals and "binomial" in evals and all("Matched" in e["methods"] for e in evals.values()):
        g, b = evals["gaussian"], evals["binomial"]          # the tiers approx and d32 have no matched arm
        four = {"Proposed", "Matched", "ESTR", "OFUL"}
        add("both", "X1", "PASS" if four <= set(g["methods"]) and four <= set(b["methods"]) else "FAIL",
            "gaussian: %s; binomial: %s" % (",".join(g["methods"]), ",".join(b["methods"])))
        rep = lambda e: [o for o in e["od"] if o["case_id"].startswith("sph_d16_r2_K16")]
        holds = lambda cells: bool(cells) and all(o["P1_all_k"] and o["P2_all_configs"] for o in cells)
        hg, hb = holds(rep(g)), holds(rep(b))
        add("both", "X2", "n/a" if (g["debug"] or b["debug"]) else "PASS" if (hg and hb) else "WATCH" if (hg or hb) else "FAIL",
            "representative case: P1 and P2 hold under the identity link: %s; under the logit link: %s" % (hg, hb))
    crit = {c[0]: c for c in CRITERIA}
    overall = worst([r["status"] for r in rows])
    debug_any = any(r["id"] == "C3" and "debug" in r["note"] for r in rows)
    if debug_any:
        headline = ("DEBUG PASS: pipeline complete end to end, no statistical claim (fewer replications than the plan)"
                    if overall != "FAIL" else "DEBUG: a structural criterion FAILED -- see the table")
    else:
        headline = {"PASS": "PASS: the expectations stated in advance are met at the declared thresholds",
                    "WATCH": "WATCH: nothing contradicts the predictions, but at least one criterion is not established",
                    "FAIL": "FAIL: at least one criterion is contradicted"}.get(overall, overall)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "WATCH", "FAIL", "INFO", "n/a")}
    head = ["# Regret experiment: verdict", "",
            "Generated %s by verdict_regret.py from %s." % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ", ".join(runs)), "",
            "## Verdict: %s" % headline, "",
            "Criteria: %d PASS, %d WATCH, %d FAIL, %d INFO, %d n/a." % tuple(counts.values()), "",
            "| run | id | status | value | note |", "|---|---|---|---|---|",
            *["| %s | %s | %s | %s | %s |" % (r["run"], r["id"], r["status"], r["value"], r["note"]) for r in rows], "",
            "### The standard (declared before the data)", "",
            *md_table(["id", "criterion", "threshold"], [dict(id=c[0], criterion=c[1], threshold=c[2]) for c in CRITERIA]), ""]
    tail = ["## How to read this", "",
            "- Every arm saw the same environment in a (case, replication), so the paired 95% interval is the statement.",
            "- Our method runs at one fixed c_tau with its penalty constant chosen by CV inside each run; every competitor "
            "is reported over the authors' whole grid and compared at every configuration, so E2 is the claim "
            "'better than every configuration of OFUL and ESTR'.",
            "- P1 and P2 are the claims of the paper; P3-P6 the mechanism predictions (P6 secondary); P7 (ESTR with our "
            "exploration length) separates the exploration length from the learner.", ""]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "verdict_regret.md"), "w").write("\n".join(head + md + tail))
    with open(os.path.join(a.out, "verdict_regret.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "id", "status", "value", "note", "criterion", "threshold"], delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "criterion": crit[r["id"]][1], "threshold": crit[r["id"]][2]})
    print("\n".join(head[4:9 + len(rows)]))
    print("\nverdict written:", os.path.join(a.out, "verdict_regret.md"))


if __name__ == "__main__":
    main()
