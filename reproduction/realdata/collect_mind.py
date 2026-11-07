#!/usr/bin/env python3
"""Collect the MIND regret study's cells.

    python3 collect_mind.py DIR --setup SETUP_NPZ [--block down4 | main] [--best BEST_TSV] [--horizon n]
                            [--replications B] [--incomplete]

One block per run directory (run_mind.BLOCKS): down4 (the main text, DIR = mind_regret) or main (the appendix, DIR =
mind_regret_main, with --best the down4 block's collected/best.tsv, as for run_mind.py).

Writes DIR/collected/:
  status.tsv       every planned cell's status (ok, error, stale = made under another setup or package version,
                   missing)
  replicates.tsv   every result row (one per cell and configuration)
  summary.tsv      per configuration: B, regret (mean, s.e., median, 5 % and 95 % quantiles), clicks, the stream
                   constants (sum of the best candidate's mean, regret of uniform play), our final ||Theta_hat - Theta*||_F
  best.tsv         per competitor: the configuration with the smallest mean regret (in hindsight); for ESTR also its
                   best multiplier at each T1 (ESTR@k)
  comparisons.tsv  paired differences "comparator - ours" in regret (positive = ours better), 95 % t-intervals and
                   two-sided paired-t p-values: Proposed against every configuration of every comparator, against the
                   matched arm at each T1; the matched arm against ESTR at the same T1; ESTR* against OFUL*; each
                   sensitivity point of c_tau against the reported one
  verdict_mind.md  the comparisons stated in advance: M1-M3, each an intersection-union test over the
                   comparator's configurations, Holm over the three; M4, M5 secondary; M6: the
                   two-track collection of the inference against every configuration of the same three comparators,
                   intersection-union tests, Holm over the three, a family of its own; the rest reported (GLM-ESTR at
                   T1 = our tau; the main-effect greedy).  The main block: the same tables,
                   reported (no prediction)
Every cell is checked for regret + expected = the stream's best_possible (sum over rounds of the best candidate's
mean), the same for every method on a stream.  Completeness is checked against run_mind.tasks (the same pure
function); the collection stops on missing or failed cells unless --incomplete is given.
"""
import argparse, json, os, sys
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
from scipy import stats
import run_mind as R

MAIN_ARM = R.OURS                                            # our reported configuration (c_tau = 1/20)
TWO_TRACK = "two_track"                                      # the inference's data collection (M6)
COMPETITORS = ("OFUL", "ESTR", "AuxGreedy", "ESTR_tau", "CMGreedy", "OFUL_pm1", "ESTR_pm1")
PRIMARY_BLOCK = "down4"                                      # the block of the predictions (run_mind.BLOCKS)
PRIMARY = ("OFUL", "ESTR", "AuxGreedy")                      # M1-M3 and M6: logistic OFUL, GLM-ESTR, the auxiliary-
                                                             # covariate greedy; the 2y - 1 versions and CMGreedy are
                                                             # reported
SIGN = {"PASS": "> 0", "FAIL": "< 0", "WATCH": "contains 0"}       # a reported interval's position
LABEL = {"OFUL": "logistic OFUL", "ESTR": "GLM-ESTR (our adaptation)", "AuxGreedy": "auxiliary-covariate greedy",
         "ESTR_tau": "GLM-ESTR at T1 = our tau", "CMGreedy": "main-effect greedy (c, m)",
         "OFUL_pm1": "OFUL, authors' 2y - 1", "ESTR_pm1": "ESTR, authors' 2y - 1"}


def write_tsv(rows, path):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else ("%.10g" % r[c] if isinstance(r[c], float) else str(r[c]))
                              for c in columns) + "\n")


def load_cells(out, tasks, setup_sha):
    status, rows, curves, identity = [], [], [], []
    for tk in tasks:
        path = os.path.join(out, "cells", tk["key"] + ".json")
        if not os.path.exists(path):
            status.append(dict(key=tk["key"], status="missing", error=""))
            continue
        z = json.load(open(path))
        if z["status"] == "ok" and z.get("stamp") != R.stamp(tk, setup_sha):
            z["status"], z["error"] = "stale", "made under another setup or package version"
        status.append(dict(key=tk["key"], status=z["status"], elapsed=z["elapsed"],
                           error=z["error"].strip().split("\n")[-1] if z["error"] else "",
                           **{k: z.get(k, "") for k in ("python", "numpy", "scipy", "node")}))
        if z["status"] != "ok":
            continue
        best = z["stream"]["best_possible"]
        for r in z["rows"]:
            if abs(r["regret"] + r["expected"] - best) > 1e-6 * max(1.0, best):
                identity.append("%s %s" % (tk["key"], r["config_id"]))
            rows.append(dict({k: v for k, v in r.items() if k != "gamma_n"}, block=tk["block"], replication=tk["replication"],
                             arm=tk["arm"], best_possible=best, uniform=z["stream"]["uniform"], n=z["stream"]["n"]))
        for cid, cv in z["curves"].items():
            curves.append(dict(block=tk["block"], replication=tk["replication"], config_id=cid, **cv))
    return status, rows, curves, identity


def paired(a, b):
    """a - b over the replications both have: mean, sd, 95 % t-interval, two-sided paired-t p."""
    common = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in common])
    if len(d) < 2:
        return None
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, len(d) - 1) * se
    p = 2 * stats.t.sf(abs(d.mean()) / se, len(d) - 1) if se > 0 else 0.0
    return dict(B=len(d), diff=d.mean(), sd=d.std(ddof=1), lo=d.mean() - half, hi=d.mean() + half, p=p)


def status_of(lo, hi):
    """The prediction is a positive difference (comparator - ours): PASS, FAIL or WATCH."""
    return "PASS" if lo > 0 else "FAIL" if hi < 0 else "WATCH"


def holm(p):
    """Holm's step-down adjusted p-values (Holm 1979)."""
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def summarise(rows, curves, out):
    reg = {}                                                  # (block, config) -> {replication: regret}
    by = {}
    for r in rows:
        reg.setdefault((r["block"], r["config_id"]), {})[r["replication"]] = r["regret"]
        by.setdefault((r["block"], r["method"], r["config_id"]), []).append(r)
    summary = []
    for (block, method, cid), z in sorted(by.items()):
        v = np.array([r["regret"] for r in z])
        te = [r["theta_error_n"] for r in z if r.get("theta_error_n") is not None and np.isfinite(r["theta_error_n"])]
        summary.append(dict(block=block, method=method, config_id=cid, tau=z[0]["tau"], c_tau=z[0].get("c_tau"),
                            multiplier=z[0].get("multiplier"), B=len(z), regret=v.mean(),
                            post_regret=float(np.mean([r["post_regret"] for r in z])),
                            se=v.std(ddof=1) / np.sqrt(len(z)) if len(z) > 1 else None, median=float(np.median(v)),
                            q05=float(np.quantile(v, 0.05)), q95=float(np.quantile(v, 0.95)),
                            clicks=float(np.mean([r["clicks"] for r in z])),
                            best_possible=float(np.mean([r["best_possible"] for r in z])),
                            uniform=float(np.mean([r["uniform"] for r in z])),
                            theta_error=float(np.mean(te)) if te else None))
    write_tsv(summary, os.path.join(out, "summary.tsv"))

    best, best_rows = {}, []                                  # each competitor's best configuration, in hindsight
    for block in sorted({s["block"] for s in summary}):
        mine = [s for s in summary if s["block"] == block]
        for method in COMPETITORS:
            cand = [s for s in mine if s["method"] == method]
            if not cand:
                continue
            b = min(cand, key=lambda s: s["regret"])
            best[(block, method)] = b["config_id"]
            best_rows.append(dict(block=block, method=method, config_id=b["config_id"], tau=b["tau"],
                                  multiplier=b["multiplier"], regret=b["regret"], se=b["se"], B=b["B"]))
            if method in ("ESTR", "ESTR_pm1"):                # the methods on ESTR's T1 grid
                for k in range(R.T1_STEPS):
                    at = [s for s in cand if s["config_id"].startswith("ESTR_BM_g1_t%02d_" % k)]
                    if at:
                        bk = min(at, key=lambda s: s["regret"])
                        best[(block, "%s@%d" % (method, k))] = bk["config_id"]
                        best_rows.append(dict(block=block, method="%s@%d" % (method, k), config_id=bk["config_id"],
                                              tau=bk["tau"], multiplier=bk["multiplier"], regret=bk["regret"],
                                              se=bk["se"], B=bk["B"]))
    write_tsv(best_rows, os.path.join(out, "best.tsv"))

    comps = []

    def add(block, kind, other, mine):
        z = paired(reg.get((block, other), {}), reg.get((block, mine), {}))
        if z is not None:
            comps.append(dict(block=block, kind=kind, comparator=other, ours=mine, **z,
                              status=status_of(z["lo"], z["hi"])))

    for block in sorted({s["block"] for s in summary}):
        for s in [s for s in summary if s["block"] == block]:
            if s["method"] in COMPETITORS:
                add(block, "vs_config_" + s["method"], s["config_id"], MAIN_ARM)          # M1-M3 and the 2y - 1 versions
                add(block, "M6_vs_config_" + s["method"], s["config_id"], TWO_TRACK)      # M6
            if s["method"] == "Proposed" and s["config_id"] != MAIN_ARM:
                add(block, "sensitivity", s["config_id"], MAIN_ARM)                       # c_tau, reported
        add(block, "cost_of_inference", TWO_TRACK, MAIN_ARM)                              # reported
        for k in range(R.T1_STEPS):
            add(block, "O_vs_M", "M_k%d" % k, MAIN_ARM)                                    # M4
            for E in ("ESTR", "ESTR_pm1"):
                if (block, "%s@%d" % (E, k)) in best:
                    add(block, "M_vs_" + E + "_same_T1", best[(block, "%s@%d" % (E, k))], "M_k%d" % k)   # M5
        for E, F in (("ESTR", "OFUL"), ("ESTR_pm1", "OFUL_pm1")):
            if (block, E) in best and (block, F) in best:
                add(block, "Ebest_vs_Fbest" + ("_pm1" if "pm1" in E else ""), best[(block, F)], best[(block, E)])
    write_tsv(comps, os.path.join(out, "comparisons.tsv"))

    agg = {}
    for cv in curves:
        for t, v in zip(cv["t"], cv["cumulative_regret"]):
            agg.setdefault((cv["block"], cv["config_id"], t), []).append(v)
    write_tsv([dict(block=b, config_id=c, t=t, B=len(v), mean=float(np.mean(v)),
                    se=float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None)
               for (b, c, t), v in sorted(agg.items())], os.path.join(out, "regret_curves.tsv"))
    return summary, best, comps


def iut(comps, block, kinds):
    """The intersection-union test over the comparisons of `kinds` (Berger 1982): PASS iff every interval lies above 0;
    its p-value is the largest of the individual p-values; the binding comparison is the one whose interval reaches
    lowest."""
    z = [c for c in comps if c["block"] == block and c["kind"] in kinds]
    if not z:
        return None
    binding = min(z, key=lambda c: c["lo"])
    st = "PASS" if all(c["status"] == "PASS" for c in z) else "FAIL" if any(c["status"] == "FAIL" for c in z) else "WATCH"
    return dict(n=len(z), passed=sum(c["status"] == "PASS" for c in z), binding=binding, status=st,
                p=max(c["p"] if c["diff"] > 0 else 1.0 for c in z))


def primary_table(comps, best, block, prefix, ours, head):
    """The table of one family of primary comparisons (M1-M3 or M6): per comparator of PRIMARY, the intersection-union
    test over its configurations, Holm over the three; reported only outside the primary block."""
    prim = [(m, iut(comps, block, (prefix + m,))) for m in PRIMARY]
    prim = [(m, t) for m, t in prim if t is not None]
    if not prim:
        return []
    primary = block == PRIMARY_BLOCK
    md = ["## %s block -- %s%s" % (block, "" if primary else "reported (no prediction): ", head), "",
          "| comparator | configurations | intervals above 0 | best config (in hindsight): difference [95 %%] | "
          "binding config: difference [95 %%] | IUT p | %s | status |" % ("Holm p" if primary else "-"),
          "|---|---|---|---|---|---|---|---|"]
    adj = holm([t["p"] for _, t in prim])
    for (m, t), a in zip(prim, adj):
        bc = next((c for c in comps if c["block"] == block and c["kind"] == prefix + m
                   and c["comparator"] == best.get((block, m))), None)
        b = t["binding"]
        md.append("| %s | %d | %d / %d | %s %.1f [%.1f, %.1f] | %s %.1f [%.1f, %.1f] | %.2g | %s | %s |" % (
            LABEL[m], t["n"], t["passed"], t["n"], bc["comparator"], bc["diff"], bc["lo"], bc["hi"],
            b["comparator"], b["diff"], b["lo"], b["hi"], t["p"], "%.2g" % a if primary else "-",
            t["status"] if primary else "(%s)" % t["status"]))
    if primary:
        ok = len(prim) == len(PRIMARY) and all(t["status"] == "PASS" for _, t in prim)
        md += ["", "- %s beats every configuration of every comparator (intersection-union, level 5 %%): %s"
               % (ours, "YES" if ok else "NO")]
    return md + [""]


def verdict(summary, best, comps, out, debug):
    mean = {(s["block"], s["config_id"]): s for s in summary}
    md = ["# MIND verdict", ""]
    if debug:
        md += ["**DEBUG RUN** (fewer streams or a shorter horizon than the plan): the pipeline only, no claim.", ""]
    md += ["Differences are comparator - ours in expected regret (positive = ours better), paired over the streams, 95 %% "
           "t-intervals.  Ours: %s (c_tau = 1/%d, the penalty constant by CV); for M6 the inference's two-track "
           "collection %s (the same tau, two tracks after the shared exploration, CV inside each track).  A comparator "
           "with a grid is tested over every configuration (intersection-union test: PASS iff every interval lies above "
           "0; p = the largest p)." % (MAIN_ARM, round(1 / R.C_TAU), TWO_TRACK), ""]
    blocks = sorted({s["block"] for s in summary}, key=lambda b: (b != PRIMARY_BLOCK, b))
    for block in blocks:
        for prefix, ours, head in (("vs_config_", MAIN_ARM, "M1-M3: ours (%s) against each comparator" % MAIN_ARM),
                                   ("M6_vs_config_", TWO_TRACK, "M6: the inference's two-track collection (%s) "
                                    "against each comparator (a family of its own)" % TWO_TRACK)):
            md += primary_table(comps, best, block, prefix, ours, head)
        m45 = [c for c in comps if c["block"] == block and c["kind"] in ("O_vs_M", "M_vs_ESTR_same_T1")]
        if m45:
            md += ["### %s block -- %s" % (block, "secondary: M4 (O < M at every T1) and M5 (M < E at every T1)"
                                           if block == PRIMARY_BLOCK else "M4 and M5 (reported)"), "",
                   "| comparison | T1 | difference [95 %] | status |", "|---|---|---|---|"]
            for kind, lab in (("O_vs_M", "M4: M_k - Proposed"), ("M_vs_ESTR_same_T1", "M5: ESTR@k - M_k")):
                for c in [c for c in m45 if c["kind"] == kind]:
                    k = int(c["comparator"][3]) if kind == "O_vs_M" else int(c["comparator"].split("_t")[1][:2])
                    md.append("| %s | %s | %.1f [%.1f, %.1f] | %s |" % (lab, mean[(block, "M_k%d" % k)]["tau"], c["diff"],
                                                                        c["lo"], c["hi"], c["status"]))
            md.append("")
        c = next((c for c in comps if c["block"] == block and c["kind"] == "cost_of_inference"), None)
        if c is not None:                                     # the two tracks against the single track: reported
            md += ["### %s block -- the cost of the inference (reported): the two-track collection against ours" % block,
                   "", "| two_track - %s | ratio of the mean regrets | ratio after the exploration |" % MAIN_ARM,
                   "|---|---|---|",
                   "| %.1f [%.1f, %.1f] | %.3f | %.3f |" % (
                       c["diff"], c["lo"], c["hi"], mean[(block, TWO_TRACK)]["regret"] / mean[(block, MAIN_ARM)]["regret"],
                       mean[(block, TWO_TRACK)]["post_regret"] / mean[(block, MAIN_ARM)]["post_regret"]), ""]
        md += ["### %s block -- reported: GLM-ESTR at T1 = our tau, the main-effect greedy, "
               "the authors' 2y - 1 versions, ESTR* against OFUL*, the c_tau sensitivity" % block, "",
               "| comparison | difference [95 %] | sign of the interval |", "|---|---|---|"]
        for m in ("ESTR_tau", "CMGreedy", "OFUL_pm1", "ESTR_pm1"):     # reported
            t = iut(comps, block, ("vs_config_" + m,))
            if t:
                b = t["binding"]
                md.append("| %s - Proposed: %d / %d configurations above 0; binding %s | %.1f [%.1f, %.1f] | %s |" % (
                    LABEL[m], t["passed"], t["n"], b["comparator"], b["diff"], b["lo"], b["hi"], SIGN[b["status"]]))
        for c in [c for c in comps if c["block"] == block and c["kind"].startswith("Ebest_vs_Fbest")]:
            md.append("| OFUL* - ESTR*%s (%s, %s; positive: ESTR better) | %.1f [%.1f, %.1f] | %s |" % (
                " (2y - 1)" if "pm1" in c["kind"] else "", c["comparator"], c["ours"], c["diff"], c["lo"], c["hi"], SIGN[c["status"]]))
        for c in [c for c in comps if c["block"] == block and c["kind"] == "sensitivity"]:
            md.append("| %s (c_tau 1/%d) - Proposed | %.1f [%.1f, %.1f] | %s |" % (
                c["comparator"], round(1 / mean[(block, c["comparator"])]["c_tau"]), c["diff"], c["lo"], c["hi"], SIGN[c["status"]]))
        md.append("")
    md += ["## Mean regret per configuration", "", "| block | method | configuration | tau | B | regret | s.e. | clicks |",
           "|---|---|---|---|---|---|---|---|"]
    md += ["| %s | %s | %s | %s | %d | %.1f | %s | %.1f |" % (s["block"], s["method"], s["config_id"], s["tau"], s["B"],
                                                             s["regret"], "%.1f" % s["se"] if s["se"] else "-", s["clicks"])
           for s in sorted(summary, key=lambda s: (s["block"] != PRIMARY_BLOCK, s["block"], s["method"], s["config_id"]))]
    open(os.path.join(out, "verdict_mind.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--setup", required=True, help="the setup (its rank fixes the task list)")
    ap.add_argument("--block", default="down4", choices=sorted(R.BLOCKS))
    ap.add_argument("--best", default=None, help="main block: the down4 block's collected/best.tsv (as for run_mind.py)")
    ap.add_argument("--horizon", type=int, default=None, help="default: the block's")
    ap.add_argument("--replications", type=int, default=None)
    ap.add_argument("--incomplete", action="store_true", help="summarize what exists")
    a = ap.parse_args()
    setup = dict(np.load(a.setup))
    tasks = R.tasks(setup, a.block, a.horizon, a.replications, a.best)
    status, rows, curves, identity = load_cells(a.out, tasks, R.sha16(a.setup))
    collected = os.path.join(a.out, "collected")
    os.makedirs(collected, exist_ok=True)
    write_tsv(status, os.path.join(collected, "status.tsv"))
    bad = [s for s in status if s["status"] != "ok"]
    print("cells: %d ok, %d failed or stale, %d missing of %d planned" % (
        len(status) - len(bad), sum(s["status"] in ("error", "stale") for s in bad),
        sum(s["status"] == "missing" for s in bad), len(status)))
    for s in bad[:30]:
        print("  %-8s %s %s" % (s["status"], s["key"], s["error"]))
    print("identity (regret + expected = the stream's sum of the best candidate's mean): %s"
          % ("holds on all %d rows" % len(rows) if not identity else "FAILS on %d rows, e.g. %s" % (len(identity), identity[:3])))
    if identity:
        sys.exit("a cell's regret is not the stream's regret: stop")
    if bad and not a.incomplete:
        sys.exit("incomplete: re-run the missing / failed tasks, or pass --incomplete")
    write_tsv(rows, os.path.join(collected, "replicates.tsv"))
    summary, best, comps = summarise(rows, curves, collected)
    spec = R.BLOCKS[a.block]
    debug = tasks[0]["horizon"] != spec["horizon"] or (a.replications or spec["B"]) < spec["B"] or bool(bad)
    verdict(summary, best, comps, collected, debug)
    print("collected into", collected)


if __name__ == "__main__":
    main()
