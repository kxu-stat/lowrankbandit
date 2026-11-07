#!/usr/bin/env python3
"""Gather the cells of the regret experiment, check completeness, and summarize.

    python3 collect_regret.py FAMILY [the same --tiers/--designs/--levels/--horizon/--replications/--out options as
                                      run_regret.py] [--incomplete]

Writes, into OUT/collected/:
  status.tsv, environment.tsv   every planned cell's status; the software versions the cells were made with
  replicates.tsv                every result row (one per cell and configuration)
  summary.tsv                   per case and configuration: B, mean regret, sd, se (every configuration of every method)
  best.tsv                      per case and method: the configuration with the smallest mean regret (in hindsight);
                                for ESTR also its best multiplier at each T1 (method ESTR@k)
  comparisons.tsv               paired differences (left - right) with 95 % t intervals, every arm on the same environment:
                                  Proposed against every configuration of every competitor (the intersection-union test
                                  "Proposed beats every configuration" = every interval below 0; ESTR at T1 = tau
                                  is the competitor ESTR_tau, outside P2),
                                  Proposed against the matched arm at each T1, the matched arm against ESTR at the same T1,
                                  ESTR's best against OFUL's best, and each sensitivity point against the reported c_tau
  ordering.tsv                  the predictions P1-P7 per case
  regret_curves.tsv             mean cumulative regret of every configuration on the time grid
  scaling.tsv                   (tier scale) per case and horizon n: O, E* and F* at n (each competitor at its best
                                configuration at that n), the log-log slope from the previous n, and O's slope on
                                log n (OLS over the horizons)
No cell is ever dropped for its outcome; failed and missing cells are listed, and the collection stops on them unless
--incomplete is given.  Regret is in the model's units (the signal sd is 1).
"""
import argparse, json, os, sys
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np
from scipy import stats
import simulation as S
from run_regret import tasks, stamp

PREDICTIONS = [
    ("P1", "O < M: our method (c_tau = C_TAU) beats the same learner at each of ESTR's five exploration lengths T1"),
    ("P2", "O < every configuration of OFUL and of ESTR (intersection-union over the authors' grids)"),
    ("P3", "Gaussian rewards: E* < F* when d/r >= 8; F* < E* when d/r <= 4 (each method at its best configuration); "
           "either order in between.  Binary rewards: reported, not predicted"),
    ("P4", "sign(E* - F*) is K-invariant at fixed (d, r, noise)"),
    ("P5", "the advantage (E* - O)/E* grows with d/r (the noise slope is reported, not predicted)"),
    ("P6", "M_k < E_k at each T1: the matched arm beats ESTR at the same T1, ESTR at its best multiplier there (secondary)"),
    ("P7", "O < every multiplier of ESTR run with our exploration length (T1 = tau of O): the advantage is more than the "
           "shorter exploration")]
MAIN_ARM = "A_t%d" % S.C_TAU_GRID.index(S.C_TAU)          # the reported configuration of our method


def write_tsv(rows, path, columns=None):
    if columns is None:                       # the union of the rows' keys, in order of appearance
        columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else ("%.10g" % r[c] if isinstance(r[c], float) else str(r[c]))
                              for c in columns) + "\n")


def load_cells(out, all_tasks):
    status, rows, curves = [], [], []
    for tk in all_tasks:
        path = os.path.join(out, "cells", tk["key"] + ".json")
        case = tk["case"]
        if not os.path.exists(path):
            status.append(dict(key=tk["key"], status="missing", error="", elapsed=None))
            continue
        z = json.load(open(path))
        if z["status"] == "ok" and z.get("stamp") != stamp(tk):
            z["status"], z["error"] = "stale", "a cell made by another package version"
        status.append(dict(key=tk["key"], status=z["status"], error=z["error"].strip().split("\n")[-1] if z["error"] else "",
                           elapsed=z["elapsed"], **{k: z.get(k, "") for k in ("python", "numpy", "scipy", "node")}))
        if z["status"] != "ok":
            continue
        meta = dict(case_id=case["case_id"], group=case["group"], tier=case["tier"], design=case["design"],
                    d=case["d1"], rank=case["rank"], K=case["K"], d_over_r=case["d_over_r"],
                    noise_level=case["noise_level"], replication=tk["replication"], arm=tk["arm"],
                    truth=case.get("truth", "exact"), horizon=tk["horizon"])
        for r in z["rows"]:
            rows.append({**meta, **r})
        for cid, cv in z["curves"].items():
            curves.append({**meta, "config_id": cid, **cv})
    return status, rows, curves


def paired(by, cid, reps, left, right):
    """Paired difference left - right over the replications both have: mean, 95 % t interval, sign test."""
    diff = np.array([by[(cid, b, left)] - by[(cid, b, right)] for b in reps if (cid, b, left) in by and (cid, b, right) in by])
    if len(diff) < 2:
        return None
    t = stats.ttest_1samp(diff, 0.0)
    lo, hi = t.confidence_interval()
    return dict(B=len(diff), mean_diff=diff.mean(), se=diff.std(ddof=1) / np.sqrt(len(diff)), ci_lo=lo, ci_hi=hi,
                p_paired_t=t.pvalue, frac_left_smaller=float(np.mean(diff < 0)),
                p_sign=stats.binomtest(int(np.sum(diff < 0)), len(diff)).pvalue)


def summarise(rows, curves, out, family):
    # ---- every configuration's mean
    by = {}                                                   # (case, replication, config) -> regret
    groups = {}
    for r in rows:
        key = (r["case_id"], r["replication"], r["config_id"])
        if key in by:
            sys.exit("duplicate (case, replication, configuration) %s" % (key,))
        by[key] = r["regret"]
        groups.setdefault((r["case_id"], r["method"], r["config_id"]), []).append(r)
    summary = []
    for (cid, method, config), z in sorted(groups.items()):
        reg = np.array([r["regret"] for r in z])
        summary.append(dict(case_id=cid, tier=z[0]["tier"], truth=z[0].get("truth", "exact"), design=z[0]["design"],
                            d=z[0]["d"], rank=z[0]["rank"], K=z[0]["K"], horizon=z[0]["horizon"],
                            d_over_r=z[0]["d_over_r"], noise_level=z[0]["noise_level"], method=method, config_id=config,
                            tau=z[0]["tau"], c_tau=z[0].get("c_tau"), multiplier=z[0].get("multiplier"), B=len(z),
                            regret=reg.mean(), sd=reg.std(ddof=1) if len(z) > 1 else None,
                            se=reg.std(ddof=1) / np.sqrt(len(z)) if len(z) > 1 else None,
                            exploration=np.mean([r["exploration_regret"] for r in z]),
                            post=np.mean([r["post_regret"] for r in z]),
                            err_n=np.mean([r["err_n"] for r in z if r.get("err_n") is not None] or [np.nan]),
                            rank_n=np.mean([r["rank_n"] for r in z if r.get("rank_n") is not None] or [np.nan]),
                            # ESTR's stage-1 subspace error (baselines.stage1_subspace_error) and the share of
                            # replications in which stage 1 missed a dominant direction (error above 0.5)
                            bm_sin=np.mean([r["bm_sin"] for r in z if r.get("bm_sin") is not None] or [np.nan]),
                            bm_fail=np.mean([r["bm_sin"] > 0.5 for r in z if r.get("bm_sin") is not None] or [np.nan])))
    write_tsv(summary, os.path.join(out, "summary.tsv"))

    # ---- each competitor's best configuration (in hindsight), and ESTR's best multiplier at each T1
    present = {s["method"] for s in summary}                 # ESTR_r4: ESTR given rank 4 (tier approx)
    competitors = [m for m in ("OFUL", "ESTR", "ESTR_r4", "ESTR_tau", "OFUL_pm1", "ESTR_pm1") if m in present]
    best, best_rows = {}, []
    for cid in sorted(set(s["case_id"] for s in summary)):
        mine = [s for s in summary if s["case_id"] == cid]
        for method in competitors:
            cand = [s for s in mine if s["method"] == method]
            if not cand:
                continue
            b = min(cand, key=lambda s: s["regret"])
            best[(cid, method)] = b["config_id"]
            best_rows.append(dict(case_id=cid, method=method, config_id=b["config_id"], tau=b["tau"],
                                  multiplier=b["multiplier"], regret=b["regret"], se=b["se"], B=b["B"]))
            if method in ("ESTR", "ESTR_r4", "ESTR_pm1"):     # the methods on ESTR's T1 grid
                for k in range(S.T1_STEPS):
                    at = [s for s in cand if s["config_id"].startswith("ESTR_BM_g1_t%02d_" % k)]
                    if at:
                        bk = min(at, key=lambda s: s["regret"])
                        best[(cid, "%s@%d" % (method, k))] = bk["config_id"]
                        best_rows.append(dict(case_id=cid, method="%s@%d" % (method, k), config_id=bk["config_id"],
                                              tau=bk["tau"], multiplier=bk["multiplier"], regret=bk["regret"],
                                              se=bk["se"], B=bk["B"]))
    write_tsv(best_rows, os.path.join(out, "best.tsv"))

    # ---- paired comparisons
    comps = []

    def add(cid, kind, left, right, reps):
        z = paired(by, cid, reps, left, right)
        if z is not None:
            comps.append(dict(case_id=cid, kind=kind, left=left, right=right, **z, left_smaller=z["ci_hi"] < 0))

    for cid in sorted(set(s["case_id"] for s in summary)):
        reps = sorted(set(r["replication"] for r in rows if r["case_id"] == cid))
        configs = [s for s in summary if s["case_id"] == cid]
        for s in configs:                                     # O against every configuration of every competitor
            if s["method"] in competitors:
                add(cid, "O_vs_config_" + s["method"], MAIN_ARM, s["config_id"], reps)
        for k in range(S.T1_STEPS):
            add(cid, "O_vs_M", MAIN_ARM, "M_k%d" % k, reps)                       # P1
            for method in ("ESTR", "ESTR_pm1"):
                if (cid, "%s@%d" % (method, k)) in best:
                    add(cid, "M_vs_" + method + "_same_T1", "M_k%d" % k, best[(cid, "%s@%d" % (method, k))], reps)  # P6
        for E, F in (("ESTR", "OFUL"), ("ESTR_pm1", "OFUL_pm1"), ("ESTR_r4", "OFUL")):
            if (cid, E) in best and (cid, F) in best:
                add(cid, "Ebest_vs_Fbest" + E[4:], best[(cid, E)], best[(cid, F)], reps)        # P3 ("", _pm1, _r4)
        for s in configs:                                     # the sensitivity of c_tau
            if s["method"] == "Proposed" and s["config_id"] != MAIN_ARM:
                add(cid, "sensitivity", s["config_id"], MAIN_ARM, reps)
    write_tsv(comps, os.path.join(out, "comparisons.tsv"))

    # ---- the predictions per case
    mean_of = {(s["case_id"], s["config_id"]): s["regret"] for s in summary}
    ordering = []
    for cid in sorted(set(s["case_id"] for s in summary)):
        z = next(s for s in summary if s["case_id"] == cid)
        mine = [c for c in comps if c["case_id"] == cid]
        O = mean_of.get((cid, MAIN_ARM), np.nan)
        E, F = mean_of.get((cid, best.get((cid, "ESTR"))), np.nan), mean_of.get((cid, best.get((cid, "OFUL"))), np.nan)
        p1 = [c["left_smaller"] for c in mine if c["kind"] == "O_vs_M"]
        p2 = [c["left_smaller"] for c in mine if c["kind"] in ("O_vs_config_OFUL", "O_vs_config_ESTR", "O_vs_config_ESTR_r4")]
        p6 = [c["left_smaller"] for c in mine if c["kind"] == "M_vs_ESTR_same_T1"]
        p7 = [c["left_smaller"] for c in mine if c["kind"] == "O_vs_config_ESTR_tau"]
        ef = next((c for c in mine if c["kind"] == "Ebest_vs_Fbest"), None)
        if family == S.P3_FAMILY and ef is not None and z.get("truth", "exact") == "exact":
            p3 = (ef["ci_hi"] < 0 if z["d_over_r"] >= S.DR_REF else ef["ci_lo"] > 0 if z["d_over_r"] <= S.DR_LOW else None)
        else:
            p3 = None                     # binary rewards and the approx tier: reported
        row = dict(case_id=cid, design=z["design"], d=z["d"], rank=z["rank"], K=z["K"], d_over_r=z["d_over_r"],
                   noise_level=z["noise_level"], O=O, E_best=E, F_best=F,
                   E_config=best.get((cid, "ESTR")), F_config=best.get((cid, "OFUL")),
                   **{"M_k%d" % k: mean_of.get((cid, "M_k%d" % k), np.nan) for k in range(S.T1_STEPS)},
                   P1_all_k=bool(p1) and all(p1), P1_count="%d/%d" % (sum(p1), len(p1)),
                   P2_all_configs=bool(p2) and all(p2), P2_count="%d/%d" % (sum(p2), len(p2)),
                   E_lt_F=bool(E < F), P3_holds=p3, P6_all_k=bool(p6) and all(p6), P6_count="%d/%d" % (sum(p6), len(p6)),
                   adv_E=(E - O) / E if E else None, truth=z.get("truth", "exact"))
        if p7:                                               # ESTR at T1 = tau (main and karm tiers)
            row.update(E_tau_best=mean_of.get((cid, best[(cid, "ESTR_tau")]), np.nan), E_tau_config=best[(cid, "ESTR_tau")],
                       P7_all_configs=all(p7), P7_count="%d/%d" % (sum(p7), len(p7)))
        if (cid, "ESTR_r4") in best:                         # tier approx: ESTR given rank 4
            row.update(E_r4_best=mean_of.get((cid, best[(cid, "ESTR_r4")]), np.nan), E_r4_config=best[(cid, "ESTR_r4")])
        if family == S.PM1_FAMILY:
            Ep = mean_of.get((cid, best.get((cid, "ESTR_pm1"))), np.nan)
            Fp = mean_of.get((cid, best.get((cid, "OFUL_pm1"))), np.nan)
            pp = [c["left_smaller"] for c in mine if c["kind"] in ("O_vs_config_OFUL_pm1", "O_vs_config_ESTR_pm1")]
            row.update(E_pm1_best=Ep, F_pm1_best=Fp, E_pm1_lt_F_pm1=bool(Ep < Fp),
                       O_beats_all_pm1="%d/%d" % (sum(pp), len(pp)))
        ordering.append(row)
    write_tsv(ordering, os.path.join(out, "ordering.tsv"))

    # ---- the tier scale: the regret at the horizon against n, ours and each competitor at its best (in hindsight)
    scaling = []
    for s0 in [s for s in summary if s["config_id"] == MAIN_ARM and s["tier"] in S.TIER_HORIZONS]:
        cid, n = s0["case_id"], int(s0["horizon"])
        row = dict(case=cid[:-len("_n%06d" % n)], design=s0["design"], noise_level=s0["noise_level"], n=n,
                   B=s0["B"], tau=s0["tau"], O=s0["regret"], O_se=s0["se"])
        for m, tag in (("ESTR", "E"), ("OFUL", "F")):
            b = next(s for s in summary if s["case_id"] == cid and s["config_id"] == best[(cid, m)])
            row.update({tag: b["regret"], tag + "_se": b["se"], tag + "_config": b["config_id"]})
        scaling.append(row)
    if scaling:
        scaling.sort(key=lambda r: (r["case"], r["n"]))
        for case in sorted({r["case"] for r in scaling}):          # O = a + b log n over the horizons, and the
            rows = [r for r in scaling if r["case"] == case]       # log-log slope between consecutive horizons
            b_log = np.polyfit(np.log([r["n"] for r in rows]), [r["O"] for r in rows], 1)[0]
            for prev, r in zip([None] + rows[:-1], rows):
                r["O_per_log_n"] = b_log
                for tag in ("O", "E", "F"):
                    r[tag + "_slope"] = (np.log(r[tag] / prev[tag]) / np.log(r["n"] / prev["n"])) if prev else None
        write_tsv(scaling, os.path.join(out, "scaling.tsv"))

    # ---- curves
    agg = {}
    for cv in curves:
        for t, v in zip(cv["t"], cv["cumulative_regret"]):
            agg.setdefault((cv["case_id"], cv["config_id"], t), []).append(v)
    write_tsv([dict(case_id=cid, config_id=cfg, t=t, B=len(v), mean=float(np.mean(v)),
                    se=float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None)
               for (cid, cfg, t), v in sorted(agg.items())], os.path.join(out, "regret_curves.tsv"))

    # ---- printed summary
    if scaling:                       # the tier scale: the predictions are judged on the main grid, not here
        print("\nregret at the horizon against n (tier scale; E*, F* the best configuration at each n; slope: log-log, "
              "from the previous n):")
        for r in scaling:
            print("  %-22s n %6d  tau %2d  O %8.1f (%5.1f)  E* %8.1f  F* %8.1f   slopes O %s E* %s F* %s   O per log n %.1f" % (
                r["case"], r["n"], r["tau"], r["O"], r["O_se"], r["E"], r["F"],
                *("%5.2f" % r[k] if r[k] is not None else "    -" for k in ("O_slope", "E_slope", "F_slope")),
                r["O_per_log_n"]))
        return
    with1 = [o for o in ordering if o["P1_count"] != "0/0"]       # the cases with the matched arm
    with6 = [o for o in ordering if o["P6_count"] != "0/0"]
    print("\npredictions (%d cases; P1-P6 stated in advance, P7 added later; paired 95 %% t intervals):" % len(ordering))
    print("  P1  O < M at every T1              : %d / %d cases" % (sum(o["P1_all_k"] for o in with1), len(with1)))
    print("  P2  O < every OFUL / ESTR config   : %d / %d cases" % (sum(o["P2_all_configs"] for o in ordering), len(ordering)))
    if family == S.P3_FAMILY:
        exact = [o for o in ordering if o.get("truth", "exact") == "exact"]           # P3: exact low-rank truths only
        k1 = [o for o in exact if o["d_over_r"] >= S.DR_REF]
        k0 = [o for o in exact if o["d_over_r"] <= S.DR_LOW]
        print("  P3  E* < F* when d/r >= 8          : %d / %d" % (sum(bool(o["P3_holds"]) for o in k1), len(k1)))
        print("  P3  F* < E* when d/r <= 4          : %d / %d" % (sum(bool(o["P3_holds"]) for o in k0), len(k0)))
    else:
        print("  (P3 not declared for binary rewards) E* < F* in %d / %d cases, reported" % (
            sum(o["E_lt_F"] for o in ordering), len(ordering)))
    print("  P6  M < E at every T1              : %d / %d cases" % (sum(o["P6_all_k"] for o in with6), len(with6)))
    with7 = [o for o in ordering if "P7_count" in o]
    if with7:
        print("  P7  O < ESTR at T1 = tau (every m) : %d / %d cases" % (sum(o["P7_all_configs"] for o in with7), len(with7)))
    print("  P4 and P5: verdict_regret.py (E4, E5)")
    for o in ordering:
        print("  %-24s d/r %4.1f  O %8.3f  E* %8.3f (%s)  F* %8.3f (%s)  P1 %s  P2 %s  P6 %s%s" % (
            o["case_id"], o["d_over_r"], o["O"], o["E_best"], o["E_config"], o["F_best"], o["F_config"],
            o["P1_count"], o["P2_count"], o["P6_count"],
            "  E(tau) %.3f P7 %s" % (o["E_tau_best"], o["P7_count"]) if "P7_count" in o else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["gaussian", "binomial"])
    ap.add_argument("--tiers", default=None, type=lambda v: v.split(","))
    ap.add_argument("--designs", default=None, type=lambda v: v.split(","))
    ap.add_argument("--levels", default=None)
    ap.add_argument("--horizon", type=int, default=S.HORIZON)
    ap.add_argument("--replications", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--incomplete", action="store_true", help="summarize what exists")
    a = ap.parse_args()
    levels = None if a.levels is None else [float(v) if a.family == "gaussian" else int(v) for v in a.levels.split(",")]
    out = a.out or "regret_%s" % a.family
    all_tasks = tasks(a.family, a.tiers, levels, a.horizon, a.replications, a.designs)
    status, rows, curves = load_cells(out, all_tasks)
    collected = os.path.join(out, "collected")
    os.makedirs(collected, exist_ok=True)
    write_tsv(status, os.path.join(collected, "status.tsv"))
    keys = ("python", "numpy", "scipy")
    stamp_of = lambda s: tuple(s.get(k, "") for k in keys)
    seen = sorted({stamp_of(s) for s in status if s.get("python")})
    if seen:
        write_tsv([dict(zip(keys, v), cells=sum(1 for s in status if stamp_of(s) == v)) for v in seen],
                  os.path.join(collected, "environment.tsv"))
    bad = [s for s in status if s["status"] != "ok"]
    print("cells: %d ok, %d failed, %d missing of %d planned" % (
        len(status) - len(bad), sum(s["status"] in ("error", "stale") for s in bad),
        sum(s["status"] == "missing" for s in bad), len(status)))
    for s in bad[:50]:
        print("  %-8s %s %s" % (s["status"], s["key"], s["error"]))
    if bad and not a.incomplete:
        sys.exit("incomplete: re-run the missing/failed tasks, or pass --incomplete")
    write_tsv(rows, os.path.join(collected, "replicates.tsv"))
    summarise(rows, curves, collected, a.family)
    print("collected into", collected)


if __name__ == "__main__":
    main()
