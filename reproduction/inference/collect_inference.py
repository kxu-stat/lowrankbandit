#!/usr/bin/env python3
"""Gather the parts of the inference experiment, check completeness, and write the summaries and the verification
report.

    python3 collect_inference.py [--debug] [--out DIR] [--min-B 50] [--regret-curves G.tsv,B.tsv]

Writes, under OUT/collected/: one merged table per (block, regime) with every replicate row as the array wrote it
(with its seed), status.tsv (with the count of each task's recorded collections), environment.tsv, summary.tsv
(one row per block, regime, target, arm and grid point), regret.tsv (the two-track collection's cumulative expected
regret per block, regime and n), vcr.tsv (the Cramer-Rao comparison per block, regime, target and n, the displayed
choice simulation.PRIMARY, the design-stage split with cap 3/4), targets.tsv (per
target its true value mu*, ||A||, ||P_T A|| and their ratio, and at the largest n under the displayed choice the
mean V_pqv, mu* / V_pqv and the shares of the feasible and oracle intervals that exclude 0), records_BLOCK_REGIME.tsv
(one row per recorded collection, OUT/records: its seed, tau, the exploration sample's numbers -- the information
rule's kappa0, kappa_D, the responses' sum, the pilot's rank, the penalty score --, the CV choices, the solver caps and
the information rule's n0 at every n; replay_inference.py replays the collections themselves) and the report
inference_verdict.md.  summary.tsv also carries mu_star, excl0 (|mu_hat| > 1.96 V_hat), excl0_pqv (|mu_hat| > 1.96
V_pqv) and PT_ratio.
Every check below is made for every target separately (simulation.TARGET_NAMES; I3: its main effect).
--regret-curves: the regret study's collected/regret_curves.tsv (Gaussian, binary); with them regret_vs_single.tsv
compares the two tracks' regret with the single-track Algorithm 1 (A_t2) of the same case at the same n: the ratio of
means with a delta-method 95 % interval (unpaired: different replicates and B).

The statistic.  T = (mu_hat - mu*) / V_pqv, V_pqv the oracle standard error (the predictable quadratic variation of
the correction martingale: the procedure's own weights, the true conditional variance).  Every CLT quantity -- mean,
sd, shape, coverage, interval length -- uses V_pqv; the interval is mu_hat +- 1.96 V_pqv.  The procedure's own V_hat
enters only through the variance check V3c; the feasible interval mu_hat +- 1.96 V_hat is kept as a record
(summary.tsv cover_Vhat, length_Vhat).  The remainder mu_hat - mu* - martingale is recorded in its parts (Taylor
r_tay, normal r_norA + r_norz, calibration r_cal) over V_pqv.

The verification standard, applied to every choice of n0 (simulation.N0_CHOICES) alike, along the grid
(every n is the same runs stopped at n).  "end" = the block's largest n (I3: N); "upper half" = the grid points
n >= end / 2.  Monte-Carlo se: mean T sd/sqrt(B); sd T sd/sqrt(2(B-1)); coverage sqrt(0.95 x 0.05 / B); skewness
sqrt(6/B), excess kurtosis sqrt(24/B).  PASS / WATCH / FAIL; an arm's verdict on a block is its worst check.
  V1a  each remainder part (Taylor, normal, calibration, total) / V_pqv shrinks: no increase between consecutive
       upper-half points by more than 2 se (3 se: WATCH, beyond: FAIL); WATCH also if |end| > |first| (parts below
       0.01 count as zero)
  V1b  |total remainder| / V_pqv at the end <= 0.03 PASS, <= 0.05 WATCH, else FAIL
  V1c  |mean T| at the end <= 2 se PASS, <= 3 se WATCH
  V2a  KS p of T against N(0, 1) at the end >= 0.05 PASS, >= 0.01 WATCH (the QQ plot inside the band)
  V2b  skewness and excess kurtosis at the end within 2 se PASS, 3 se WATCH
  V2c  KS distance at the end <= at the first point, or <= the null 95 % value; else WATCH
  V3a  sd T at the end within 1 +- 2 se PASS, 3 se WATCH
  V3b  oracle coverage at the end within 0.95 +- 2 se PASS, 3 se WATCH
  V3c  the variance estimate: V_hat / V_pqv at the end in [0.98, 1.05] PASS, else WATCH
  V4   V1b, V3a, V3b at every upper-half point (a WATCH among many 2-se checks is expected by chance; the count and
       its expectation are printed; a FAIL anywhere is FAIL)
  N7   solver caps: replicates with a proximal-Newton solve at its iteration cap (logit link; the identity link's FISTA
       stops at its iteration limit without a flag), 0 PASS, <= 10 % WATCH
  N8   completeness: B observed >= B planned
The choices of n0 are then compared on the same replicates: per n, the RMS of the total remainder / V_pqv and the
oracle coverage of each choice, and each choice's largest excess over the best fixed share in (0, 1) at the same n (the
benchmark excludes n0 = n_tr, which has no refit and is outside the theorem).  The displayed choice is the
design-stage split with cap 3/4 (simulation.PRIMARY): the figures, the regret, the targets' table and the Cramer-Rao
comparison are its rows; every choice is judged by the same standard.  The rank arms of the appendix "unknown r0"
(with the displayed choice's n0) are reported in their own section, descriptively.
The one-step comparator (the inverse empirical Hessian; standardized by its Monte-Carlo sd, since it has no oracle
standard error) is in summary.tsv.  With B below --min-B (the debug plan) the checks V1-V4 are n/a.
"""
import argparse, csv, datetime, glob, os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
from scipy import stats
import simulation as S
import algorithms as A

Z_PASS, Z_WATCH = 2.0, 3.0
BIAS_PASS, BIAS_WATCH = 0.03, 0.05
KS_PASS, KS_WATCH = 0.05, 0.01
NEGLIGIBLE = 0.01
PARTS = ("taylor", "normal", "calibration", "total")
LEVEL = {"n/a": 0, "PASS": 1, "WATCH": 2, "FAIL": 3}


def worst(statuses):
    statuses = [s for s in statuses if s in LEVEL]
    return max(statuses, key=lambda s: LEVEL[s]) if statuses else "n/a"


def grade(z):
    """PASS / WATCH / FAIL from a distance in Monte-Carlo se."""
    return "PASS" if z <= Z_PASS else ("WATCH" if z <= Z_WATCH else "FAIL")


def write_tsv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def summary(G):
    """The summary of one grid point: G = the rows of one (block, regime, target, arm, n)."""
    col = lambda c: np.array([float(r[c]) for r in G])
    B = len(G)
    err, Vp, V = col("mu") - col("mu_star"), col("V_pqv"), col("V")
    T, TV = err / Vp, err / V
    part = {"taylor": col("r_tay") / Vp, "normal": (col("r_norA") + col("r_norz")) / Vp, "calibration": col("r_cal") / Vp}
    part["total"] = part["taylor"] + part["normal"] + part["calibration"]
    sd = T.std(ddof=1) if B > 1 else np.nan
    ks = stats.kstest(T, "norm") if B >= 5 else None
    ols = "mu_ols" in G[0]                                             # the one-step comparator (not on MIND)
    T_ols = col("T_ols") if ols else np.array([])
    T_ols = T_ols[np.isfinite(T_ols)]
    e_ols = col("mu_ols") - col("mu_star") if ols else np.array([])
    e_ols = e_ols[np.isfinite(e_ols)]
    Z_ols = e_ols / e_ols.std(ddof=1) if len(e_ols) > 2 else np.array([])
    out = dict(B=B, mean_T=T.mean(), se_mean=sd / np.sqrt(B), sd_T=sd, se_sd=sd / np.sqrt(2 * max(B - 1, 1)),
               skew=stats.skew(T) if B > 2 else np.nan, kurt=stats.kurtosis(T) if B > 3 else np.nan,
               ks_D=ks.statistic if ks else np.nan, ks_p=ks.pvalue if ks else np.nan,
               ks_null95=stats.kstwo.ppf(0.95, B) if B >= 5 else np.nan,
               cover=np.mean(np.abs(T) <= 1.959964), cover_Vhat=np.mean(np.abs(TV) <= 1.959964),
               se_cover=np.sqrt(0.95 * 0.05 / B), Vhat_over_Vpqv=np.mean(V / Vp), share=col("n0_frac").mean(),
               Vpqv_over_sdMC=Vp.mean() / err.std(ddof=1) if B > 2 else np.nan,
               length=np.mean(2 * 1.959964 * Vp), length_Vhat=np.mean(2 * 1.959964 * V),
               caps=int(np.sum(col("col_caps") + col("nuis_caps") > 0)),
               bias_sd=err.mean() / err.std(ddof=1) if B > 2 else np.nan,
               bias_sd_ols=Z_ols.mean() if len(Z_ols) else np.nan,
               ks_D_ols=stats.kstest(Z_ols, "norm").statistic if len(Z_ols) >= 5 else np.nan,
               cover_ols=np.mean(np.abs(Z_ols) <= 1.959964) if len(Z_ols) else np.nan,
               mean_T_ols=T_ols.mean() if len(T_ols) else np.nan, sd_T_ols=T_ols.std(ddof=1) if len(T_ols) > 1 else np.nan,
               cover_ols_own=np.mean(np.abs(T_ols) <= 1.959964) if len(T_ols) else np.nan)
    for k, v in part.items():
        out["rem_" + k] = v.mean()
        out["se_rem_" + k] = v.std(ddof=1) / np.sqrt(B) if B > 1 else np.nan
    out["rms_rem_total"] = float(np.sqrt(np.mean(part["total"] ** 2)))
    # the target's true value and whether the intervals exclude 0 (the practical reading of an interval)
    out.update(mu_star=col("mu_star").mean(), excl0=np.mean(np.abs(col("mu")) > 1.959964 * V),
               excl0_pqv=np.mean(np.abs(col("mu")) > 1.959964 * Vp))
    if G[0].get("A_norm", "") not in ("", None):                       # the tangent share ||P_T A|| / ||A||
        out.update(A_norm=col("A_norm").mean(), PT_ratio=col("PT_A").mean() / col("A_norm").mean())
    if G[0].get("V_CR", "") not in ("", None):                         # the Cramer-Rao comparison
        out.update(V_CR=col("V_CR").mean(), VCR_ratio=col("VCR_ratio").mean(),
                   se_VCR_ratio=col("VCR_ratio").std(ddof=1) / np.sqrt(B) if B > 1 else np.nan,
                   VCR_ratio_pred=col("VCR_ratio_pred").mean(), kappa_A=col("kappa_A").mean(),
                   kap_hat=col("kap_hat").mean(), PT_A=col("PT_A").mean())
    if "rank_0" in G[0]:                                               # the truncation ranks of the two folds
        r0, lo, hi = col("r0"), np.minimum(col("rank_0"), col("rank_1")), np.maximum(col("rank_0"), col("rank_1"))
        out.update(rank_ok=np.mean((lo == r0) & (hi == r0)), rank_under=np.mean(lo < r0), rank_over=np.mean(hi > r0))
    out["_T"] = T
    return out


def checks(S_):
    """The checks V1-V4 of the verification standard on one arm's summaries S_ along the grid (sorted by n)."""
    end, first = S_[-1], S_[0]
    upper = [s for s in S_ if s["n"] >= end["n"] / 2]
    out = []
    for k in PARTS:                                                   # V1a
        z_up = 0.0
        for a, b in zip(upper[:-1], upper[1:]):
            ma, mb = abs(a["rem_" + k]), abs(b["rem_" + k])
            if max(ma, mb) < NEGLIGIBLE:
                continue
            se = np.hypot(a["se_rem_" + k], b["se_rem_" + k])
            z_up = max(z_up, (mb - ma) / se if se > 0 else 0.0)
        v = grade(z_up)
        if abs(end["rem_" + k]) > abs(first["rem_" + k]) and abs(end["rem_" + k]) >= NEGLIGIBLE:
            v = worst([v, "WATCH"])
        out.append(("V1a", "%s part shrinks" % k, "first %+.3f -> end %+.3f; largest upper-half increase %.1f se"
                    % (first["rem_" + k], end["rem_" + k], z_up), v))
    b = abs(end["rem_total"])                                        # V1b
    out.append(("V1b", "|total remainder| / V_pqv at the end", "%.3f (se %.3f)" % (b, end["se_rem_total"]),
                "PASS" if b <= BIAS_PASS else "WATCH" if b <= BIAS_WATCH else "FAIL"))
    z = abs(end["mean_T"]) / end["se_mean"]                          # V1c
    out.append(("V1c", "mean T at the end", "%+.3f (%.1f se)" % (end["mean_T"], z), grade(z)))
    p = end["ks_p"]                                                  # V2a
    out.append(("V2a", "KS p at the end (QQ inside the band)", "%.3f" % p,
                "PASS" if p >= KS_PASS else "WATCH" if p >= KS_WATCH else "FAIL"))
    zs, zk = abs(end["skew"]) / np.sqrt(6 / end["B"]), abs(end["kurt"]) / np.sqrt(24 / end["B"])
    out.append(("V2b", "skewness, excess kurtosis at the end", "%+.2f (%.1f se), %+.2f (%.1f se)"
                % (end["skew"], zs, end["kurt"], zk), grade(max(zs, zk))))
    out.append(("V2c", "KS distance falls or is at noise level", "first %.3f -> end %.3f (null 95%% %.3f)"
                % (first["ks_D"], end["ks_D"], end["ks_null95"]),
                "PASS" if end["ks_D"] <= max(first["ks_D"], end["ks_null95"]) else "WATCH"))
    z = abs(end["sd_T"] - 1) / end["se_sd"]                          # V3a
    out.append(("V3a", "sd T at the end", "%.3f (%.1f se)" % (end["sd_T"], z), grade(z)))
    z = abs(end["cover"] - 0.95) / end["se_cover"]                   # V3b
    out.append(("V3b", "oracle coverage at the end", "%.3f (%.1f se)" % (end["cover"], z), grade(z)))
    r = end["Vhat_over_Vpqv"]                                        # V3c
    out.append(("V3c", "variance estimate V_hat / V_pqv at the end", "%.3f" % r,
                "PASS" if 0.98 <= r <= 1.05 else "WATCH"))
    vs, n_watch = [], 0                                              # V4
    for s in upper:
        b = abs(s["rem_total"])
        two_se = [grade(abs(s["sd_T"] - 1) / s["se_sd"]), grade(abs(s["cover"] - 0.95) / s["se_cover"])]
        vs += ["PASS" if b <= BIAS_PASS else "WATCH" if b <= BIAS_WATCH else "FAIL"] + two_se
        n_watch += sum(g == "WATCH" for g in two_se)
    out.append(("V4", "V1b, V3a, V3b at every upper-half point", "n = %s; WATCH %d of %d 2-se checks (%.1f expected)"
                % (", ".join("%d" % s["n"] for s in upper), n_watch, 2 * len(upper), 0.046 * 2 * len(upper)), worst(vs)))
    return out


def standard(summ, B_planned, min_B):
    """The verification standard on one arm's summaries along the grid: the checks (V1-V4 only with B >= min_B,
    then N7 and N8) and the verdict, the worst check ("n/a" below min_B unless N7 or N8 fails)."""
    B = summ[-1]["B"]
    caps = sum(s["caps"] for s in summ)
    n7 = "PASS" if caps == 0 else ("WATCH" if caps <= 0.10 * B * len(summ) or B < min_B else "FAIL")
    n8 = "PASS" if B >= B_planned else "FAIL"
    C = [("N7", "solver caps (replicate x grid point)", "%d" % caps, n7),
         ("N8", "completeness", "B = %d of %d planned" % (B, B_planned), n8)]
    if B < min_B:
        return C, ("n/a" if worst([n7, n8]) == "PASS" else worst([n7, n8]))
    C = checks(summ) + C
    return C, worst([c[3] for c in C])


N0_CHOICES = tuple(a for a, _ in S.N0_CHOICES)                      # the choices of n0, compared on equal footing
LABEL = {"info_rule": "information rule (n0_rule)", "no_refit": "n0 = n_tr: no refit (outside the theorem)",
         **{a: "n0 = %s n_tr" % how for a, how in S.N0_CHOICES if a.startswith("share")},
         **{a: "design-stage split, cap %s" % how.split()[1] for a, how in S.N0_CHOICES if a.startswith("design")}}
SHORT = {"info_rule": "rule", "no_refit": "s100", **{a: a.replace("share_", "s").replace("design_", "d")
                                                     for a, _ in S.N0_CHOICES if a.startswith(("share", "design"))}}


def md_n0_comparison(key, summ_by_arm, choices):
    """The comparison of the choices of n0 on one block: per n, the RMS of the total remainder / V_pqv and the oracle
    coverage; then each choice's largest excess over the best fixed share in (0, 1) at the same n (RMS)."""
    ns = [s["n"] for s in summ_by_arm[choices[0]]]
    get = lambda arm, n, f: next((s[f] for s in summ_by_arm[arm] if s["n"] == n), np.nan)
    head = ["| %s | %s |" % (key, " | ".join(SHORT[a] for a in choices)), "|---|" + "---|" * len(choices)]
    lines = ["RMS of the total remainder / V_pqv:", ""] + head
    lines += ["| %d | %s |" % (n, " | ".join("%.3f" % get(a, n, "rms_rem_total") for a in choices)) for n in ns]
    shares = [a for a in choices if a.startswith("share")]
    if shares:
        best = {n: min(get(a, n, "rms_rem_total") for a in shares) for n in ns}
        lines += ["| largest excess over the best share in (0, 1) | %s |" % " | ".join(
            "%.3f" % max(get(a, n, "rms_rem_total") - best[n] for n in ns) for a in choices)]
    lines += ["", "Oracle coverage:", ""] + head
    lines += ["| %d | %s |" % (n, " | ".join("%.3f" % get(a, n, "cover") for a in choices)) for n in ns]
    return lines + [""]


def md_table(key, summ, C):
    """The markdown of one arm of one block: the per-n table, then its checks C."""
    lines = ["| %s | B | mean T (se) | sd T | skew | kurt | KS p | cover | V_hat/V_pqv | Taylor | normal | calib | total (se) | cover V_hat (record) |" % key,
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += ["| %d | %d | %+.3f (%.3f) | %.3f | %+.2f | %+.2f | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.4f | %+.3f (%.3f) | %.3f |"
              % (s["n"], s["B"], s["mean_T"], s["se_mean"], s["sd_T"], s["skew"], s["kurt"], s["ks_p"], s["cover"],
                 s["Vhat_over_Vpqv"], s["rem_taylor"], s["rem_normal"], s["rem_calibration"],
                 s["rem_total"], s["se_rem_total"], s["cover_Vhat"]) for s in summ]
    lines += ["", "| check | criterion | value | verdict |", "|---|---|---|---|"]
    return lines + ["| %s | %s | %s | %s |" % c for c in C] + [""]


def targets_of(rows):
    """The targets of a table, in the order of the rows."""
    return list(dict.fromkeys(r["target"] for r in rows))


REGRET_CASE = {("I1", "lowrank"): "sph_d16_r2_K16_rho020", ("I1", "normal"): "nor_d16_r2_K16_rho020",
               ("I2l", "lowrank"): "sph_d16_r2_K16_nb01", ("I2l", "normal"): "nor_d16_r2_K16_nb01"}
REGRET_N_MAX = 10000                                                 # the regret study's horizon


def record_rows(out, block, regime):
    """One row per recorded collection of (block, regime) (OUT/records): the record's numbers and the information
    rule's n0 at every n (I3: one collection per N, its N and n0)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "records", "%s_%s_r*.npz" % (block, regime)))):
        z = np.load(path)
        k = list(z["arms"]).index("info_rule")
        row = dict(rep=int(z["rep"]), seed=int(z["seed"]), tau=int(z["tau"]), horizon=int(z["horizon"]),
                   kappa0=float(z["kappa0"]), kappa_D=float(z["kappa_D"]), y_ex_sum=float(z["y_ex_sum"]),
                   pilot_rank=int(z["pilot_rank"]), score=float(z["score"]),
                   cv_choices=" ".join("%g" % c for c in np.ravel(z["cv_c"])), solver_caps=int(z["solver"][2]))
        if block == "I3":
            row.update(N=int(z["N"][0]), n0_info_rule=int(z["n0"][0, k]))
        else:
            row.update({"n0_info_rule_n%d" % (N + row["tau"]): int(z["n0"][i, k]) for i, N in enumerate(z["N"])})
        rows.append(row)
    return rows


def regret_tables(merged, curves_paths):
    """The two-track collection's cumulative expected regret (rows of the displayed choice and the first target: the
    collection does not depend on the arm or the target), per block, regime and n; and, with the regret
    study's curves, the ratio to the single-track Algorithm 1 (A_t2) of the same case at the same n: of the total
    regret, and of the regret after the exploration (both runs explore tau rounds; the single track's exploration
    regret is the summary.tsv beside its curves).  Where the regret per round decays like s^(-1/2) the second ratio is
    near sqrt(2); where the regret after exploration is concentrated in the first rounds (each track then pays its own
    learning phase) it is near 2."""
    reg, comp = [], []
    single, single_ex = {}, {}
    for path in curves_paths:
        if not os.path.exists(path):                                   # the regret run is not collected (yet)
            print("regret curves %s: not found, the comparison with the single track is left out" % path)
            continue
        for r in csv.DictReader(open(path), delimiter="\t"):
            if r["config_id"] == "A_t2":
                single[(r["case_id"], int(float(r["t"])))] = (float(r["mean"]), float(r["se"] or "nan"), int(float(r["B"])))
        summary = os.path.join(os.path.dirname(path), "summary.tsv")
        for r in (csv.DictReader(open(summary), delimiter="\t") if os.path.exists(summary) else []):
            if r["config_id"] == "A_t2":
                single_ex[r["case_id"]] = float(r["exploration"])
    for (block, regime), rows in sorted(merged.items()):
        if "regret" not in rows[0] or rows[0]["regret"] == "":
            continue
        first = targets_of(rows)[0]
        key = "N" if block == "I3" else "n"
        g = {}
        for r in rows:
            if r["arm"] == S.PRIMARY and r["target"] == first:
                g.setdefault(int(float(r[key])), []).append(r)
        for n, G in sorted(g.items()):
            v = {c: np.array([float(r[c]) for r in G]) for c in ("regret", "regret_ex", "regret_t0", "regret_t1")}
            B = len(G)
            row = dict(block=block, regime=regime, n=n, B=B, rounds=int(float(G[0]["regret_rounds"])),
                       regret=v["regret"].mean(), se=v["regret"].std(ddof=1) / np.sqrt(B) if B > 1 else np.nan,
                       regret_ex=v["regret_ex"].mean(), regret_t0=v["regret_t0"].mean(), regret_t1=v["regret_t1"].mean())
            reg.append(row)
            case = REGRET_CASE.get((block, regime))
            if case and key == "n" and (case, n) in single and n <= REGRET_N_MAX:
                m1, se1, B1 = single[(case, n)]
                R = row["regret"] / m1
                se_R = R * np.sqrt((row["se"] / row["regret"]) ** 2 + (se1 / m1) ** 2)
                c = dict(block=block, regime=regime, n=n, case=case, B_two=B, B_single=B1,
                         two_track=row["regret"], se_two=row["se"], single=m1, se_single=se1, ratio=R,
                         ci_lo=R - 1.959964 * se_R, ci_hi=R + 1.959964 * se_R,
                         tau_two=int(float(G[0]["tau"])), tau_single=A.tau_rule(REGRET_N_MAX, 16, 16, S.C_TAU))
                if case in single_ex:              # after the exploration (the single track's se: its total's)
                    post = v["regret"] - v["regret_ex"]
                    p2, p1 = post.mean(), m1 - single_ex[case]
                    se_p2 = post.std(ddof=1) / np.sqrt(B) if B > 1 else np.nan
                    P = p2 / p1
                    se_P = P * np.sqrt((se_p2 / p2) ** 2 + (se1 / p1) ** 2)
                    c.update(two_post=p2, single_post=p1, ratio_post=P, post_lo=P - 1.959964 * se_P,
                             post_hi=P + 1.959964 * se_P)
                comp.append(c)
    return reg, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-B", type=int, default=50)
    ap.add_argument("--regret-curves", default=None, type=lambda v: v.split(","),
                    help="the regret study's collected/regret_curves.tsv files (Gaussian, binary)")
    a = ap.parse_args()
    blocks = S.DEBUG_BLOCKS if a.debug else S.BLOCKS
    out = a.out or ("inference_debug" if a.debug else "inference_results")
    collected = os.path.join(out, "collected")
    os.makedirs(collected, exist_ok=True)
    # ---- merge the parts, one table per (block, regime)
    merged, status = {}, []
    for task in S.tasks(blocks):
        path = os.path.join(out, "parts", task["key"] + ".tsv")
        reps = set()
        if os.path.exists(path):
            rows = list(csv.DictReader(open(path), delimiter="\t"))
            merged.setdefault((task["block"], task["regime"]), []).extend(rows)
            reps = {int(r["rep"]) for r in rows}
        expected = task["hi"] - task["lo"] + 1
        records = sum(len(glob.glob(os.path.join(out, "records", "%s_%s_r%04d*.npz" % (task["block"], task["regime"], rep))))
                      for rep in range(task["lo"], task["hi"] + 1))
        status.append(dict(key=task["key"], replicates=len(reps), planned=expected, records=records,
                           status="ok" if len(reps) == expected else ("partial" if reps else "missing")))
    write_tsv(status, os.path.join(collected, "status.tsv"))
    bad = [s for s in status if s["status"] != "ok"]
    print("tasks: %d complete, %d partial, %d missing of %d" % (
        len(status) - len(bad), sum(s["status"] == "partial" for s in bad), sum(s["status"] == "missing" for s in bad), len(status)))
    for s in bad:
        print("  %-8s %s (%d/%d replicates)" % (s["status"], s["key"], s["replicates"], s["planned"]))
    print("records: %d collections recorded (OUT/records)" % sum(s["records"] for s in status))
    for (block, regime), rows in merged.items():
        write_tsv(rows, os.path.join(collected, "%s_%s.tsv" % (block, regime)))
        rec_rows = record_rows(out, block, regime)
        if rec_rows:
            write_tsv(rec_rows, os.path.join(collected, "records_%s_%s.tsv" % (block, regime)))
    seen = sorted({tuple(r.get(k, "") for k in ("python", "numpy", "scipy")) for rows in merged.values() for r in rows})
    if seen and any(v[0] for v in seen):              # provenance: the versions the replicates were computed with
        write_tsv([dict(zip(("python", "numpy", "scipy"), v)) for v in seen if v[0]], os.path.join(collected, "environment.tsv"))
    if not merged:
        return
    # ---- per (block, regime, target, arm, grid point) summaries; the standard on every choice of n0 and target
    table, verdicts, md_arms, md_n0, md_rank, consts, md_vcr, vcr_rows = [], {}, [], [], [], [], [], []
    target_rows = []                                     # targets.tsv: each target's truth at the largest n
    for (block, regime), rows_all in sorted(merged.items()):
        law = block if regime in ("lowrank", "tie") else "%s (%s)" % (block, regime)   # one entry per covariate law
        used = sorted({(r.get("c_tau", "?") or "tie: %d rounds" % S.TIE_TAU, r.get("c_lam", "?")) for r in rows_all})
        consts.append("%s: c_tau %s, c_lam %s" % (law, " / ".join(u[0] for u in used), " / ".join(u[1] for u in used)))
        key = "N" if block == "I3" else "n"
        unit = "N post-exploration" if key == "N" else "n total rounds"
        for target in targets_of(rows_all):
            rows = [r for r in rows_all if r["target"] == target]
            label = "%s, %s" % (law, target.replace("_", " "))
            summ_by_arm = {}
            for arm, _ in S.arms(block):
                groups = {}
                for r in rows:
                    if r["arm"] == arm:
                        groups.setdefault(int(float(r[key])), []).append(r)
                summ = []
                for n, G in sorted(groups.items()):
                    s = summary(G)
                    s.update(block=block, regime=regime, target=target, arm=arm, n=n)
                    summ.append(s)
                    table.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items() if k != "_T"})
                if summ:
                    summ_by_arm[arm] = summ
            if summ_by_arm.get(S.PRIMARY):                # the target's true value, tangent share, interval at the end
                e = summ_by_arm[S.PRIMARY][-1]
                V_end = e["length"] / (2 * 1.959964)
                target_rows.append(dict(block=block, regime=regime, target=target, mu_star=e["mu_star"],
                                        A_norm=e.get("A_norm", ""), PT_A=e.get("PT_A", ""),
                                        PT_ratio=e.get("PT_ratio", ""), n=e["n"], B=e["B"], V_pqv=V_end,
                                        mu_star_over_V=e["mu_star"] / V_end, excl0=e["excl0"],
                                        excl0_pqv=e["excl0_pqv"], cover=e["cover"]))
            choices = [arm for arm in N0_CHOICES if arm in summ_by_arm]
            for arm in choices:                                          # the standard, on every choice of n0
                C, v = standard(summ_by_arm[arm], blocks[block]["B"], a.min_B)
                verdicts[(label, arm)] = v
                md_arms += ["### %s (%s), %s: %s" % (label, unit, LABEL[arm], v), ""] + md_table(key, summ_by_arm[arm], C)
            if len(choices) > 1:
                md_n0 += ["### %s (%s)" % (label, unit), ""] + md_n0_comparison(key, summ_by_arm, choices)
            known = {s["n"]: s for s in summ_by_arm.get(S.PRIMARY, [])}    # the rank arms' reference: the known rank
            for arm in [arm for arm in S.RANK_ARMS if arm in summ_by_arm]:                     # descriptive
                summ = summ_by_arm[arm]
                md_rank += ["### %s (%s), %s rank (price %s)" % (label, unit, S.RANK_ARMS[arm].upper(),
                                                                  "log(s)/s" if arm == "rank_gic" else "1/s"), "",
                            "| %s | B | both folds r0 | a fold < r0 | a fold > r0 | cover | sd T | mean T | total rem (se) "
                            "| length / known r0 | cover V_hat (record) |" % key, "|---|---|---|---|---|---|---|---|---|---|---|"]
                md_rank += ["| %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | %+.3f (%.3f) | %s | %.3f |"
                            % (s["n"], s["B"], s["rank_ok"], s["rank_under"], s["rank_over"], s["cover"], s["sd_T"],
                               s["mean_T"], s["rem_total"], s["se_rem_total"],
                               "%.3f" % (s["length"] / known[s["n"]]["length"]) if s["n"] in known else "-",
                               s["cover_Vhat"]) for s in summ] + [""]
            if known and "VCR_ratio" in next(iter(known.values())):     # the Cramer-Rao comparison
                md_vcr += ["### %s (%s), %s" % (label, unit, LABEL[S.PRIMARY]), "",
                           "| %s | B | V_pqv / V_CR (se) | predicted sqrt(C_cal kappa_A log n_tr / kappa_hat) | kappa_A "
                           "| kappa_hat | V_CR | norm of P_T A |" % key, "|---|---|---|---|---|---|---|---|"]
                md_vcr += ["| %d | %d | %.3f (%.3f) | %.3f | %.3f | %.3f | %.4g | %.3f |"
                           % (s["n"], s["B"], s["VCR_ratio"], s["se_VCR_ratio"], s["VCR_ratio_pred"], s["kappa_A"],
                              s["kap_hat"], s["V_CR"], s["PT_A"]) for s in known.values()] + [""]
                vcr_rows += [dict(block=block, regime=regime, target=target, n=s["n"], B=s["B"], VCR_ratio=s["VCR_ratio"],
                                  se_VCR_ratio=s["se_VCR_ratio"], VCR_ratio_pred=s["VCR_ratio_pred"], kappa_A=s["kappa_A"],
                                  kap_hat=s["kap_hat"], V_CR=s["V_CR"], PT_A=s["PT_A"]) for s in known.values()]
    write_tsv(table, os.path.join(collected, "summary.tsv"))
    if target_rows:
        write_tsv(target_rows, os.path.join(collected, "targets.tsv"))
    if vcr_rows:
        write_tsv(vcr_rows, os.path.join(collected, "vcr.tsv"))
    debug = all(s["B"] < a.min_B for s in table)
    labels = sorted({l for l, _ in verdicts})
    cols = [c for c in N0_CHOICES if any((l, c) in verdicts for l in labels)]
    verdict_table = ["| block | %s |" % " | ".join(SHORT[c] for c in cols), "|---|" + "---|" * len(cols),
                     *["| %s | %s |" % (l, " | ".join(verdicts.get((l, c), "-") for c in cols)) for l in labels]]
    debug_line = []                                                  # the debug plan: is the pipeline sound?
    if debug:
        debug_line = ["## Verdict: %s" % (
            "DEBUG PASS: every task produced replicates and the tables build; no statistical claim (B below %d)" % a.min_B
            if worst(list(verdicts.values())) in ("n/a", "PASS") else
            "DEBUG with a note: a structural check (caps or completeness) is not PASS -- see the tables"), ""]
    md = ["# Inference experiment: verification", "",
          "Generated %s by collect_inference.py from %d tables in `%s`." % (
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), len(merged), collected), "", *debug_line,
          "## Verdicts by choice of n0 (the same standard for every choice)", "", *verdict_table, "",
          "Columns: %s." % "; ".join("%s = %s" % (SHORT[c], LABEL[c]) for c in cols), "",
          "Constants: C_ref 1/%g, C_cal %g; c0 1/%g in both splits (the information rule and the design-stage split); "
          "the collection's (c_tau, c_lam) by block -- %s.  T = (mu_hat - mu*) / V_pqv (oracle); cover = the oracle interval "
          "mu_hat +- 1.96 V_pqv; the remainder columns are over V_pqv; V_hat is judged only through V_hat/V_pqv "
          "(check V3c); cover V_hat is the feasible interval, kept as a record; the standard is in the header of "
          "collect_inference.py." % (1 / A.C_REF, A.C_CAL, 1 / A.C0, "; ".join(consts)), ""]
    if md_n0:
        md += ["## The choices of n0 compared on the same replicates", "", *md_n0]
    md += ["## Each choice of n0 under the verification standard", "", *md_arms]
    if md_rank:
        md += ["## Appendix: unknown r0 (the truncation rank chosen by GIC or an AIC-type criterion)", "",
               "The same replicates and n0 as the displayed choice (%s); in each fold the rank is chosen on that fold's "
               "pilot data at its nuisance fit (algorithms.rank_rule); n0 and tau keep the declared rank.  Where both "
               "folds choose r0 the replicate equals the displayed choice's exactly.  Descriptive." % LABEL[S.PRIMARY],
               "", *md_rank]
    if md_vcr:
        md += ["## The Cramer-Rao comparison", "",
               "V_CR: the standard error of the efficient estimator if the collected design had been fixed in advance "
               "and the tangent space known (algorithms.cramer_rao; offline).  The prediction from the variance expansion "
               "of the theorem is V_pqv / V_CR = sqrt(C_cal kappa_A log n_tr / kappa_hat): the sqrt(log n) price of the "
               "predictable calibration.  Descriptive.", "", *md_vcr]
    reg, comp = regret_tables(merged, a.regret_curves or [])
    if reg:
        write_tsv(reg, os.path.join(collected, "regret.tsv"))
        md += ["## The regret of the two-track collection", "",
               "Cumulative expected regret of the rounds the inference at n uses (tau exploration rounds and n_tr rounds "
               "of each track), true means, the probability scale under the logit link.", "",
               "| block | regime | n | B | regret (se) | exploration | track 0 | track 1 |", "|---|---|---|---|---|---|---|---|",
               *["| %s | %s | %d | %d | %.2f (%.2f) | %.2f | %.2f | %.2f |" % (r["block"], r["regime"], r["n"], r["B"],
                 r["regret"], r["se"], r["regret_ex"], r["regret_t0"], r["regret_t1"]) for r in reg], ""]
    if comp:
        write_tsv(comp, os.path.join(collected, "regret_vs_single.tsv"))
        md += ["Against the single-track Algorithm 1 of the regret study (A_t2, the same case; unpaired, delta-method "
               "95 % interval of the ratio of means):", "",
               "| block | regime | n | two tracks | single track | ratio [95 % CI] | after exploration: two / single, "
               "ratio [95 % CI] | tau two / single |",
               "|---|---|---|---|---|---|---|---|",
               *["| %s | %s | %d | %.2f | %.2f | %.3f [%.3f, %.3f] | %s | %d / %d |" % (r["block"], r["regime"], r["n"],
                 r["two_track"], r["single"], r["ratio"], r["ci_lo"], r["ci_hi"],
                 "%.2f / %.2f, %.3f [%.3f, %.3f]" % (r["two_post"], r["single_post"], r["ratio_post"], r["post_lo"],
                                                    r["post_hi"]) if "ratio_post" in r else "-",
                 r["tau_two"], r["tau_single"]) for r in comp],
               "", "The ratio after the exploration is near sqrt(2) = 1.414 where the regret per round decays like "
               "s^(-1/2) (regret of order sqrt(n)), and near 2 where the regret after the exploration is concentrated "
               "in the first rounds (each track pays its own learning phase).", ""]
    md += ["The one-step comparator (standardized by its Monte-Carlo sd: bias_sd_ols, ks_D_ols, cover_ols) is in "
           "summary.tsv."]
    open(os.path.join(collected, "inference_verdict.md"), "w").write("\n".join(md))
    print("\n".join(debug_line + verdict_table))
    print("report written:", os.path.join(collected, "inference_verdict.md"))


if __name__ == "__main__":
    main()
