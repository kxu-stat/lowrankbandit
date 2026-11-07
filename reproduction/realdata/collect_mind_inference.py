#!/usr/bin/env python3
"""Collect the MIND inference study (run_mind_inference.py), by hand after the
array:

    python3 collect_mind_inference.py PHASE --setup SETUP_NPZ [--dir DIR] [--min-B 50] [--incomplete]
                                      [--arm ARM] [--category CATEGORY]

The numbers are defined exactly as in the inference simulations, by inference/collect_inference.py's own functions
(summary, standard, md_table, md_n0_comparison): T = (mu_hat - mu*) / V_pqv, V_pqv the oracle standard error; the
oracle interval mu_hat +- 1.96 V_pqv is judged; the feasible interval mu_hat +- 1.96 V_hat is reported and enters the
checks only through V_hat / V_pqv (V3c).  Each world is collected on its own (one world's cells in memory at a time).
The simulations' verification standard (V1a-V4, N7, N8; the header of collect_inference.py) is applied to every target
under every choice of n0 alike, with end = the largest n and upper half = n >= end / 2, and the choices of n0 are
compared on the same streams, for each kind of target pooled over its 8 categories.

The reported n of a world and a choice of n0 (a rule fixed before the run): the smallest n of the grid from which
on -- at n and at every larger n -- each kind of target (or, orx, ort, lvl) has, averaged over its 8 categories, an
absolute deviation of mean T from 0, of sd T from 1 and of the oracle coverage from 0.95 within 2 Monte-Carlo
standard errors of one target of B streams: 2 / sqrt(B), 2 / sqrt(2 (B - 1)) and 2 sqrt(0.95 x 0.05 / B), that is
0.089, 0.063 and 0.019 at B = 500.  If the largest n does not pass, the rule is not met and the largest n is reported
as such.

Writes, under DIR/collected/:
  status.tsv                   every planned stream of both worlds: ok, error, stale (made under another setup or
                               package version), missing
  reported_n.md                the reported n of every world and choice of n0; for the choice ARM, the three
                               averaged deviations of every kind of target at every n
  reported_n.pdf/.png          the same deviations of the choice ARM against n, both worlds, with the rule's bounds
and under DIR/collected/WORLD/, for each world:
  rows.tsv.gz                  every stream, n, choice of n0 and target, as the array wrote it
  summary.tsv                  one row per target, choice of n0 and n (with mu_star, excl0, PT_ratio)
  targets.tsv                  per target its true value mu*, ||A||, ||P_T A|| and their ratio, and at the
                               largest n under the choice ARM the mean V_pqv, mu* / V_pqv and the shares of the
                               feasible and oracle intervals that exclude 0
  streams.tsv                  per stream its seed, tau, the exploration sample's numbers (the information rule's
                               kappa0, the exploration clicks, the pilot's rank, the penalty score), the CV choices and
                               the information rule's n0 at every n (the cells' records; the other choices' splits
                               are the same in every stream)
  verdict_mind_inference.md    the choices of n0 compared, and the verdict of every target under every choice
  details_mind_inference.md    per target and choice: the table along n and the checks
  n0_choice.pdf/.png           per kind of target, against n, one line per choice of n0: the RMS of the total
                               remainder / V_pqv over the 8 categories and the streams, and the oracle coverage
                               averaged over the 8 categories (band: 0.95 +- 2 se of one target)
  n0_share.pdf/.png            against n0 / n_tr, one line per n (10,000, 20,000, 40,000, 80,000 when in the grid):
                               mean T, sd T and the coverage of mu_hat +- 1.96 V_hat, averaged over the 32 targets;
                               no refit at 1 (open), the choice ARM at its average share (star) and the information
                               rule at its average share (cross)
  decomposition.pdf/.png       every target at the world's figure n (run_mind_inference.FIGURE_N: 40,000 in the
                               down-sampled world, 80,000 in the main world) for
                               the choice ARM:
                               mean T, the mean of its martingale part sum_t omega_t (y_t - p_t) / V_pqv (exact mean
                               0; its spread is the streams' Monte Carlo error) and the mean of the remainder
                               (Taylor + normal + calibration) / V_pqv, the estimator's error; T = the two parts
                               exactly; +- 2 se
  regret.tsv                   the two-track collection's cumulative expected regret per n, over the streams
  vcr.tsv                      per kind of target and n, the choice ARM: V_pqv / V_CR against its
                               prediction (algorithms.cramer_rao), the 8 categories averaged
  ARM/inference_trend.pdf/.png, ARM/inference_qq.pdf/.png
                               the simulations' trend and QQ figures (figures_inference.trend, qq) of the choice ARM
                               (default design_75, the design-stage split with cap 3/4, the displayed choice), one row
                               per kind,
                               for the category CATEGORY (default news, the category with the most displayed pairs),
                               QQ plots at every n
  ARM/paper_qq_CATEGORY_n<n>.pdf/.png
                               the paper's figure: the four kinds of target of CATEGORY at the world's figure n,
                               a QQ plot of T above its histogram, one column per kind
Each world has its own grid (run_mind_inference.GRID; the same in both worlds).
"""
import argparse, csv, gzip, json, os, sys
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
import run_mind_inference as RMI              # also puts ../inference on the path
import collect_inference as CI                # inference/collect_inference.py
import figures_inference as FI                # inference/figures_inference.py

KINDS = {"or": "log odds ratio, readers vs all users", "orx": "its user-vector part",
         "ort": "its interaction part", "lvl": "readers' content log-odds"}       # setup_mind.targets
WORLD_LABEL = {"main": "main world", "down4": "down-sampled world (1/4 of the non-clicks)"}
WORLD_TEXT = {"main": "the main world of setup_mind.py",
              "down4": "the down-sampled world: the click model of MIND's logs after keeping every click and 1/4 of "
                       "the non-clicks, i.e. the main truth with every log-odds raised by log 4 (mind.downsampled), "
                       "on the same rounds, every round kept"}
WORLD_STYLE = {"main": dict(color="#1a1a19", marker="o"), "down4": dict(color="#2a78d6", marker="s")}


def status_of(phase, out, setup_sha):
    """The status of every planned stream of both worlds: ok, error, stale or missing."""
    status = []
    for tk in RMI.tasks(phase):
        path = os.path.join(out, "cells", tk["key"] + ".json")
        if not os.path.exists(path):
            status.append(dict(key=tk["key"], status="missing", elapsed="", error=""))
            continue
        z = json.load(open(path))
        if z["status"] == "ok" and z.get("stamp") != RMI.stamp(setup_sha):
            z["status"], z["error"] = "stale", "made under another setup or package version"
        status.append(dict(key=tk["key"], status=z["status"], elapsed=z["elapsed"],
                           error=z["error"].strip().split("\n")[-1] if z["error"] else ""))
    return status


def rows_of(phase, out, setup_sha, world):
    """The rows of the ok streams of one world (one per stream, n, choice of n0 and target, with the stream's world,
    replicate number, seed and the collection's solver caps), their tau and timings, and one record per stream
    (the exploration sample's numbers, the CV choices and the information rule's n0 at every n)."""
    rows, times, streams = [], [], []
    for tk in RMI.tasks(phase):
        path = os.path.join(out, "cells", tk["key"] + ".json")
        if tk["world"] != world or not os.path.exists(path):
            continue
        z = json.load(open(path))
        if z["status"] != "ok" or z.get("stamp") != RMI.stamp(setup_sha):
            continue
        rows += [dict(r, world=world, rep=tk["replication"], seed=tk["seed"], col_caps=z["solver"]["caps"])
                 for r in z["results"]]
        times.append(dict(tau=z["tau"], t_collect=z["t_collect"], t_infer=z["t_infer"]))
        streams.append(dict(rep=tk["replication"], seed=tk["seed"], tau=z["tau"], kappa0=z["kappa0"],
                            kappa_D=z["kappa_D"], clicks_ex=z["clicks_ex"], pilot_rank=z["pilot_rank"],
                            score=z["score"], cv=json.dumps(z["cv"]), t_collect=z["t_collect"],
                            **{"n0_info_rule_%s" % n: v["info_rule"] for n, v in z["n0"].items()}))
    return rows, times, streams


def martingale_part(G):
    """The mean of T's martingale part, T minus the remainder (Taylor + normal + calibration) / V_pqv, and its se."""
    col = lambda c: np.array([float(r[c]) for r in G])
    m = (col("mu") - col("mu_star")) / col("V_pqv") - FI.parts(G)["total"]
    return dict(mean_mart=float(m.mean()), se_mart=float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else np.nan)


def of_kind(names, kind):
    return [name for name in names if name.split("_")[0] == kind]


def n0_choice(groups, names, choices, grid, out):
    """Per kind of target, against n, one line per choice of n0: the RMS of the total remainder / V_pqv over the 8
    categories and the streams, and the oracle coverage averaged over the 8 categories."""
    fig, axes = FI.plt.subplots(len(KINDS), 2, figsize=(6.4, 1.6 * len(KINDS) + 0.6), squeeze=False, layout="constrained")
    for i, kind in enumerate(KINDS):
        mine = of_kind(names, kind)
        for arm in choices:
            st = dict(dict(lw=1.0, ls="-", marker="o", ms=2.2), **FI.N0_STYLE[arm])
            rms = [np.sqrt(np.mean(np.concatenate([FI.parts(groups[name, arm, n])["total"] for name in mine]) ** 2))
                   for n in grid]
            cover = [np.mean([np.mean(np.abs(FI.oracle_T(groups[name, arm, n])) <= FI.Z95) for name in mine]) for n in grid]
            axes[i, 0].plot(grid, rms, label=FI.N0_LABEL[arm], **st)
            axes[i, 1].plot(grid, cover, **st)
        se_c = np.sqrt(0.95 * 0.05 / len(groups[mine[0], choices[0], grid[-1]]))
        axes[i, 1].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=FI.GRID, lw=0)
        axes[i, 1].axhline(0.95, color=FI.INK2, lw=0.6)
        axes[i, 0].set_yscale("log")
        for ax in axes[i]:
            FI.n_axis(ax, grid)
        axes[i, 0].set_ylabel("%s (%s)" % (kind, KINDS[kind]))
    axes[0, 0].set_title(r"RMS of total remainder / $V_{pqv}$, 8 categories")
    axes[0, 1].set_title(r"oracle coverage, mean of 8 categories")
    for ax in axes[-1]:
        ax.set_xlabel("total rounds $n$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=5.5)
    FI.save(fig, out, "n0_choice")


SHARE_OF = {"share_10": 0.1, "share_30": 0.3, "share_50": 0.5, "share_70": 0.7, "share_90": 0.9}
SHARE_N = (10000, 20000, 40000, 80000)              # the n of n0_share (the grid's last four when none is in it)
SEQ = ("#a9cdf2", "#6aa6e8", "#2a78d6", "#123f7a")  # one hue, light to dark: n increasing


def n0_share(summ, names, grid, arm, out):
    """Against n0 / n_tr, one line per n: mean T, sd T and the coverage of mu_hat +- 1.96 V_hat, averaged over the
    targets; no refit (n0 = n_tr) at 1, open; the choice arm at its average share, a star, and the information rule at
    its average share, a cross.  Band: +- 2 Monte-Carlo se of one target."""
    ns = [n for n in grid if n in SHARE_N] or list(grid)[-4:]
    at = lambda c, n, key: np.mean([next(s for s in summ[name, c] if s["n"] == n)[key] for name in names])
    B = min(next(s for s in summ[name, arm] if s["n"] == ns[0])["B"] for name in names)
    fig, axes = FI.plt.subplots(1, 3, figsize=(6.4, 2.4), layout="constrained")
    for i, n in enumerate(ns):
        color = SEQ[i % len(SEQ)]
        for ax, key in zip(axes, ("mean_T", "sd_T", "cover_Vhat")):
            ys = [at(c, n, key) for c in SHARE_OF]
            ax.plot(list(SHARE_OF.values()), ys, "-o", ms=2.5, lw=1.0, color=color, label="n = %d" % n)
            ax.plot([0.9, 1.0], [ys[-1], at("no_refit", n, key)], ":", lw=0.7, color=color)
            ax.plot([1.0], [at("no_refit", n, key)], "o", ms=3, mfc="white", mec=color)
            ax.plot([at(arm, n, "share")], [at(arm, n, key)], "*", ms=6, color=color, mec="#1a1a19", mew=0.4)
            ax.plot([at("info_rule", n, "share")], [at("info_rule", n, key)], "x", ms=4, color=color, mew=0.8)
    for ax, key, centre, se in ((axes[0], "mean_T", 0, 1 / np.sqrt(B)), (axes[1], "sd_T", 1, 1 / np.sqrt(2 * (B - 1))),
                                (axes[2], "cover_Vhat", 0.95, np.sqrt(0.95 * 0.05 / B))):
        ax.axhspan(centre - 2 * se, centre + 2 * se, color=FI.GRID, lw=0)
        ax.axhline(centre, color=FI.INK2, lw=0.6)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        ax.set_xticklabels(["0.1", "0.3", "0.5", "0.7", "0.9", "1\n(no refit)"])
        ax.set_xlabel(r"$n_0 / n_{\mathrm{tr}}$")
    axes[0].set_title("mean $T$, %d targets" % len(names))
    axes[1].set_title("sd $T$")
    axes[2].set_title(r"coverage of $\hat\mu \pm 1.96 \hat V$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(ns), fontsize=6)
    fig.suptitle("star: %s at its average share; cross: the information rule; open: no refit; band: 2 se of one target "
                 "(B = %d)" % (CI.LABEL[arm], B), fontsize=6)
    FI.save(fig, out, "n0_share")


def decomposition(summ, names, n, arm, out):
    """Every target at n for the choice arm: mean T, the mean of its martingale part and the mean of the remainder,
    each +- 2 se, one row per category, one marker per kind of target."""
    cats = list(dict.fromkeys(name.split("_", 1)[1] for name in names))
    at = {name: next(s for s in summ[name, arm] if s["n"] == n) for name in names}
    fig, axes = FI.plt.subplots(1, 3, figsize=(6.4, 0.3 * len(cats) + 1.1), sharey=True, layout="constrained")
    ys = np.arange(len(cats))[::-1]
    for kk, kind in enumerate(KINDS):
        off = (kk - (len(KINDS) - 1) / 2) * 0.17
        st = dict(color=FI.N0_STYLE["share_%d" % (10 + 20 * kk)]["color"], fmt="os^D"[kk], ms=3, lw=0.8, capsize=0)
        S = [at["%s_%s" % (kind, c)] for c in cats]
        axes[0].errorbar([s["mean_T"] for s in S], ys + off, xerr=[2 * s["se_mean"] for s in S], label=kind, **st)
        axes[1].errorbar([s["mean_mart"] for s in S], ys + off, xerr=[2 * s["se_mart"] for s in S], **st)
        axes[2].errorbar([s["rem_total"] for s in S], ys + off, xerr=[2 * s["se_rem_total"] for s in S], **st)
    B = min(s["B"] for s in at.values())
    for ax in axes[:2]:
        ax.axvspan(-2 / np.sqrt(B), 2 / np.sqrt(B), color=FI.GRID, lw=0)
    for ax in axes:
        ax.axvline(0, color=FI.INK2, lw=0.6, ls="--")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(cats)
    axes[0].set_xlabel("mean $T$")
    axes[1].set_xlabel("mean martingale part (exact mean 0)")
    axes[2].set_xlabel("mean remainder (estimation)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(KINDS), fontsize=6)
    fig.suptitle("%s, n = %d, B = %d: T = martingale part + remainder; +- 2 se; band: 2 se of one target"
                 % (CI.LABEL[arm], n, B), fontsize=6)
    FI.save(fig, out, "decomposition")


def paper_figure(groups, arm, n, category, out):
    """The paper's figure: the four kinds of target of the category at n under
    the choice arm, one column each: the QQ plot of T with the 95 % simultaneous KS band (figures_inference.qq_panel),
    above its histogram with the N(0, 1) density."""
    sel = ["%s_%s" % (kind, category) for kind in KINDS]
    fig, axes = FI.plt.subplots(2, len(sel), figsize=(1.75 * len(sel) + 0.5, 3.4), squeeze=False, layout="constrained")
    x = np.linspace(-4, 4, 200)
    for k, name in enumerate(sel):
        T = FI.oracle_T(groups[name, arm, n])
        FI.qq_panel(axes[0, k], T, FI.BLUE)
        axes[0, k].set_title("%s\n%s" % (name, KINDS[name.split("_")[0]]), fontsize=6)
        axes[1, k].hist(T, bins=np.linspace(-4, 4, 33), density=True, color="#9ecae1", edgecolor="white", lw=0.3)
        axes[1, k].plot(x, FI.stats.norm.pdf(x), color="#b2182b", lw=0.9)
        axes[1, k].set_ylim(0, 0.48)
        axes[1, k].set_title("mean %+.2f, sd %.2f, coverage %.3f" % (T.mean(), T.std(ddof=1),
                                                                     np.mean(np.abs(T) <= FI.Z95)), fontsize=6)
    axes[0, 0].set_ylabel("QQ plot of $T$")
    axes[1, 0].set_ylabel("histogram of $T$")
    fig.suptitle("%s, n = %d, B = %d" % (CI.LABEL[arm], n, len(groups[sel[0], arm, n])), fontsize=6.5)
    FI.save(fig, out, "paper_qq_%s_n%d" % (category, n))


# ---------------------------------------------------------------- the reported n
DEVIATIONS = ("dev_mean", "dev_sd", "dev_cover")                  # the rule's three statistics
DEV_TITLE = {"dev_mean": r"$|$mean $T|$", "dev_sd": r"$|$sd $T - 1|$", "dev_cover": r"$|$coverage $- 0.95|$"}


def bounds(B):
    """The rule's bounds for the three averaged deviations: 2 Monte-Carlo se of one target of B streams."""
    return {"dev_mean": 2 / np.sqrt(B), "dev_sd": 2 / np.sqrt(2 * (B - 1)), "dev_cover": 2 * np.sqrt(0.95 * 0.05 / B)}


def kind_averages(summ, names, arm, n):
    """Per kind of target at n: the averages over its 8 categories of |mean T|, |sd T - 1| and |coverage - 0.95|, and
    the smallest B among the 8."""
    out = {}
    for kind in KINDS:
        S = [next(s for s in summ[name, arm] if s["n"] == n) for name in of_kind(names, kind)]
        out[kind] = dict(dev_mean=np.mean([abs(s["mean_T"]) for s in S]), dev_sd=np.mean([abs(s["sd_T"] - 1) for s in S]),
                         dev_cover=np.mean([abs(s["cover"] - 0.95) for s in S]), B=min(s["B"] for s in S))
    return out


def passes(avg):
    """The rule at one n: every kind's three averaged deviations within their bounds (a nan fails)."""
    for v in avg.values():
        b = bounds(v["B"])
        for stat in DEVIATIONS:
            if not v[stat] <= b[stat]:
                return False
    return True


def reported_n(avgs, grid):
    """The smallest n of the grid from which on every n passes; None if the largest n does not pass."""
    n_rep = None
    for n in reversed(grid):
        if not passes(avgs[n]):
            break
        n_rep = n
    return n_rep


def show_n(n, grid):
    if n is None:
        return "not met (%d)" % grid[-1]
    return "%d" % n


def reported_figure(avgs, rep, arm, grids, B, out):
    """Per kind of target, against n, the choice `arm` in both worlds (each on its own grid, grids[world]): the three
    averaged deviations, with the rule's bounds for B streams (dashed); the legend gives each world's reported n."""
    every = sorted({n for g in grids.values() for n in g})
    b = bounds(B)
    fig, axes = FI.plt.subplots(len(KINDS), 3, figsize=(6.4, 1.5 * len(KINDS) + 0.6), squeeze=False, layout="constrained")
    for i, kind in enumerate(KINDS):
        for j, stat in enumerate(DEVIATIONS):
            ax = axes[i, j]
            ax.axhline(b[stat], color="#b2182b", lw=0.8, ls="--")
            for world in avgs:
                grid = grids[world]
                label = "%s: reported n %s" % (WORLD_LABEL[world], show_n(rep[world], grid))
                ax.plot(grid, [avgs[world][n][kind][stat] for n in grid], "-", ms=2.5, lw=1.0, label=label,
                        **WORLD_STYLE[world])
            ax.set_ylim(bottom=0)
            FI.n_axis(ax, every)
        axes[i, 0].set_ylabel("%s (%s)" % (kind, KINDS[kind]))
    for ax, stat in zip(axes[0], DEVIATIONS):
        ax.set_title(DEV_TITLE[stat] + ", mean of 8 categories")
    for ax in axes[-1]:
        ax.set_xlabel("total rounds $n$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=1, fontsize=6)
    fig.suptitle("%s; dashed: the rule's bound, 2 Monte-Carlo se of one target (B = %d)" % (CI.LABEL[arm], B),
                 fontsize=6.5)
    FI.save(fig, out, "reported_n")


# ---------------------------------------------------------------- one world
def report(world, rows, times, names, grid, setup, phase, min_B, arm, category, out):
    """Everything of one world under out/: the rows, the summaries and the standard, the comparison of the choices of
    n0, the regret and the Cramer-Rao comparison, the figures.  Returns the summaries."""
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "rows.tsv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    choices = [c for c, _ in RMI.ARMS]
    B_planned = RMI.planned_B(phase, world)
    groups = {}
    for r in rows:
        groups.setdefault((r["target"], r["arm"], r["n"]), []).append(r)
    summ, table, verdicts, details = {}, [], {}, []
    for name in names:
        for c in choices:
            summ[name, c] = [dict(CI.summary(groups[name, c, n]), n=n, **martingale_part(groups[name, c, n]))
                             for n in grid]
            table += [dict(target=name, arm=c, **{k: v for k, v in s.items() if k != "_T"}) for s in summ[name, c]]
            C, verdicts[name, c] = CI.standard(summ[name, c], B_planned, min_B)
            details += ["### %s, %s: %s" % (name, CI.LABEL[c], verdicts[name, c]), ""] + CI.md_table("n", summ[name, c], C)
    CI.write_tsv(table, os.path.join(out, "summary.tsv"))
    ends = [(name, summ[name, arm][-1]) for name in names]           # the truth of every target, at the end
    CI.write_tsv([dict(target=name, mu_star=e["mu_star"], A_norm=e.get("A_norm", ""), PT_A=e.get("PT_A", ""),
                       PT_ratio=e.get("PT_ratio", ""), n=e["n"], B=e["B"], V_pqv=e["length"] / (2 * 1.959964),
                       mu_star_over_V=e["mu_star"] / (e["length"] / (2 * 1.959964)), excl0=e["excl0"],
                       excl0_pqv=e["excl0_pqv"], cover=e["cover"]) for name, e in ends], os.path.join(out, "targets.tsv"))
    md_n0 = []
    for kind in KINDS:                                               # the choices compared, 8 categories pooled
        mine = of_kind(names, kind)
        pooled = {c: [dict(CI.summary([r for name in mine for r in groups[name, c, n]]), n=n) for n in grid]
                  for c in choices}
        md_n0 += ["### %s (%s), the 8 categories pooled" % (kind, KINDS[kind]), ""] + CI.md_n0_comparison("n", pooled, choices)
    # ---- the report
    B = min(s["B"] for S_ in summ.values() for s in S_)
    count = {c: {v: sum(verdicts[name, c] == v for name in names) for v in ("PASS", "WATCH", "FAIL")} for c in choices}
    tc, ti = np.array([t["t_collect"] for t in times]), np.array([t["t_infer"] for t in times])
    taus = " / ".join(str(t) for t in sorted({t["tau"] for t in times}))
    md = ["# MIND inference, phase %s, %s: verification" % (phase, WORLD_LABEL[world]), "",
          "World: %s (truth's intercept %.3f); rank-2 truth, r0 = 2 given, rounds with replacement, tau = %s, "
          "c_tau = %g, the penalty constant by CV; %d of %d streams; n = %s.  Time per stream: collection %.0f s "
          "(median; max %.0f), Steps 2-4 %.0f s (median; max %.0f)." % (
              WORLD_TEXT[world], setup["truth_gamma"][0], taus, RMI.C_TAU, B, B_planned,
              ", ".join("%d" % n for n in grid), np.median(tc), tc.max(), np.median(ti), ti.max()), "",
          "T = (mu_hat - mu*) / V_pqv (oracle); cover = the oracle interval mu_hat +- 1.96 V_pqv; the remainder is over "
          "V_pqv; V_hat enters only through V_hat / V_pqv (V3c); the standard is in the header of "
          "inference/collect_inference.py.  Constants: C_ref 1/%g, C_cal %g; c0 1/%g in both splits (the information "
          "rule's kappa0 from the exploration sample, the design-stage split's kappa_D %.4f, the world's)." % (
              1 / RMI.IA.C_REF, RMI.IA.C_CAL, 1 / RMI.IA.C0, float(setup["kappa_D"])), "",
          "## The choices of n0 compared on the same streams", "",
          "Each kind of target pooled over its 8 categories: the RMS is over categories and streams, the coverage the "
          "mean over categories.", "", *md_n0,
          "## Verdicts: every target under every choice of n0%s" % (
              "" if B >= min_B else " (DEBUG: B below %d, no statistical claim)" % min_B), "",
          "| target | %s |" % " | ".join(CI.SHORT[c] for c in choices), "|---|" + "---|" * len(choices),
          *["| %s | %s |" % (name, " | ".join(verdicts[name, c] for c in choices)) for name in names],
          "| PASS / WATCH / FAIL | %s |" % " | ".join("%d / %d / %d" % (count[c]["PASS"], count[c]["WATCH"], count[c]["FAIL"])
                                                    for c in choices), "",
          "Columns: %s.  The tables and checks of every cell are in details_mind_inference.md." % "; ".join(
              "%s = %s" % (CI.SHORT[c], CI.LABEL[c]) for c in choices), ""]
    first = names[0]                                                 # regret and the Cramer-Rao comparison
    reg = []
    for n in grid:
        G = groups[first, arm, n]                                  # the collection's: the same under every choice
        v = np.array([r["regret"] for r in G])
        reg.append(dict(n=n, B=len(G), rounds=G[0]["regret_rounds"], regret=v.mean(),
                        se=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                        regret_ex=np.mean([r["regret_ex"] for r in G]), regret_t0=np.mean([r["regret_t0"] for r in G]),
                        regret_t1=np.mean([r["regret_t1"] for r in G])))
    CI.write_tsv(reg, os.path.join(out, "regret.tsv"))
    vcr = []
    for kind in KINDS:
        mine = of_kind(names, kind)
        for n in grid:
            G = [r for name in mine for r in groups[name, arm, n]]
            vcr.append(dict(kind=kind, n=n, ratio=np.mean([r["VCR_ratio"] for r in G]),
                            pred=np.mean([r["VCR_ratio_pred"] for r in G]), kappa_A=np.mean([r["kappa_A"] for r in G]),
                            kap_hat=np.mean([r["kap_hat"] for r in G]), V_CR=np.mean([r["V_CR"] for r in G])))
    CI.write_tsv(vcr, os.path.join(out, "vcr.tsv"))
    md += ["## The regret of the two-track collection", "",
           "Cumulative expected regret (on this world's click probabilities) of the rounds the inference at n uses; "
           "descriptive.  The regret study runs this collection's rule as its arm two_track on its own streams, "
           "with rounds drawn the same way (with replacement): the down-sampled world to 40,000 (tau = 55), the main "
           "world to 80,000 (tau = 57).", "",
           "| n | B | regret (se) | exploration | track 0 | track 1 |", "|---|---|---|---|---|---|",
           *["| %d | %d | %.1f (%.1f) | %.1f | %.1f | %.1f |" % (r["n"], r["B"], r["regret"], r["se"], r["regret_ex"],
                                                             r["regret_t0"], r["regret_t1"]) for r in reg], "",
           "## The Cramer-Rao comparison, %s, 8 categories averaged" % CI.LABEL[arm], "",
           "| kind | n | V_pqv / V_CR | predicted | kappa_A | kappa_hat |", "|---|---|---|---|---|---|",
           *["| %s | %d | %.3f | %.3f | %.4f | %.4f |" % (r["kind"], r["n"], r["ratio"], r["pred"], r["kappa_A"],
                                                       r["kap_hat"]) for r in vcr], ""]
    open(os.path.join(out, "verdict_mind_inference.md"), "w").write("\n".join(md))
    open(os.path.join(out, "details_mind_inference.md"), "w").write("\n".join(
        ["# MIND inference, phase %s, %s: every target under every choice of n0" % (phase, WORLD_LABEL[world]), ""]
        + details))
    print("\n".join(md[:3]))
    print("report written:", os.path.join(out, "verdict_mind_inference.md"))
    # ---- figures
    n0_choice(groups, names, choices, grid, out)
    n0_share(summ, names, grid, arm, out)
    sel = ["%s_%s" % (kind, category) for kind in KINDS]
    FI.BLOCKS = sel
    FI.BLABEL = {name: "%s\n%s" % (category, KINDS[name.split("_")[0]]) for name in sel}
    FI.QQ_AT = {name: tuple(grid) for name in sel}
    rows_arm = {name: [r for n in grid for r in groups[name, arm, n]] for name in sel}
    FI.trend(rows_arm, os.path.join(out, arm))
    FI.qq(rows_arm, os.path.join(out, arm))
    paper_figure(groups, arm, RMI.figure_n(phase, world), category, os.path.join(out, arm))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(RMI.PHASES))
    ap.add_argument("--setup", required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--min-B", type=int, default=50)
    ap.add_argument("--incomplete", action="store_true", help="collect the ok streams even if some are not")
    ap.add_argument("--arm", default=RMI.IS.PRIMARY, choices=[c for c, _ in RMI.ARMS])  # the displayed choice
    ap.add_argument("--category", default="news")         # the most displayed pairs
    a = ap.parse_args()
    out = a.dir or "mind_inference_%s" % a.phase
    col = os.path.join(out, "collected")
    os.makedirs(col, exist_ok=True)
    setup_sha = RMI.sha16(a.setup)
    status = status_of(a.phase, out, setup_sha)
    CI.write_tsv(status, os.path.join(col, "status.tsv"))
    bad = [s for s in status if s["status"] != "ok"]
    print("%s: %d streams planned, %d ok%s" % (a.phase, len(status), len(status) - len(bad),
          "; not ok: " + ", ".join("%s (%s)" % (s["key"], s["status"]) for s in bad[:5]) if bad else ""))
    if (bad and not a.incomplete) or len(bad) == len(status):
        sys.exit("incomplete: nothing else written (--incomplete collects the ok streams)")
    setup = dict(np.load(a.setup))
    names = [str(s) for s in setup["target_names"]]
    choices = [c for c, _ in RMI.ARMS]
    avgs, rep, B_world, grids = {}, {}, {}, {}
    for world in RMI.WORLDS:
        grid = grids[world] = list(RMI.grid_of(a.phase, world))
        FI.TICKS = tuple(grid)
        rows, times, streams = rows_of(a.phase, out, setup_sha, world)
        if not rows:
            print("%s: no ok stream" % world)
            continue
        os.makedirs(os.path.join(col, world), exist_ok=True)
        CI.write_tsv(streams, os.path.join(col, world, "streams.tsv"))
        summ = report(world, rows, times, names, grid, RMI.world_setup(setup, world), a.phase, a.min_B, a.arm,
                      a.category, os.path.join(col, world))
        del rows                                                     # one world's rows in memory at a time
        avgs[world] = {c: {n: kind_averages(summ, names, c, n) for n in grid} for c in choices}
        rep[world] = {c: reported_n(avgs[world][c], grid) for c in choices}
        B_world[world] = min(v["B"] for v in avgs[world][a.arm][grid[-1]].values())
        decomposition(summ, names, RMI.figure_n(a.phase, world), a.arm, os.path.join(col, world))
    # ---- the reported n, both worlds
    B = min(B_world.values())
    table = ["| world | %s |" % " | ".join(CI.SHORT[c] for c in choices), "|---|" + "---|" * len(choices)]
    for world in rep:
        table.append("| %s | %s |" % (WORLD_LABEL[world], " | ".join(show_n(rep[world][c], grids[world])
                                                                    for c in choices)))
    md = ["# MIND inference, phase %s: the reported n%s" % (
              a.phase, "" if B >= a.min_B else " (DEBUG: B below %d, no claim)" % a.min_B), "",
          "The rule (fixed before the run): the smallest n of the grid from which on (at n and every larger "
          "n) each kind of target has, averaged over its 8 categories, |mean T|, |sd T - 1| and |coverage - 0.95| "
          "within 2 Monte-Carlo standard errors of one target (2 / sqrt(B), 2 / sqrt(2 (B - 1)), "
          "2 sqrt(0.95 x 0.05 / B)).  \"not met\": the largest n does not pass; the largest n is then reported as "
          "such.", "",
          "The main text reports the down-sampled world, the appendix the main world (fixed before the run); "
          "their figures are drawn at n = %s, whatever the reported "
          "n." % " and ".join("%d (%s)" % (RMI.figure_n(a.phase, w), w) for w in rep), "", *table, "",
          "Columns: %s." % "; ".join("%s = %s" % (CI.SHORT[c], CI.LABEL[c]) for c in choices), ""]
    for world in rep:
        b = bounds(B_world[world])
        md += ["## %s, %s: the averaged deviations (B = %d; bounds %.3f, %.3f, %.3f)" % (
                   WORLD_LABEL[world], CI.LABEL[a.arm], B_world[world], b["dev_mean"], b["dev_sd"], b["dev_cover"]), "",
               "| n | %s | passes |" % " | ".join("%s: abs mean T, abs(sd T - 1), abs(cover - 0.95)" % kind for kind in KINDS),
               "|---|" + "---|" * (len(KINDS) + 1)]
        for n in grids[world]:
            v = avgs[world][a.arm][n]
            md.append("| %d | %s | %s |" % (n, " | ".join("%.3f, %.3f, %.3f" % (v[k]["dev_mean"], v[k]["dev_sd"],
                                                                              v[k]["dev_cover"]) for k in KINDS),
                                            "yes" if passes(v) else "no"))
        md.append("")
    open(os.path.join(col, "reported_n.md"), "w").write("\n".join(md))
    print("\n".join(table))
    print("written:", os.path.join(col, "reported_n.md"))
    reported_figure({world: avgs[world][a.arm] for world in avgs}, {world: rep[world][a.arm] for world in rep},
                    a.arm, grids, B, col)


if __name__ == "__main__":
    main()
