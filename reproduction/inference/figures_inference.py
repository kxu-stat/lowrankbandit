#!/usr/bin/env python3
"""Figures of the inference experiment, from the collected tables.

    python3 figures_inference.py [--collected inference_results/collected] [--out figures] [--arm ARM] [--target T]

Every figure is drawn for one target (--target: dense, the default, entry_max, entry_min or rank1; another target's
figures go to OUT/<target>/); the block I1d32 (identity link, d = 32) is a row of its own; with --arm the figure
`targets` puts the four targets side by side.

Always drawn
  n0_choice           the choices of n0 on the same replicates (one row per block): the RMS of the total remainder
                      / V_pqv against n, and the oracle coverage with the 0.95 +- 2 se band, one line per choice
The figures of one choice of n0, --arm ARM (design_75, the default and the displayed choice; info_rule,
share_10, ..., design_90):
  inference_trend     one row per block (I1, I2l): |remainder| / V_pqv by part (log scale, lines at 0.03 and 0.05),
                      mean T, KS distance of T to N(0, 1) with its null 95 % value, sd T, and the oracle coverage with
                      the 0.95 +- 2 se band -- the checks V1-V3 of the verification standard
  inference_qq        per block a row of QQ plots (95 % simultaneous KS band) and a row of histograms (N(0, 1)
                      density) at four n (QQ_AT)
  variance            V_hat / V_pqv (the variance estimate against the oracle) and mean V_pqv / sd_MC(mu_hat)
  remainder           the remainder parts / V_pqv, signed, +- 2 se, against n
  comparator          ours (T with V_pqv) against the one-step estimator with the inverse empirical Hessian,
                      standardized by its Monte-Carlo sd: bias, KS distance, oracle coverage, QQ plot at the largest n
  refit               the choice against n0 = n_tr (the nuisance fitted on the whole track, no refit) on the same
                      replicates: total remainder, sd T, oracle coverage, QQ plot at the largest n
  tie                 the adaptive-tie design I3 (the displayed choice, there the equal split): the same comparison
                      against N
  rank                the appendix "unknown r0": the truncation rank chosen in each fold by GIC (price log(s)/s) or an
                      AIC-type criterion (price 1/s), against the known rank (both with the displayed choice's n0)
  targets             the four targets under the choice ARM: |total remainder| / V_pqv, sd T, oracle coverage,
                      and V_pqv / V_CR against the prediction sqrt(C_cal kappa_A log n_tr / kappa_hat) (dashed)
Every CLT quantity is under the oracle standard error: T = (mu_hat - mu*) / V_pqv and the interval
mu_hat +- 1.96 V_pqv; the procedure's V_hat appears only in the variance figure.  Rows are labeled by the model,
not by the block codes I1 / I2l, and B is left to the captions.
"""
import argparse, csv, glob, os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulation as S

BLOCKS = ["I1", "I2l", "I1d32"]
BLABEL = {"I1": "identity link", "I2l": r"logistic, $n_b = 1$", "I1d32": r"identity link, $d = 32$"}   # no block codes
QQ_AT = {"I1": (500, 1000, 2000, 5000), "I2l": (1000, 5000, 20000, 40000), "I1d32": (1000, 2000, 4000, 8000)}
#   the logistic block's last column at 40,000, the n of the MIND figure in the main text
TARGET_LABEL = {"dense": r"dense $UJV^\top/d$", "entry_max": "entry, largest tangent part",
                "entry_min": "entry, smallest tangent part", "rank1": r"rank one $m_0x_0^\top$"}
TARGET_COLOR = {"dense": "#1a1a19", "entry_max": "#2a78d6", "entry_min": "#b2182b", "rank1": "#1baf7a"}
TICKS = (500, 1000, 2000, 5000, 10000, 20000, 60000)
PARTS = (("taylor", "#e08a3c"), ("normal", "#2a78d6"), ("calibration", "#8a8983"), ("total", "#b2182b"))
OURS, OLS, BLUE = "#1a1a19", "#8a8983", "#2a78d6"
INK2, GRID = "#52514e", "#e6e5e1"
Z95 = 1.959964

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.0, "lines.markersize": 2.5, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def load(collected, regime="lowrank", arm=S.PRIMARY, target="dense"):
    """{block: rows of the arm} and {block: rows of every arm} of one covariate regime and target (the tie design I3
    goes with `lowrank`; its arm is the same choice when it runs it, otherwise the displayed one, and it has one
    target), fields numeric.  target None: every target (the figure `targets`)."""
    chosen, every = {}, {}
    paths = glob.glob(os.path.join(collected, "I*_%s.tsv" % regime))
    if regime == "lowrank":
        paths += glob.glob(os.path.join(collected, "I3_tie.tsv"))
    for path in sorted(paths):
        block = os.path.basename(path).split("_")[0]
        rows = list(csv.DictReader(open(path), delimiter="\t"))
        if block != "I3" and target is not None:
            rows = [r for r in rows if r["target"] == target]
        for r in rows:
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
        every[block] = rows
        own = arm if block != "I3" or arm in dict(S.arms("I3")) else S.PRIMARY
        chosen[block] = [r for r in rows if r["arm"] == own]
    return chosen, every


def by_n(rows, key="n"):
    out = {}
    for r in rows:
        out.setdefault(int(r[key]), []).append(r)
    return dict(sorted(out.items()))


def col(G, k):
    return np.array([r[k] for r in G], float)


def oracle_T(G):
    return (col(G, "mu") - col(G, "mu_star")) / col(G, "V_pqv")


def parts(G):
    Vp = col(G, "V_pqv")
    p = {"taylor": col(G, "r_tay") / Vp, "normal": (col(G, "r_norA") + col(G, "r_norz")) / Vp,
         "calibration": col(G, "r_cal") / Vp}
    p["total"] = p["taylor"] + p["normal"] + p["calibration"]
    return p


def n_axis(ax, ns):
    ax.set_xscale("log")
    ax.minorticks_off()
    t = [x for x in TICKS if min(ns) <= x <= max(ns)]
    ax.set_xticks(t)
    ax.set_xticklabels(["%d" % x if x < 1000 else "%gk" % (x / 1000) for x in t])


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


# ---------------------------------------------------------------- main text 1: the trend
def trend(chosen, out):
    blocks = [b for b in BLOCKS if b in chosen]
    fig, axes = plt.subplots(len(blocks), 5, figsize=(9.6, 1.85 * len(blocks) + 0.4), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        g = by_n(chosen[block])
        ns = np.array(list(g))
        T = [oracle_T(G) for G in g.values()]
        B = np.array([len(t) for t in T])
        ax = axes[i, 0]
        for name, color in PARTS:
            P = [parts(G)[name] for G in g.values()]
            m = np.abs([p.mean() for p in P]); e = np.array([p.std(ddof=1) / np.sqrt(len(p)) for p in P])
            ax.plot(ns, np.maximum(m, 1e-3), "-o", color=color, lw=1.3 if name == "total" else 0.8, label=name)
            if name == "total":
                ax.fill_between(ns, np.maximum(m - 2 * e, 1e-3), m + 2 * e, color=color, alpha=0.12, lw=0)
        ax.axhline(0.03, color="#1a9850", lw=0.7, ls="--"); ax.axhline(0.05, color="#d73027", lw=0.7, ls=":")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1)
        m = np.array([t.mean() for t in T]); e = np.array([t.std(ddof=1) for t in T]) / np.sqrt(B)
        axes[i, 1].errorbar(ns, m, yerr=2 * e, fmt="-o", color=OURS, elinewidth=0.6, capsize=1.2)
        axes[i, 1].axhline(0, color=INK2, lw=0.6)
        D = [stats.kstest(t, "norm").statistic for t in T]
        axes[i, 2].plot(ns, D, "-o", color=OURS, label="KS distance")
        axes[i, 2].plot(ns, [stats.kstwo.ppf(0.95, b) for b in B], "--", color="#b2182b", lw=0.8, label="null 95% value")
        axes[i, 2].set_ylim(bottom=0)
        sd = np.array([t.std(ddof=1) for t in T])
        axes[i, 3].errorbar(ns, sd, yerr=2 * sd / np.sqrt(2 * (B - 1)), fmt="-o", color=OURS, elinewidth=0.6, capsize=1.2)
        axes[i, 3].axhline(1, color=INK2, lw=0.6)
        se_c = np.sqrt(0.95 * 0.05 / B)
        axes[i, 4].fill_between(ns, 0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
        axes[i, 4].axhline(0.95, color=INK2, lw=0.6)
        axes[i, 4].plot(ns, [np.mean(np.abs(t) <= Z95) for t in T], "-o", color=OURS)
        for ax in axes[i]:
            n_axis(ax, ns)
        axes[i, 0].set_ylabel(BLABEL[block])
        if i == 0:
            for ax, title in zip(axes[0], (r"|remainder| / $V_{pqv}$ (lines 0.03, 0.05)", r"mean $T$ ($\pm$2 se)",
                                           r"KS distance of $T$ to $N(0,1)$", r"sd $T$ ($\pm$2 se)", r"oracle coverage (0.95 $\pm$ 2 se)")):
                ax.set_title(title)
            axes[0, 0].legend(fontsize=5.5, loc="lower left"); axes[0, 2].legend(fontsize=5.5)
        if i == len(blocks) - 1:
            for ax in axes[i]:
                ax.set_xlabel("total rounds $n$")
    save(fig, out, "inference_trend")


# ---------------------------------------------------------------- main text 2: QQ plots and histograms
def pick(ns, wanted):
    """The wanted grid points that exist (the debug plan has fewer): else first, middle, last."""
    got = [n for n in wanted if n in ns]
    return got if got else sorted({ns[0], ns[len(ns) // 2], ns[-1]})


def qq_panel(ax, T, color, band=True):
    T = np.sort(T); B = len(T); j = np.arange(1, B + 1); q = stats.norm.ppf((j - 0.5) / B)
    if band:
        dks = stats.kstwo.ppf(0.95, B)                                   # simultaneous 95% KS band
        lo = stats.norm.ppf(np.clip(j / B - dks, 1e-6, 1)); hi = stats.norm.ppf(np.clip((j - 1) / B + dks, 0, 1 - 1e-6))
        ax.fill_between(q, lo, hi, color=GRID, lw=0)
    ax.plot([-3.5, 3.5], [-3.5, 3.5], color=INK2, lw=0.6, ls="--")
    ax.plot(q, T, "o", ms=1.4, color=color, markeredgewidth=0)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-4, 4)


def qq(chosen, out):
    blocks = [b for b in BLOCKS if b in chosen]
    ncol = max(len(pick(list(by_n(chosen[b])), QQ_AT[b])) for b in blocks)
    fig, axes = plt.subplots(2 * len(blocks), ncol, figsize=(1.75 * ncol + 0.5, 3.3 * len(blocks)), squeeze=False, layout="constrained")
    grid = np.linspace(-4, 4, 200)
    for i, block in enumerate(blocks):
        g = by_n(chosen[block])
        sel = pick(list(g), QQ_AT[block])
        for k in range(ncol):
            a_q, a_h = axes[2 * i, k], axes[2 * i + 1, k]
            if k >= len(sel):
                a_q.set_visible(False); a_h.set_visible(False)
                continue
            T = oracle_T(g[sel[k]])
            qq_panel(a_q, T, BLUE)
            a_q.set_title("n = %d" % sel[k], fontsize=6)
            a_h.hist(T, bins=np.linspace(-4, 4, 33), density=True, color="#9ecae1", edgecolor="white", lw=0.3)
            a_h.plot(grid, stats.norm.pdf(grid), color="#b2182b", lw=0.9)
            a_h.set_ylim(0, 0.48)                                           # one scale for every histogram
            a_h.set_title("mean %+.2f, sd %.2f, coverage %.3f" % (T.mean(), T.std(ddof=1), np.mean(np.abs(T) <= Z95)), fontsize=6)
        axes[2 * i, 0].set_ylabel(BLABEL[block] + "\nQQ plot of $T$")
        axes[2 * i + 1, 0].set_ylabel(BLABEL[block] + "\nhistogram of $T$")
    save(fig, out, "inference_qq")


# ---------------------------------------------------------------- appendix: variance, remainder, comparator, tie
def variance(chosen, out):
    blocks = [b for b in BLOCKS if b in chosen]
    fig, axes = plt.subplots(2, len(blocks), figsize=(2.3 * len(blocks) + 0.4, 3.6), squeeze=False, layout="constrained")
    for j, block in enumerate(blocks):
        g = by_n(chosen[block]); ns = np.array(list(g))
        r = [col(G, "V") / col(G, "V_pqv") for G in g.values()]
        axes[0, j].errorbar(ns, [x.mean() for x in r], yerr=[2 * x.std(ddof=1) / np.sqrt(len(x)) for x in r],
                            fmt="-o", color=BLUE, elinewidth=0.6, capsize=1.2)
        axes[0, j].axhline(1, color=INK2, lw=0.6)
        err = [col(G, "mu") - col(G, "mu_star") for G in g.values()]
        q = np.array([col(G, "V_pqv").mean() / e.std(ddof=1) for G, e in zip(g.values(), err)])
        B = np.array([len(e) for e in err])
        axes[1, j].errorbar(ns, q, yerr=2 * q / np.sqrt(2 * (B - 1)), fmt="-o", color=OURS, elinewidth=0.6, capsize=1.2)
        axes[1, j].axhline(1, color=INK2, lw=0.6)
        axes[0, j].set_title(BLABEL[block])
        for ax in axes[:, j]:
            n_axis(ax, ns)
        axes[1, j].set_xlabel("total rounds $n$")
    axes[0, 0].set_ylabel(r"mean $\hat V\ /\ V_{pqv}$ ($\pm$2 se)")
    axes[1, 0].set_ylabel(r"mean $V_{pqv}\ /\ \mathrm{sd}_{MC}(\hat\mu)$ ($\pm$2 se)")
    save(fig, out, "variance")


def remainder(chosen, out):
    blocks = [b for b in BLOCKS if b in chosen]
    fig, axes = plt.subplots(1, len(blocks), figsize=(2.3 * len(blocks) + 0.4, 2.1), squeeze=False, layout="constrained")
    for j, block in enumerate(blocks):
        ax = axes[0, j]
        g = by_n(chosen[block]); ns = np.array(list(g))
        for name, color in PARTS:
            P = [parts(G)[name] for G in g.values()]
            m = np.array([p.mean() for p in P]); e = np.array([p.std(ddof=1) / np.sqrt(len(p)) for p in P])
            ax.plot(ns, m, "-o", color=color, lw=1.3 if name == "total" else 0.8, label=name)
            ax.fill_between(ns, m - 2 * e, m + 2 * e, color=color, alpha=0.12, lw=0)
        ax.axhspan(-0.03, 0.03, color="#e8f3e8", lw=0, zorder=0)
        ax.axhline(0, color=INK2, lw=0.6)
        n_axis(ax, ns)
        ax.set_title(BLABEL[block]); ax.set_xlabel("total rounds $n$")
    axes[0, 0].set_ylabel(r"remainder / $V_{pqv}$ (mean $\pm$ 2 se)")
    axes[0, 0].legend(fontsize=5.5)
    save(fig, out, "remainder")


def one_step_Z(G):
    """The one-step estimator standardized by its Monte-Carlo sd at this grid point: it has no
    oracle standard error (its weights, the inverse empirical Hessian, use all the data)."""
    e = col(G, "mu_ols") - col(G, "mu_star")
    e = e[np.isfinite(e)]
    return e / e.std(ddof=1)


COMPARE_TITLES = (r"bias / $\mathrm{sd}_{MC}(\hat\mu)$ ($\pm$2 se)", r"KS distance to $N(0,1)$",
                  r"oracle coverage (0.95 $\pm$ 2 se)")


def compare_row(ax, g):
    """Ours (T with V_pqv) against the one-step estimator (with its sd_MC) along one grid g: bias in
    Monte-Carlo sd units, KS distance to N(0, 1) (null 95 % value dashed), oracle coverage, and the QQ
    plot at the largest grid point.  The one-step estimator is undefined where a track has fewer rounds than
    coordinates (recorded as nan, e.g. I1 at n = 500 and I1d32 below n = 3000); those grid points are left out
    of its first three panels.  Returns the grid."""
    ns = np.array(list(g))
    B = np.array([len(G) for G in g.values()])
    for which, color, lab in (("ours", OURS, r"ours, $T$ with $V_{pqv}$"),
                              ("one-step", OLS, r"one-step, empirical Hessian, with $\mathrm{sd}_{MC}$")):
        mu = "mu" if which == "ours" else "mu_ols"
        err = [col(G, mu) - col(G, "mu_star") for G in g.values()]
        err = [e[np.isfinite(e)] for e in err]
        keep = np.array([which == "ours" or len(e) > 2 for e in err])            # the one-step: where it is defined
        Z = [oracle_T(G) if which == "ours" else (one_step_Z(G) if k else np.array([0.0]))
             for G, k in zip(g.values(), keep)]
        ax[0].errorbar(ns[keep], [e.mean() / e.std(ddof=1) for e, k in zip(err, keep) if k],
                       yerr=[2 / np.sqrt(len(e)) for e, k in zip(err, keep) if k],
                       fmt="-o", color=color, elinewidth=0.6, capsize=1.2, label=lab)
        ax[1].plot(ns[keep], [stats.kstest(z, "norm").statistic for z, k in zip(Z, keep) if k], "-o", color=color)
        ax[2].plot(ns[keep], [np.mean(np.abs(z) <= Z95) for z, k in zip(Z, keep) if k], "-o", color=color)
        if keep[-1]:
            qq_panel(ax[3], Z[-1], color, band=(which == "ours"))
    ax[0].axhline(0, color=INK2, lw=0.6)
    ax[1].plot(ns, [stats.kstwo.ppf(0.95, b) for b in B], "--", color="#b2182b", lw=0.8)
    ax[1].set_ylim(bottom=0)
    se_c = np.sqrt(0.95 * 0.05 / B[-1])
    ax[2].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
    ax[2].axhline(0.95, color=INK2, lw=0.6)
    return ns


def comparator(chosen, out):
    blocks = [b for b in BLOCKS if b in chosen]
    fig, axes = plt.subplots(len(blocks), 4, figsize=(8.0, 1.85 * len(blocks) + 0.4), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        ns = compare_row(axes[i], by_n(chosen[block]))
        for ax in axes[i, :3]:
            n_axis(ax, ns)
        axes[i, 3].set_title("QQ at n = %d" % ns[-1], fontsize=6)
        axes[i, 0].set_ylabel(BLABEL[block])
    for ax, title in zip(axes[0, :3], COMPARE_TITLES):
        ax.set_title(title)
    axes[0, 0].legend(fontsize=5.5)
    for ax in axes[-1, :3]:
        ax.set_xlabel("total rounds $n$")
    save(fig, out, "comparator")


def tie(every, out):
    if "I3" not in every:
        return
    g = by_n([r for r in every["I3"] if r["arm"] == S.PRIMARY], key="N")
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.1), layout="constrained")
    Ns = compare_row(axes, g)
    for ax, title in zip(axes[:3], COMPARE_TITLES):
        ax.set_xscale("log"); ax.set_xlabel("post-exploration rounds $N$"); ax.set_title(title)
    axes[3].set_title("QQ at N = %d" % Ns[-1], fontsize=6)
    axes[0].legend(fontsize=5.5, loc="lower left")
    save(fig, out, "tie")


def refit(every, out, arm):
    """Step 2 with and without the refit, on the same replicates: the choice of n0 `arm` (the predictable refit on
    the rest of the track) against n0 = n_tr (nuisance on the whole track, no refit).  Total remainder / V_pqv
    (mean +- 2 se), sd T (+- 2 se), oracle coverage, and the QQ plot at the largest n."""
    blocks = [b for b in BLOCKS if b in every]
    fig, axes = plt.subplots(len(blocks), 4, figsize=(8.0, 1.85 * len(blocks) + 0.4), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        for which, color, lab in ((arm, OURS, "%s, refit" % N0_LABEL[arm]),
                                  ("no_refit", BLUE, r"$n_0 = n_{\mathrm{tr}}$, no refit")):
            g = by_n([r for r in every[block] if r["arm"] == which])
            ns = np.array(list(g))
            tot = [parts(G)["total"] for G in g.values()]
            m = np.array([t.mean() for t in tot]); e = np.array([t.std(ddof=1) / np.sqrt(len(t)) for t in tot])
            axes[i, 0].plot(ns, m, "-o", color=color, label=lab)
            axes[i, 0].fill_between(ns, m - 2 * e, m + 2 * e, color=color, alpha=0.15, lw=0)
            T = [oracle_T(G) for G in g.values()]
            sd = np.array([t.std(ddof=1) for t in T]); B = np.array([len(t) for t in T])
            axes[i, 1].errorbar(ns, sd, yerr=2 * sd / np.sqrt(2 * (B - 1)), fmt="-o", color=color, elinewidth=0.6, capsize=1.2)
            axes[i, 2].plot(ns, [np.mean(np.abs(t) <= Z95) for t in T], "-o", color=color)
            qq_panel(axes[i, 3], T[-1], color, band=(which == arm))
        axes[i, 0].axhspan(-0.03, 0.03, color="#e8f3e8", lw=0, zorder=0)
        axes[i, 0].axhline(0, color=INK2, lw=0.6)
        axes[i, 1].axhline(1, color=INK2, lw=0.6)
        se_c = np.sqrt(0.95 * 0.05 / B[-1])
        axes[i, 2].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
        axes[i, 2].axhline(0.95, color=INK2, lw=0.6)
        for ax in axes[i, :3]:
            n_axis(ax, ns)
        axes[i, 3].set_title("QQ at n = %d" % ns[-1], fontsize=6)
        axes[i, 0].set_ylabel(BLABEL[block])
    for ax, title in zip(axes[0, :3], (r"total remainder / $V_{pqv}$ (mean $\pm$ 2 se)", r"sd $T$ ($\pm$2 se)",
                                       r"oracle coverage (0.95 $\pm$ 2 se)")):
        ax.set_title(title)
    axes[0, 0].legend(fontsize=5.5, loc="lower right")
    for ax in axes[-1, :3]:
        ax.set_xlabel("total rounds $n$")
    save(fig, out, "refit")


def rank(every, out):
    """The appendix "unknown r0": the known rank against the rank chosen by GIC and by the AIC-type criterion in each
    fold, on the same replicates, all with the displayed choice's n0."""
    blocks = [b for b in BLOCKS if b in every and any(r["arm"] == "rank_gic" for r in every[b])]
    if not blocks:
        return
    arms = ((S.PRIMARY, OURS, r"known $r_0$", "-o"), ("rank_gic", "#b2182b", r"GIC, $\log(s)/s$", "--s"),
            ("rank_aic", BLUE, r"AIC-type, $1/s$", ":^"))       # distinct styles: the arms coincide where both folds pick r0
    fig, axes = plt.subplots(len(blocks), 4, figsize=(8.0, 1.85 * len(blocks) + 0.4), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        g0 = by_n([r for r in every[block] if r["arm"] == S.PRIMARY])
        len0 = {n: np.mean(col(G, "V_pqv")) for n, G in g0.items()}
        for arm, color, lab, fmt in arms:
            g = by_n([r for r in every[block] if r["arm"] == arm])
            ns = np.array(list(g))
            if arm != S.PRIMARY:
                ok = [np.mean((col(G, "rank_0") == col(G, "r0")) & (col(G, "rank_1") == col(G, "r0"))) for G in g.values()]
                axes[i, 0].plot(ns, ok, fmt, color=color, label=lab, mfc="none")
                axes[i, 3].plot(ns, [np.mean(col(G, "V_pqv")) / len0[n] for n, G in g.items()], fmt, color=color, mfc="none")
            T = [oracle_T(G) for G in g.values()]
            B = np.array([len(t) for t in T])
            axes[i, 1].plot(ns, [np.mean(np.abs(t) <= Z95) for t in T], fmt, color=color, label=lab, mfc="none")
            sd = np.array([t.std(ddof=1) for t in T])
            axes[i, 2].errorbar(ns, sd, yerr=2 * sd / np.sqrt(2 * (B - 1)), fmt=fmt, color=color, elinewidth=0.6,
                                capsize=1.2, mfc="none")
        se_c = np.sqrt(0.95 * 0.05 / B[-1])
        axes[i, 1].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
        axes[i, 1].axhline(0.95, color=INK2, lw=0.6)
        axes[i, 2].axhline(1, color=INK2, lw=0.6)
        axes[i, 3].axhline(1, color=INK2, lw=0.6)
        axes[i, 0].set_ylim(-0.03, 1.03)
        for ax in axes[i]:
            n_axis(ax, ns)
        axes[i, 0].set_ylabel(BLABEL[block])
    for ax, title in zip(axes[0], (r"share with both folds at $r_0$", r"oracle coverage (0.95 $\pm$ 2 se)",
                                   r"sd $T$ ($\pm$2 se)", r"interval length / known-$r_0$ length")):
        ax.set_title(title)
    axes[0, 1].legend(fontsize=5.5, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("total rounds $n$")
    save(fig, out, "rank")


N0_LABEL = {"info_rule": "information rule", "no_refit": r"$n_0 = n_{\mathrm{tr}}$ (no refit)",
            **{a: r"$n_0 = %s\,n_{\mathrm{tr}}$" % how for a, how in S.N0_CHOICES if a.startswith("share")},
            **{a: "design-stage split, cap %s" % how.split()[1] for a, how in S.N0_CHOICES if a.startswith("design")}}
N0_STYLE = {"info_rule": dict(color="#8a8983", ls="--", marker=None),
            "share_10": dict(color="#1baf7a"), "share_30": dict(color="#2a78d6"), "share_50": dict(color="#eb6834"),
            "share_70": dict(color="#b3366b"), "share_90": dict(color="#7c5cc4"),
            "no_refit": dict(color="#8a8983", ls=":", marker="x"),
            "design_75": dict(color="#1a1a19", ls=":", marker="s", mfc="none"),
            "design_90": dict(color="#1a1a19", ls="-", marker="o", mfc="none")}


def n0_choice(every, out):
    """The choices of n0 on the same replicates: RMS of the total remainder / V_pqv (log scale) and the oracle
    coverage (0.95 +- 2 se) against n, one line per choice."""
    blocks = [b for b in BLOCKS if b in every]
    fig, axes = plt.subplots(len(blocks), 2, figsize=(6.4, 1.9 * len(blocks) + 0.6), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        drawn = []                                                   # the choices present in this block's rows
        for arm in N0_STYLE:
            g = by_n([r for r in every[block] if r["arm"] == arm])
            if not g:
                continue
            drawn.append(g)
            ns = np.array(list(g))
            st = dict(dict(lw=1.0, ls="-", marker="o", ms=2.2), **N0_STYLE[arm])
            rms = [np.sqrt(np.mean(parts(G)["total"] ** 2)) for G in g.values()]
            cov = [np.mean(np.abs(oracle_T(G)) <= Z95) for G in g.values()]
            axes[i, 0].plot(ns, rms, label=N0_LABEL[arm], **st)
            axes[i, 1].plot(ns, cov, **st)
        ns = np.array(list(drawn[0]))
        se_c = np.sqrt(0.95 * 0.05 / len(next(iter(drawn[0].values()))))
        axes[i, 1].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
        axes[i, 1].axhline(0.95, color=INK2, lw=0.6)
        axes[i, 0].set_yscale("log")
        for ax in axes[i]:
            n_axis(ax, ns)
        axes[i, 0].set_ylabel(BLABEL[block])
    axes[0, 0].set_title(r"RMS of total remainder / $V_{pqv}$")
    axes[0, 1].set_title(r"oracle coverage (0.95 $\pm$ 2 se)")
    for ax in axes[-1]:
        ax.set_xlabel("total rounds $n$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=5.5)
    save(fig, out, "n0_choice")


def targets(collected, regime, arm, out):
    """The four targets under one choice of n0, one row per block: |total remainder| / V_pqv (log scale, lines
    at 0.03 and 0.05), sd T (+- 2 se), oracle coverage (0.95 +- 2 se), and V_pqv / V_CR with the theorem's
    prediction sqrt(C_cal kappa_A log n_tr / kappa_hat) dashed."""
    _, every = load(collected, regime, arm, target=None)
    blocks = [b for b in BLOCKS if b in every and any("target" in r for r in every[b])]
    if not blocks:
        return
    fig, axes = plt.subplots(len(blocks), 4, figsize=(8.0, 1.85 * len(blocks) + 0.5), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        for target, color in TARGET_COLOR.items():
            g = by_n([r for r in every[block] if r["arm"] == arm and r.get("target") == target])
            if not g:
                continue
            ns = np.array(list(g))
            tot = [parts(G)["total"] for G in g.values()]
            axes[i, 0].plot(ns, np.maximum(np.abs([t.mean() for t in tot]), 1e-3), "-o", color=color,
                            label=TARGET_LABEL[target])
            T = [oracle_T(G) for G in g.values()]
            B = np.array([len(t) for t in T]); sd = np.array([t.std(ddof=1) for t in T])
            axes[i, 1].errorbar(ns, sd, yerr=2 * sd / np.sqrt(2 * (B - 1)), fmt="-o", color=color, elinewidth=0.6,
                                capsize=1.2)
            axes[i, 2].plot(ns, [np.mean(np.abs(t) <= Z95) for t in T], "-o", color=color)
            axes[i, 3].plot(ns, [np.mean(col(G, "VCR_ratio")) for G in g.values()], "-o", color=color)
            axes[i, 3].plot(ns, [np.mean(col(G, "VCR_ratio_pred")) for G in g.values()], "--", color=color, lw=0.7)
        axes[i, 0].axhline(0.03, color="#1a9850", lw=0.7, ls="--"); axes[i, 0].axhline(0.05, color="#d73027", lw=0.7, ls=":")
        axes[i, 0].set_yscale("log"); axes[i, 0].set_ylim(1e-3, 1)
        axes[i, 1].axhline(1, color=INK2, lw=0.6)
        se_c = np.sqrt(0.95 * 0.05 / B[-1])
        axes[i, 2].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
        axes[i, 2].axhline(0.95, color=INK2, lw=0.6)
        for ax in axes[i]:
            n_axis(ax, ns)
        axes[i, 0].set_ylabel(BLABEL[block])
    for ax, title in zip(axes[0], (r"|total remainder| / $V_{pqv}$", r"sd $T$ ($\pm$2 se)",
                                   r"oracle coverage (0.95 $\pm$ 2 se)", r"$V_{pqv}/V_{CR}$ (dashed: prediction)")):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("total rounds $n$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=5.5)
    save(fig, out, "targets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default="inference_results/collected")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--arm", default=S.PRIMARY, choices=list(N0_LABEL), help="draw the figures of this choice of n0")
    ap.add_argument("--target", default="dense", choices=list(TARGET_LABEL), help="the target of the figures")
    a = ap.parse_args()
    root = a.out if a.target == "dense" else os.path.join(a.out, a.target)
    for regime, out in (("lowrank", root), ("normal", os.path.join(root, "normal"))):   # the N(0, I) panel: same names, own folder
        chosen, every = load(a.collected, regime, a.arm, a.target)
        if not every:
            continue
        n0_choice(every, out)
        out_arm = os.path.join(out, a.arm)
        if a.target == "dense":
            targets(a.collected, regime, a.arm, out_arm)
        trend(chosen, out_arm)
        qq(chosen, out_arm)
        variance(chosen, out_arm)
        remainder(chosen, out_arm)
        comparator(chosen, out_arm)
        if a.arm != "no_refit":                        # the choice against n0 = n_tr
            refit(every, out_arm, a.arm)
        rank(every, out_arm)
        if "I3" in every:
            tie(every, out_arm)


if __name__ == "__main__":
    main()
