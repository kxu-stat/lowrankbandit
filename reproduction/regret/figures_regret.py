#!/usr/bin/env python3
"""Figures of the regret experiment, from the tables collect_regret.py wrote.

    python3 figures_regret.py [--out figures] [RUN_DIR ...]

RUN_DIR: output roots of run_regret.py (default: regret_gaussian and regret_binomial, those collected).  A run of the
tiers approx or d32 writes its figures into OUT/<tiers>/ (the same names), and has no matched arm.

Each competitor is drawn at its best configuration in hindsight (best.tsv: the configuration of the authors' grid with
the smallest mean regret, marked *); every configuration's mean is in collected/summary.tsv (the supplement table).
The series:
  O               our method at the reported c_tau (collect_regret.MAIN_ARM, c_tau = simulation.C_TAU)
  M               the matched arm at the T1 of ESTR's best configuration
  E*, F*          ESTR and OFUL at their best configurations (binary rewards: our logistic versions, the main text)
  E*, F* (2y - 1) binary rewards only: the authors' 2y - 1 versions (the supplement)
Figures, PDF and PNG; LINK = identity | logit (the reward family), DESIGN = sphere | normal:
  regret_curves_LINK_DESIGN_KK   mean cumulative regret against t, one panel per (d, r) x noise level (the tier
                                 approx: one row per truth, sigma_j ~ j^-2 and j^-1.5)
  regret_curves_main_identity    the main-text figure: both designs side by side, d = 16 with d/r = 16, 8, 4 (MAIN_ROWS),
                                 rho = 0.5 and 1 (MAIN_LEVELS), K = 16
  t1_profile_LINK_DESIGN_KK      regret at the horizon against ESTR's five T1: ESTR at its best multiplier at each T1 and
                                 the matched arm at the same T1; O and F* as horizontal lines (P1 and P6); ESTR at
                                 T1 = our tau (its best multiplier, hollow diamond) left of the grid (P7)
  final_regret_LINK_DESIGN       mean regret at the horizon per series and case, 95% intervals
  paired_contrasts_LINK_DESIGN   the paired differences behind P1, P2, P3, P6 and P7, 95% intervals
                                 (binary rewards: ..._pm1 has the contrasts with the authors' 2y - 1 versions, reported)
  advantage_LINK_DESIGN          (E* - O) / E* against d/r (P5); not drawn for a run with one d/r (the tier approx)
  sensitivity_LINK_DESIGN        our regret over the c_tau grid, relative to the reported c_tau (supplement); not drawn
                                 for a run with one c_tau (the tiers approx and d32)
  regret_scaling_LINK            (tier scale, OUT/scale/) the regret at the horizon against n, every run tuned to its own
                                 n: O, E* and F* (each at its best configuration at that n) with 95% intervals, the
                                 OLS line of O on log n dashed; log n axis, one panel per design and noise level
  regret_scaling_loglog_LINK     the same on log-log axes (a straight line of slope a: regret of order n^a)
Regret is in the model's units (Gaussian: the signal sd is 1; binary: the probability scale).
Color follows the series throughout; markers and line styles are the second encoding, so every figure reads in
grayscale.
"""
import argparse, csv, os, re
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import simulation as S

MAIN_ARM = "A_t%d" % S.C_TAU_GRID.index(S.C_TAU)      # as in collect_regret.py

# ---------------------------------------------------------------- style
SERIES = ["O", "M", "E", "Er4", "F", "Ep", "Fp"]        # Ep, Fp: binary rewards only; Er4: the tier approx
BEST_METHOD = {"E": "ESTR", "F": "OFUL", "Ep": "ESTR_pm1", "Fp": "OFUL_pm1", "Er4": "ESTR_r4"}   # best.tsv's methods
LABEL = {"O": "Proposed", "M": r"Proposed, $\tau = T_1$ of ESTR$^*$", "E": r"ESTR$^*$", "F": r"OFUL$^*$",
         "Ep": r"ESTR$^*$, authors' $2y - 1$", "Fp": r"OFUL$^*$, authors' $2y - 1$", "Er4": r"ESTR$^*$ given rank 4"}
LABEL_BINARY = {"M": r"Proposed, $\tau = T_1$ of logistic ESTR$^*$", "E": r"logistic ESTR$^*$ (our adaptation)",
                "F": r"logistic OFUL$^*$"}
COLOR = {"O": "#2a78d6", "M": "#4a3aa7", "E": "#e34948", "F": "#eda100", "Ep": "#f09796", "Fp": "#f5c96b",
         "Er4": "#8c2d2c"}
MARKER = {"O": "o", "M": "v", "E": "D", "F": "^", "Ep": "d", "Fp": "<", "Er4": "s"}
DASH = {"O": "-", "M": "-", "E": "--", "F": "--", "Ep": ":", "Fp": ":", "Er4": "-."}
# ESTR at T1 = our tau (P7): drawn only in the T1 profile and the paired contrasts, as a hollow diamond in ESTR's color
LABEL["Et"], COLOR["Et"], MARKER["Et"], DASH["Et"] = r"ESTR$^*$ at $T_1 = \tau$ of Proposed", COLOR["E"], "D", ""
LABEL_BINARY["Et"] = r"logistic ESTR$^*$ at $T_1 = \tau$ of Proposed"
STAR_NOTE = r"$^*$: the configuration of the authors' grid with the smallest mean regret (in hindsight)"
DESIGN_TITLE = {"sphere": "arms and contexts uniform on the sphere of radius sqrt(d)",
                "normal": "arms and contexts N(0, I)"}
LINK = {"gaussian": "identity", "binomial": "logit"}    # the reward family, named by its link
# The main-text selection: the rank sweep at d = 16 (d/r = 16, 8, 4) and the two larger noise levels; the full grid
# in regret_curves_* (appendix).
MAIN_ROWS = [(16, 1), (16, 2), (16, 4)]
MAIN_LEVELS = (0.5, 1.0)
INK, INK2, GRID = "#1a1a19", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 3.5, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def read_tsv(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


class Run:
    """The collected tables of one run, indexed for the figures."""

    def __init__(self, root):
        cd = os.path.join(root, "collected")
        self.root = root
        self.summary = read_tsv(os.path.join(cd, "summary.tsv"))
        self.meta = {s["case_id"]: s for s in self.summary}
        self.family = "binomial" if any("_nb" in c for c in self.meta) else "gaussian"
        self.row = {(s["case_id"], s["config_id"]): s for s in self.summary}
        self.best = {(b["case_id"], b["method"]): b["config_id"] for b in read_tsv(os.path.join(cd, "best.tsv"))}
        self.comps = {}
        for c in read_tsv(os.path.join(cd, "comparisons.tsv")):
            self.comps.setdefault((c["case_id"], c["kind"]), []).append(c)
        self.ordering = read_tsv(os.path.join(cd, "ordering.tsv"))
        z = {}
        for q in read_tsv(os.path.join(cd, "regret_curves.tsv")):
            z.setdefault((q["case_id"], q["config_id"]), []).append((q["t"], q["mean"], q["se"] if q["se"] != "" else 0.0))
        self.curves = {k: np.array(sorted(v)).T for k, v in z.items()}         # (t, mean, se)
        matched = any(str(s["config_id"]).startswith("M_k") for s in self.summary)
        self.series = [s for s in SERIES if s == "O" or (s == "M" and matched)
                       or (s in BEST_METHOD and any(m == BEST_METHOD[s] for _, m in self.best))]
        self.has_tau = any(m == "ESTR_tau" for _, m in self.best)          # ESTR at T1 = our tau (P7)
        self.tiers = sorted({str(s.get("tier", "main")) for s in self.summary})
        self.horizon = int(max((s["horizon"] for s in self.summary if isinstance(s.get("horizon"), float)),
                               default=S.HORIZON))                     # 10,000 (a debug run: 2,000)

    def config(self, cid, series):
        """The configuration that a series shows in case cid (None when the case does not have it)."""
        if series == "O":
            return MAIN_ARM
        if series == "M":
            e = self.best.get((cid, "ESTR"))
            return None if e is None else "M_k%d" % T1_index(e)
        return self.best.get((cid, BEST_METHOD[series]))

    def label(self, series):
        return LABEL_BINARY.get(series, LABEL[series]) if self.family == "binomial" else LABEL[series]

    def cases(self, design, K=None):
        return [c for c, s in self.meta.items() if s["design"] == design and (K is None or s["K"] == K)]


def T1_index(config_id):
    """ESTR's T1 index k of a configuration id ESTR_BM_g1_t<kk>_m<mm>[_pm1]."""
    return int(re.search(r"_t(\d\d)_", config_id).group(1))


def noise_label(family, level):
    return (r"$\rho = %g$" % level) if family == "gaussian" else (r"$n_b = %d$" % level)


TRUTH_ORDER = ("exact", "decay2", "decay1.5")              # simulation.TIER_TRUTH: the tier approx has two truths


def truth_of(m):
    """The truth of a case (summary row): "exact" (rank r), or "decay<a>" (the tier approx: full rank, singular
    values sigma_j ~ j^-a)."""
    t = str(m.get("truth", "") or "exact")
    return t if t in TRUTH_ORDER else "exact"


def case_label(d, r, dr, truth="exact"):
    if truth == "exact":
        return r"$d = %d,\ r = %d,\ d/r = %g$" % (d, r, dr)
    return r"$d = %d$, $\sigma_j \propto j^{-%s}$" % (d, truth[len("decay"):]) + "\n" + r"(methods given $r = %d$)" % r


def short_truth(truth):
    """The truth in a row label: empty for the exact truth."""
    return "" if truth == "exact" else r", $\sigma_j \propto j^{-%s}$" % truth[len("decay"):]


def legend_handles(run, series, line=True):
    return [Line2D([], [], color=COLOR[s], marker=MARKER[s], linestyle=DASH[s] if line else "", label=run.label(s),
                   markerfacecolor="white" if s == "Et" else COLOR[s]) for s in series]


def legend_title(family):
    """The definition of rho (Gaussian rewards) and of the star, printed as the legend title."""
    if family != "gaussian":
        return STAR_NOTE
    return (r"$\rho = \sigma\,/\,\mathrm{sd}(m^\top\Theta^* x)$: Gaussian noise sd over the sd of the mean reward of a "
            r"random arm and context ($\|\Theta^*\|_F = 1$); regret in units of $\mathrm{sd}(m^\top\Theta^* x)$" + "\n"
            + STAR_NOTE)


# ---------------------------------------------------------------- the panel grid of one design and K
def panel_grid(run, design, K, draw, name, sharex=True, xlabel=None, ylabel=None, series=None, line=True):
    """One panel per (d, r, truth) row (d/r decreasing) and noise level; draw(ax, case_id) fills a panel."""
    cases = run.cases(design, K)
    meta = run.meta
    pairs = sorted({(meta[c]["d"], meta[c]["rank"], meta[c]["d_over_r"], truth_of(meta[c])) for c in cases},
                   key=lambda z: (-z[2], z[0], TRUTH_ORDER.index(z[3])))
    levels = sorted({meta[c]["noise_level"] for c in cases})
    fig, axes = plt.subplots(len(pairs), len(levels), figsize=(1.9 * len(levels) + 0.9, 1.55 * len(pairs) + 0.7),
                             sharex=sharex, squeeze=False, layout="constrained")
    for i, (d, r, dr, truth) in enumerate(pairs):
        for j, level in enumerate(levels):
            ax = axes[i, j]
            cid = next((c for c in cases if meta[c]["d"] == d and meta[c]["rank"] == r and truth_of(meta[c]) == truth
                        and meta[c]["noise_level"] == level), None)
            if cid is None:
                ax.set_visible(False)
                continue
            draw(ax, cid)
            ax.text(0.04, 0.96, case_label(d, r, dr, truth), transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
                    color=INK2)
            if i == 0:
                ax.set_title(noise_label(run.family, level), pad=4)
            if i == len(pairs) - 1 and xlabel:
                ax.set_xlabel(xlabel)
    if ylabel:
        fig.supylabel(ylabel, fontsize=8)
    series = series or run.series
    fig.legend(handles=legend_handles(run, series, line), loc="outside lower center", ncol=min(len(series), 4),
               columnspacing=1.4, title=legend_title(run.family), title_fontsize=6.5)
    save(fig, os.path.dirname(name), os.path.basename(name))


# ---------------------------------------------------------------- 1. regret curves
def checkpoints(n, with_zero=False):
    """The marked rounds of the linear time axis: every n/5 (n = 10^4: 2000, 4000, ..., 10000)."""
    step = max(int(n) // 5, 1)
    return list(range(0 if with_zero else step, int(n) + 1, step))


def draw_curves(run):
    def draw(ax, cid):
        """Mean cumulative regret of each series, +-2 se band, markers every n/5 rounds, linear time axis from 0."""
        tmax = 1
        for s in run.series:
            cfg = run.config(cid, s)
            if (cid, cfg) not in run.curves:
                continue
            t, mu, se = run.curves[(cid, cfg)]
            tmax = t.max()
            ax.fill_between(t, mu - 2 * se, mu + 2 * se, color=COLOR[s], alpha=0.15, linewidth=0)
            marks = [k for k, tt in enumerate(t) if tt in checkpoints(tmax)]
            ax.plot(t, mu, DASH[s], color=COLOR[s], marker=MARKER[s], markevery=marks, markeredgewidth=0)
        ax.set_xlim(0, tmax)
        ax.set_xticks(checkpoints(tmax, with_zero=True))
        ax.set_xticklabels(["%gk" % (c / 1000) if c and tmax >= 2000 else "%d" % c for c in checkpoints(tmax, True)])
        ax.set_ylim(bottom=0)
    return draw


def regret_curves_main(runs, out, rows, name, K=16):
    """Both covariate designs side by side, each with the (d, r) `rows` at MAIN_LEVELS, K = 16 (Gaussian rewards)."""
    run = runs["gaussian"]
    draw = draw_curves(run)
    fig = plt.figure(figsize=(8.4, 0.5 + 1.75 * len(rows)), layout="constrained")
    gs = fig.add_gridspec(1 + len(rows), 5, width_ratios=[1, 1, 0.08, 1, 1], height_ratios=[0.09] + [1] * len(rows))
    for g, design in enumerate(("sphere", "normal")):
        header = fig.add_subplot(gs[0, 3 * g:3 * g + 2])
        header.axis("off")
        header.text(0.5, 0.5, DESIGN_TITLE[design], ha="center", va="center", fontsize=8.5, transform=header.transAxes)
        for i, (d, r) in enumerate(rows):
            for j, level in enumerate(MAIN_LEVELS):
                ax = fig.add_subplot(gs[1 + i, 3 * g + j])
                cid = next(c for c, z in run.meta.items() if z["design"] == design and z["K"] == K and z["d"] == d
                           and z["rank"] == r and z["noise_level"] == level)
                draw(ax, cid)
                ax.text(0.04, 0.96, case_label(d, r, d / r), transform=ax.transAxes, va="top", ha="left",
                        fontsize=6.5, color=INK2)
                if i == 0:
                    ax.set_title(noise_label(run.family, level), pad=4)
                if i == len(rows) - 1:
                    ax.set_xlabel("round $t$")
                else:
                    ax.tick_params(labelbottom=False)
    fig.supylabel("cumulative regret", fontsize=8)
    fig.legend(handles=legend_handles(run, run.series), loc="outside lower center", ncol=len(run.series),
               columnspacing=1.4, title=legend_title(run.family), title_fontsize=6.5)
    save(fig, out, "%s_%s" % (name, LINK[run.family]))


# ---------------------------------------------------------------- 2. the profile over ESTR's T1 (P1, P6)
def draw_profile(run):
    def draw(ax, cid):
        """x: ESTR's five T1 (log scale); ESTR at its best multiplier at each T1 and the matched arm at that T1, with
        95% intervals; O and F* (binary: also the 2y - 1 versions) as horizontal lines with their 95% bands."""
        T1 = []
        for E, M in (("ESTR", "M"), ("ESTR_pm1", None)):
            pts = [(k, run.best.get((cid, "%s@%d" % (E, k)))) for k in range(S.T1_STEPS)]
            pts = [(k, c) for k, c in pts if c is not None]
            if not pts:
                continue
            s = "E" if E == "ESTR" else "Ep"
            x = [run.row[(cid, c)]["tau"] for _, c in pts]
            T1 = x if E == "ESTR" else T1
            for ser, cfgs in ((s, [c for _, c in pts]), (M, ["M_k%d" % k for k, _ in pts])):
                if ser is None:
                    continue
                z = [run.row.get((cid, c)) for c in cfgs]
                xs = [xx for xx, q in zip(x, z) if q]
                mu = np.array([q["regret"] for q in z if q])
                se = np.array([q["se"] if q["se"] != "" else 0.0 for q in z if q])
                ax.errorbar(xs, mu, yerr=1.96 * se, fmt=DASH[ser], color=COLOR[ser], marker=MARKER[ser],
                            markeredgewidth=0, elinewidth=0.7, capsize=0)
        et = run.best.get((cid, "ESTR_tau"))                  # ESTR at T1 = our tau, its best multiplier (P7)
        if et is not None:
            q = run.row[(cid, et)]
            ax.errorbar([q["tau"]], [q["regret"]], yerr=1.96 * (q["se"] if q["se"] != "" else 0.0), fmt=MARKER["Et"],
                        color=COLOR["Et"], markerfacecolor="white", markeredgewidth=0.9, elinewidth=0.7, capsize=0)
            T1 = [q["tau"]] + list(T1)
        for ser in [s for s in ("O", "F", "Fp") if s in run.series]:
            q = run.row.get((cid, run.config(cid, ser)))
            if q is None:
                continue
            se = q["se"] if q["se"] != "" else 0.0
            ax.axhline(q["regret"], color=COLOR[ser], linestyle=DASH[ser], linewidth=1.0)
            ax.axhspan(q["regret"] - 1.96 * se, q["regret"] + 1.96 * se, color=COLOR[ser], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.5)                 # room for the case label above the short-T1 points
        if T1:
            ax.set_xticks(T1)
            ax.set_xticklabels(["%d" % v for v in T1])
            ax.minorticks_off()
    return draw


# ---------------------------------------------------------------- 3. final regret
def final_regret(run, design, out):
    sm = [s for s in run.summary if s["design"] == design]
    levels = sorted({s["noise_level"] for s in sm})
    cases = sorted({(s["K"], -s["d_over_r"], s["d"], s["rank"], truth_of(s)) for s in sm},
                   key=lambda z: z[:4] + (TRUTH_ORDER.index(z[4]),))
    fig, axes = plt.subplots(1, len(levels), figsize=(2.3 * len(levels) + 1.4, 0.32 * len(cases) + 1.4), sharey=True,
                             squeeze=False, layout="constrained")
    off = {s: 0.12 * (i - (len(run.series) - 1) / 2) for i, s in enumerate(run.series)}
    for j, level in enumerate(levels):
        ax = axes[0, j]
        for i, (K, mdr, d, r, truth) in enumerate(cases):
            cid = next((c for c in run.cases(design, K) if run.meta[c]["d"] == d and run.meta[c]["rank"] == r
                        and truth_of(run.meta[c]) == truth and run.meta[c]["noise_level"] == level), None)
            for s in run.series:
                q = run.row.get((cid, run.config(cid, s))) if cid else None
                if q is None:
                    continue
                se = q["se"] if q["se"] != "" else 0.0
                ax.errorbar(q["regret"], i + off[s], xerr=1.96 * se, fmt=MARKER[s], color=COLOR[s],
                            markeredgewidth=0, elinewidth=0.8, capsize=0, markersize=4)
        ax.set_yticks(range(len(cases)))
        if j == 0:
            ax.set_yticklabels([r"$(%d, %d)$, $K = %d$, $d/r = %g$" % (d, r, K, -mdr) if truth == "exact" else
                                r"$(%d, %d)$, $K = %d$%s" % (d, r, K, short_truth(truth)) for K, mdr, d, r, truth in cases])
        ax.set_xlim(0, None)
        ax.set_title(noise_label(run.family, level), pad=4)
        ax.set_xlabel("regret at $n = %d$" % run.horizon)
        ax.grid(axis="y", linewidth=0.3)
        ax.set_ylim(len(cases) - 0.5, -0.5)
    fig.legend(handles=legend_handles(run, run.series, line=False), loc="outside lower center",
               ncol=min(len(run.series), 4), columnspacing=1.4, title=STAR_NOTE, title_fontsize=6.5)
    save(fig, out, "final_regret_%s_%s" % (LINK[run.family], design))


# ---------------------------------------------------------------- 4. paired contrasts
def contrast(run, cid, kinds, right=None, binding=False):
    """The comparisons.tsv row of case cid among `kinds`: the one against configuration `right`, or (binding=True) the
    one with the largest upper end, which decides an intersection-union test (all intervals below 0 iff this one is)."""
    z = [c for k in kinds for c in run.comps.get((cid, k), []) if right is None or c["right"] == right]
    if not z:
        return None
    return max(z, key=lambda c: c["ci_hi"]) if binding else z[0]


def contrasts_of(run, pm1=False):
    """(title, getter(case) -> row, declared(d/r), reversed(d/r)) per panel.  reversed: the prediction is > 0."""
    if pm1:
        return [("O $-$ F$^*$, $2y - 1$\n(reported)", lambda c: contrast(run, c, ["O_vs_config_OFUL_pm1"],
                                                                         run.config(c, "Fp")), None, None),
                ("O $-$ E$^*$, $2y - 1$\n(reported)", lambda c: contrast(run, c, ["O_vs_config_ESTR_pm1"],
                                                                         run.config(c, "Ep")), None, None),
                ("O $-$ every $2y - 1$\nconfiguration (binding)",
                 lambda c: contrast(run, c, ["O_vs_config_OFUL_pm1", "O_vs_config_ESTR_pm1"], binding=True), None, None),
                ("E$^*$ $-$ F$^*$, $2y - 1$\n(reported)", lambda c: contrast(run, c, ["Ebest_vs_Fbest_pm1"]), None, None)]
    p3 = lambda dr: run.family == S.P3_FAMILY and not (S.DR_LOW < dr < S.DR_REF)
    return [("P1: O $-$ M\n(binding $T_1$)", lambda c: contrast(run, c, ["O_vs_M"], binding=True), True, False),
            ("P2: O $-$ F$^*$", lambda c: contrast(run, c, ["O_vs_config_OFUL"], run.config(c, "F")), True, False),
            ("P2: O $-$ E$^*$", lambda c: contrast(run, c, ["O_vs_config_ESTR"], run.config(c, "E")), True, False),
            ("P2: O $-$ every\nconfiguration (binding)",
             lambda c: contrast(run, c, ["O_vs_config_OFUL", "O_vs_config_ESTR", "O_vs_config_ESTR_r4"], binding=True),
             True, False),
            ("P3: E$^*$ $-$ F$^*$", lambda c: contrast(run, c, ["Ebest_vs_Fbest"]), p3, lambda dr: dr <= S.DR_LOW),
            ("P6: M $-$ E\n(binding $T_1$)", lambda c: contrast(run, c, ["M_vs_ESTR_same_T1"], binding=True), True, False),
            ("P7: O $-$ ESTR at $\\tau$\n(binding multiplier)",
             lambda c: contrast(run, c, ["O_vs_config_ESTR_tau"], binding=True), True, False)]


def paired_contrasts(run, design, out, pm1=False):
    meta = run.meta
    cases = sorted(run.cases(design), key=lambda c: (meta[c]["K"], -meta[c]["d_over_r"], meta[c]["d"],
                                                     TRUTH_ORDER.index(truth_of(meta[c])), meta[c]["noise_level"]))
    # a panel without any row is left out (P1 and P6 in the tiers approx and d32, which have no matched arm)
    panels = [p for p in contrasts_of(run, pm1) if any(p[1](c) is not None for c in cases)]
    if not panels:                                  # one replication (a debug run): no paired intervals
        print("paired_contrasts_%s_%s%s skipped: no paired intervals (one replication?)"
              % (LINK[run.family], design, "_pm1" if pm1 else ""))
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(1.55 * len(panels) + 1.4, 0.19 * len(cases) + 1.1),
                             sharey=True, squeeze=False, layout="constrained")
    for j, (title, get, declared, rev) in enumerate(panels):
        ax = axes[0, j]
        color = COLOR["O"]
        for i, c in enumerate(cases):
            z = get(c)
            dr = meta[c]["d_over_r"]
            yes = declared(dr) if callable(declared) else bool(declared)
            if "P3" in title and meta[c].get("truth", "exact") not in ("exact", None):
                yes = False                                       # P3 is declared for exact low-rank truths only
            if not yes:                                           # no prediction: shaded row
                ax.axhspan(i - 0.5, i + 0.5, color=GRID, alpha=0.6, linewidth=0)
            if z is None:
                continue
            reverse = rev(dr) if callable(rev) else False
            ok = z["ci_lo"] > 0 if reverse else z["ci_hi"] < 0
            ax.plot([z["ci_lo"], z["ci_hi"]], [i, i], color=color if ok else INK2, linewidth=1.0, solid_capstyle="butt")
            ax.plot(z["mean_diff"], i, "o", color=color if ok else "white", markeredgecolor=color if ok else INK2,
                    markeredgewidth=0.8, markersize=3.2)
        ax.axvline(0, color=INK, linewidth=0.6)
        ax.set_title(title, pad=4)
        ax.set_yticks(range(len(cases)))
        if j == 0:
            ax.set_yticklabels([r"(%d,%d) $K$%d %s%s" % (meta[c]["d"], meta[c]["rank"], meta[c]["K"],
                                                         noise_label(run.family, meta[c]["noise_level"]),
                                                         short_truth(truth_of(meta[c]))) for c in cases])
        ax.set_ylim(len(cases) - 0.3, -0.7)
        ax.grid(axis="x", linewidth=0.4)
        ax.grid(axis="y", linewidth=0)
    fig.supxlabel("paired difference of regret at $n = %d$: mean and 95%% interval over replications (filled: the "
                  "interval on the predicted side of 0, below it except P3 at $d/r \\leq 4$; shaded rows: no prediction)"
                  "\n%s; binding: the configuration whose interval reaches highest" % (run.horizon, STAR_NOTE), fontsize=7)
    save(fig, out, "paired_contrasts_%s_%s%s" % (LINK[run.family], design, "_pm1" if pm1 else ""))


# ---------------------------------------------------------------- 5. the advantage against d/r (P5)
def advantage(run, design, out):
    od = [o for o in run.ordering if o["design"] == design and isinstance(o["E_best"], float) and o["E_best"] > 0]
    levels = sorted({o["noise_level"] for o in od})
    shade = ["#86b6ef", "#2a78d6", "#0d366b"]                     # one hue, light -> dark = less -> more noise
    dmark = {16: "^", 32: "D"}
    coord = lambda level: np.log(level) if run.family == "gaussian" else 0.0
    adv = lambda o: (o["E_best"] - o["O"]) / o["E_best"]
    main = [o for o in od if o["K"] == 16]
    if len({o["d_over_r"] for o in main}) < 2:      # the tier approx: P5 needs several d/r (verdict E5: n/a)
        print("advantage_%s_%s skipped: one d/r only" % (LINK[run.family], design))
        return
    # the P5 regression on the K = 16 cases: adv = a + b log(d/r) [+ c log(rho) when there are several noise levels]
    X = np.column_stack([np.ones(len(main)), np.log([o["d_over_r"] for o in main])]
                        + ([[coord(o["noise_level"]) for o in main]] if len(levels) > 1 else []))
    beta = np.append(np.linalg.lstsq(X, np.array([adv(o) for o in main]), rcond=None)[0], 0.0)
    fig, ax = plt.subplots(figsize=(4.0, 2.7), layout="constrained")
    dr = np.array(sorted({o["d_over_r"] for o in main}))
    for k, level in enumerate(levels):
        for o in od:
            if o["noise_level"] != level:
                continue
            if o["K"] == 16:
                ax.plot(o["d_over_r"], adv(o), dmark[int(o["d"])], color=shade[k % 3], markeredgewidth=0, markersize=4.5)
            else:
                ax.plot(o["d_over_r"], adv(o), "s", color=shade[k % 3], markerfacecolor="white", markeredgewidth=0.9,
                        markersize=4)
        if len(dr) > 1:
            grid = np.exp(np.linspace(np.log(dr.min()), np.log(dr.max()), 20))
            ax.plot(grid, beta[0] + beta[1] * np.log(grid) + beta[2] * coord(level), "-", color=shade[k % 3],
                    linewidth=1.0, label=noise_label(run.family, level))
    ax.set_xscale("log", base=2)
    ax.axhline(0, color=INK2, linewidth=0.5)
    ax.set_xlabel(r"complexity $d / r$")
    ax.set_ylabel(r"advantage $(R_{E^*} - R_O) / R_{E^*}$ at $n = %d$" % run.horizon)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker=dmark[d], color=INK2, linestyle="", label=r"$d = %d$" % d)
                for d in sorted({int(o["d"]) for o in main})]
    if any(o["K"] != 16 for o in od):
        handles.append(Line2D([], [], marker="s", color=INK2, markerfacecolor="white", linestyle="", label=r"$K = 32, 64$"))
    fig.legend(handles=handles, loc="outside right upper", fontsize=6.5)
    ax.set_title(r"lines: OLS on $\log (d/r)$%s (slope %.3f)" % (r" and $\log\rho$" if len(levels) > 1 else "", beta[1]),
                 fontsize=7, color=INK2)
    save(fig, out, "advantage_%s_%s" % (LINK[run.family], design))


# ---------------------------------------------------------------- 7. the regret against the horizon (tier scale)
def regret_scaling(run, out, loglog=False):
    sc = read_tsv(os.path.join(run.root, "collected", "scaling.tsv"))
    designs = [g for g in ("sphere", "normal") if any(r["design"] == g for r in sc)]
    levels = sorted({r["noise_level"] for r in sc})
    fig, axes = plt.subplots(len(designs), len(levels), figsize=(2.5 * len(levels) + 0.8, 2.1 * len(designs) + 0.7),
                             squeeze=False, layout="constrained")
    for i, design in enumerate(designs):
        for j, level in enumerate(levels):
            ax = axes[i, j]
            rows = sorted([r for r in sc if r["design"] == design and r["noise_level"] == level], key=lambda r: r["n"])
            n = np.array([r["n"] for r in rows])
            for ser in ("O", "E", "F"):
                mu = np.array([r[ser] for r in rows])
                se = np.array([r[ser + "_se"] if r[ser + "_se"] != "" else 0.0 for r in rows])
                ax.errorbar(n, mu, yerr=1.96 * se, fmt=DASH[ser], color=COLOR[ser], marker=MARKER[ser],
                            markeredgewidth=0, elinewidth=0.7, capsize=0)
            a, b = np.polyfit(np.log(n), [r["O"] for r in rows], 1)[::-1]        # O = a + b log n
            grid = np.exp(np.linspace(np.log(n.min()), np.log(n.max()), 50))
            ax.plot(grid, a + b * np.log(grid), linestyle=(0, (2, 2)), color=INK2, linewidth=0.8)
            ax.set_xscale("log")
            if loglog:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks(n)
            ax.set_xticklabels(["%gk" % (v / 1000) for v in n], rotation=0)
            ax.minorticks_off()
            ax.set_title("%s, %s" % (design, noise_label(run.family, level)), pad=4)
            ax.text(0.04, 0.96, "O = %.0f %s %.1f log n" % (a, "-" if b < 0 else "+", abs(b)), transform=ax.transAxes,
                    va="top", ha="left", fontsize=6.5, color=INK2)
            if i == len(designs) - 1:
                ax.set_xlabel("horizon $n$ (each run tuned to its $n$)")
    fig.supylabel("regret at the horizon", fontsize=8)
    handles = legend_handles(run, ["O", "E", "F"]) + [Line2D([], [], color=INK2, linestyle=(0, (2, 2)),
                                                             label=r"OLS of Proposed on $\log n$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.4, title=STAR_NOTE + " at each $n$",
               title_fontsize=6.5)
    save(fig, out, "regret_scaling%s_%s" % ("_loglog" if loglog else "", LINK[run.family]))


# ---------------------------------------------------------------- 6. the sensitivity to c_tau (supplement)
def sensitivity(run, design, out):
    sm = [s for s in run.summary if s["design"] == design and s["method"] == "Proposed"]
    if len({s["c_tau"] for s in sm}) < 2:           # the tiers approx and d32 run c_tau = 1/20 only
        print("sensitivity_%s_%s skipped: one c_tau only" % (LINK[run.family], design))
        return
    levels = sorted({s["noise_level"] for s in sm})
    shade = {4: "#86b6ef", 8: "#2a78d6", 16: "#0d366b"}           # one hue, light -> dark = d/r 4, 8, 16
    dmark = {16: "^", 32: "D"}
    fig, axes = plt.subplots(1, len(levels), figsize=(2.3 * len(levels) + 1.2, 2.6), sharey=True, squeeze=False,
                             layout="constrained")
    for j, level in enumerate(levels):
        ax = axes[0, j]
        for cid in sorted({s["case_id"] for s in sm if s["noise_level"] == level}):
            ref = run.row.get((cid, MAIN_ARM))
            z = sorted([s for s in sm if s["case_id"] == cid], key=lambda s: s["c_tau"])
            if ref is None or not z:
                continue
            m = run.meta[cid]
            ax.plot([s["c_tau"] for s in z], [s["regret"] / ref["regret"] for s in z], "-" if m["K"] == 16 else ":",
                    color=shade.get(int(m["d_over_r"]), INK2), marker=dmark[int(m["d"])], markeredgewidth=0,
                    linewidth=0.9)
        ax.axvline(S.C_TAU, color=INK2, linewidth=0.5)
        ax.axhline(1.0, color=INK2, linewidth=0.5)
        ax.set_xscale("log", base=2)
        ticks = sorted({s["c_tau"] for s in sm})                     # the values run (simulation.C_TAU_RUN)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["1/%d" % round(1 / c) for c in ticks])
        ax.minorticks_off()
        ax.set_title(noise_label(run.family, level), pad=4)
        ax.set_xlabel(r"$c_\tau$")
    axes[0, 0].set_ylabel(r"regret / regret at $c_\tau = 1/%d$" % round(1 / S.C_TAU))
    handles = [Line2D([], [], color=shade[k], label=r"$d/r = %d$" % k) for k in sorted(shade)
               if any(int(run.meta[s["case_id"]]["d_over_r"]) == k for s in sm)]
    handles += [Line2D([], [], marker=dmark[d], color=INK2, linestyle="", label=r"$d = %d$" % d)
                for d in sorted({int(s["d"]) for s in sm})]
    if any(s["K"] != 16 for s in sm):
        handles.append(Line2D([], [], color=INK2, linestyle=":", label=r"$K = 32, 64$"))
    fig.legend(handles=handles, loc="outside right upper", fontsize=6.5)
    save(fig, out, "sensitivity_%s_%s" % (LINK[run.family], design))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    roots = a.runs or [d for d in ("regret_gaussian", "regret_binomial")
                       if os.path.isfile(os.path.join(d, "collected", "summary.tsv"))]
    if not roots:
        raise SystemExit("no run with collected/summary.tsv here: run collect_regret.py first")
    runs = {}
    for root in roots:
        run = Run(root)
        scale = [t for t in run.tiers if t in S.TIER_HORIZONS]          # the regret against n: its own figures
        if scale:
            for loglog in (False, True):
                regret_scaling(run, os.path.join(a.out, scale[0]), loglog)
            continue
        extra = [t for t in run.tiers if t in ("approx", "d32")]       # their own folder
        out = os.path.join(a.out, "_".join(extra)) if extra else a.out
        if not extra:
            runs[run.family] = run
        for design in sorted({s["design"] for s in run.summary}):
            for K in sorted({int(s["K"]) for s in run.summary if s["design"] == design}):
                tag = "%s_%s_K%d" % (LINK[run.family], design, K)
                panel_grid(run, design, K, draw_curves(run), os.path.join(out, "regret_curves_" + tag),
                           xlabel="round $t$", ylabel="cumulative regret")
                panel_grid(run, design, K, draw_profile(run), os.path.join(out, "t1_profile_" + tag), sharex=False,
                           xlabel=r"exploration length $T_1$ (ESTR's grid; hollow: $T_1 = \tau$)" if run.has_tau
                           else r"exploration length $T_1$ (ESTR's grid)", ylabel="regret at the horizon",
                           series=[s for s in ("O", "M", "E", "F", "Ep", "Fp") if s in run.series]
                           + (["Et"] if run.has_tau else []))
            final_regret(run, design, out)
            paired_contrasts(run, design, out)
            if run.family == S.PM1_FAMILY and any(m == "OFUL_pm1" for _, m in run.best):
                paired_contrasts(run, design, out, pm1=True)
            advantage(run, design, out)
            sensitivity(run, design, out)
    if "gaussian" in runs:
        have = {(s["d"], s["rank"], s["noise_level"], s["design"]) for s in runs["gaussian"].summary if s["K"] == 16}
        for rows, name in ((MAIN_ROWS, "regret_curves_main"),):
            if {(d, r, lv, g) for d, r in rows for lv in MAIN_LEVELS for g in ("sphere", "normal")} <= have:
                regret_curves_main(runs, a.out, rows, name)
            else:
                print("%s skipped: its cases are not all in this run (a debug run)" % name)


if __name__ == "__main__":
    main()
