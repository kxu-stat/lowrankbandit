#!/usr/bin/env python3
"""Figures of the MIND regret study, from the tables collect_mind.py wrote.  Run by hand after the collect; it reads,
never writes, the cells.

    python3 figures_mind.py [DIR] [--out figures_mind]          (DIR: the run, default mind_regret)

Each competitor is drawn at its best configuration in hindsight (collected/best.tsv, marked *); every configuration is
in collected/summary.tsv.  The block of the run (run_mind.BLOCKS): down4 (mind_regret, the main text) or main
(mind_regret_main, the appendix; its file names end in _main).  The series as in regret/figures_regret.py (O ours at
c_tau = 1/20; M the matched arm at the T1 of ESTR*; E*, F* our GLM-ESTR and logistic OFUL; the authors' 2y - 1
versions) plus T, the two-track collection of the inference, and the two greedy comparators without the
interaction, each at its best tau: G the auxiliary-covariate greedy (c alone), H the main-effect greedy (c and m;
reported).
  mind_regret_curves      mean cumulative regret against t, band +-2 Monte-Carlo s.e.
  mind_regret_curves_pm1  the same with the authors' 2y - 1 versions added (supplement; down4)
  mind_t1_profile         regret at the horizon against ESTR's five T1: ESTR (both codings) at its best multiplier at
                          each T1 and the matched arm at the same T1; GLM-ESTR at T1 = our tau (hollow diamond);
                          the others as horizontal lines (down4)
  mind_sensitivity        our regret over the c_tau grid relative to c_tau = 1/20 (supplement; down4)
"""
import argparse, os, sys
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "regret"))
import figures_regret as F                    # the simulations' style (rcParams, colors, save)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import run_mind as R

TITLE = {"down4": "MIND, semi-synthetic world (down-sampled click model)", "main": "MIND, semi-synthetic world"}
MAIN_ARM = R.OURS
SERIES = ["O", "T", "M", "E", "F", "G", "H", "Ep", "Fp"]
BEST_METHOD = {"E": "ESTR", "F": "OFUL", "G": "AuxGreedy", "H": "CMGreedy", "Ep": "ESTR_pm1", "Fp": "OFUL_pm1"}
LABEL = dict(F.LABEL, **F.LABEL_BINARY, G=r"auxiliary-covariate greedy$^*$", H=r"main-effect greedy $(c, m)^*$",
             T="Proposed, two tracks (the inference's data)")
COLOR = dict(F.COLOR, G="#008300", H="#1a9e8f", T="#1c4e8a")
MARKER = dict(F.MARKER, G="P", H="X", T="s")
DASH = dict(F.DASH, G=(0, (5, 1)), H="-.", T=(0, (6, 2)))


class Run:
    def __init__(self, root):
        cd = os.path.join(root, "collected")
        self.summary = F.read_tsv(os.path.join(cd, "summary.tsv"))
        self.row = {(s["block"], s["config_id"]): s for s in self.summary}
        self.best = {(b["block"], b["method"]): b["config_id"] for b in F.read_tsv(os.path.join(cd, "best.tsv"))}
        z = {}
        for q in F.read_tsv(os.path.join(cd, "regret_curves.tsv")):
            z.setdefault((q["block"], q["config_id"]), []).append((q["t"], q["mean"], q["se"] if q["se"] != "" else 0.0))
        self.curves = {k: np.array(sorted(v)).T for k, v in z.items()}
        self.blocks = sorted({s["block"] for s in self.summary})

    def config(self, block, s):
        if s == "O":
            return MAIN_ARM
        if s == "T":
            return "two_track"
        if s == "M":
            e = self.best.get((block, "ESTR"))
            return None if e is None else "M_k%d" % F.T1_index(e)
        return self.best.get((block, BEST_METHOD[s]))


def handles(series, line=True):
    return [Line2D([], [], color=COLOR[s], marker=MARKER[s], linestyle=DASH[s] if line else "", label=LABEL[s],
                   markerfacecolor="white" if s == "Et" else COLOR[s]) for s in series]


def regret_curves(run, out, series, name):
    fig, axes = plt.subplots(1, len(run.blocks), figsize=(3.3 * len(run.blocks), 2.6), squeeze=False, layout="constrained")
    for j, block in enumerate(run.blocks):
        ax = axes[0, j]
        tmax = 1
        for s in series:
            key = (block, run.config(block, s))
            if key not in run.curves:
                continue
            t, mu, se = run.curves[key]
            tmax = t.max()
            ax.fill_between(t, mu - 2 * se, mu + 2 * se, color=COLOR[s], alpha=0.15, linewidth=0)
            marks = [k for k, tt in enumerate(t) if tt in F.checkpoints(tmax)]
            ax.plot(t, mu, linestyle=DASH[s], color=COLOR[s], marker=MARKER[s], markevery=marks, markeredgewidth=0)
        ax.set_xlim(0, tmax)
        ax.set_xticks(F.checkpoints(tmax, with_zero=True))
        ax.set_xticklabels(["%gk" % (c / 1000) if c else "0" for c in F.checkpoints(tmax, with_zero=True)])
        ax.set_ylim(bottom=0)
        ax.set_title(TITLE[block], pad=4)
        ax.set_xlabel("round $t$")
    axes[0, 0].set_ylabel("cumulative expected regret")
    shown = [s for s in series if any((b, run.config(b, s)) in run.curves for b in run.blocks)]
    fig.legend(handles=handles(shown), loc="outside lower center", ncol=4, columnspacing=1.4, title=F.STAR_NOTE,
               title_fontsize=6.5)
    F.save(fig, out, name + ("" if run.blocks == ["down4"] else "_" + "_".join(run.blocks)))


def t1_profile(run, out):
    fig, axes = plt.subplots(1, len(run.blocks), figsize=(3.3 * len(run.blocks), 2.8), squeeze=False, layout="constrained")
    for j, block in enumerate(run.blocks):
        ax = axes[0, j]
        T1 = []
        for E, s in (("ESTR", "E"), ("ESTR_pm1", "Ep")):
            pts = [(k, run.best.get((block, "%s@%d" % (E, k)))) for k in range(R.T1_STEPS)]
            pts = [(k, c) for k, c in pts if c is not None]
            if not pts:
                continue
            x = [run.row[(block, c)]["tau"] for _, c in pts]
            T1 = x if E == "ESTR" else T1
            lines = [(s, [c for _, c in pts])] + ([("M", ["M_k%d" % k for k, _ in pts])] if E == "ESTR" else [])
            for ser, cfgs in lines:
                z = [run.row.get((block, c)) for c in cfgs]
                ax.errorbar([xx for xx, q in zip(x, z) if q], [q["regret"] for q in z if q],
                            yerr=[1.96 * (q["se"] or 0.0) for q in z if q], linestyle=DASH[ser], color=COLOR[ser],
                            marker=MARKER[ser], markeredgewidth=0, elinewidth=0.7, capsize=0)
        et = run.best.get((block, "ESTR_tau"))                # GLM-ESTR at T1 = our tau, its best multiplier
        if et is not None:
            q = run.row[(block, et)]
            ax.errorbar([q["tau"]], [q["regret"]], yerr=1.96 * (q["se"] or 0.0), fmt=MARKER["Et"], color=COLOR["Et"],
                        markerfacecolor="white", markeredgewidth=0.9, elinewidth=0.7, capsize=0)
            T1 = [q["tau"]] + list(T1)
        for ser in ("O", "T", "F", "Fp", "G", "H"):
            q = run.row.get((block, run.config(block, ser)))
            if q is None:
                continue
            ax.axhline(q["regret"], color=COLOR[ser], linestyle=DASH[ser], linewidth=1.0)
            ax.axhspan(q["regret"] - 1.96 * (q["se"] or 0.0), q["regret"] + 1.96 * (q["se"] or 0.0), color=COLOR[ser],
                       alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        if T1:
            ax.set_xticks(T1)
            ax.set_xticklabels(["%d" % v for v in T1])
            ax.minorticks_off()
        ax.set_title(TITLE[block], pad=4)
        ax.set_xlabel(r"exploration length $T_1$ (ESTR's grid)")
    axes[0, 0].set_ylabel("expected regret at the horizon")
    shown = ["O", "T", "M", "E", "F", "G", "H", "Ep", "Fp"] + (["Et"] if any(m == "ESTR_tau" for _, m in run.best) else [])
    fig.legend(handles=handles(shown), loc="outside lower center", ncol=4, columnspacing=1.4, title=F.STAR_NOTE,
               title_fontsize=6.5)
    F.save(fig, out, "mind_t1_profile")


def sensitivity(run, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    for block, dash in zip(run.blocks, ("-", "--")):
        ref = run.row.get((block, MAIN_ARM))
        z = sorted([s for s in run.summary if s["block"] == block and s["method"] == "Proposed"], key=lambda s: s["c_tau"])
        if ref is None or not z:
            continue
        ax.errorbar([s["c_tau"] for s in z], [s["regret"] / ref["regret"] for s in z],
                    yerr=[1.96 * (s["se"] or 0.0) / ref["regret"] for s in z], linestyle=dash, color=COLOR["O"],
                    marker="o", markeredgewidth=0, elinewidth=0.7, capsize=0, label=TITLE[block])
    ax.axhline(1.0, color=F.INK2, linewidth=0.5)
    ax.set_xscale("log", base=2)
    ticks = sorted({s["c_tau"] for s in run.summary if s["method"] == "Proposed"})   # the feasible values run
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/%d" % round(1 / c) for c in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r"$c_\tau$")
    ax.set_ylabel(r"regret / regret at $c_\tau = 1/%d$" % round(1 / R.C_TAU))
    ax.legend(fontsize=6.5)
    F.save(fig, out, "mind_sensitivity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="mind_regret")
    ap.add_argument("--out", default="figures_mind")
    a = ap.parse_args()
    run = Run(a.run)
    regret_curves(run, a.out, ["O", "T", "M", "E", "F", "G", "H"], "mind_regret_curves")
    if run.blocks == ["down4"]:                              # the full grid: the main text's block
        regret_curves(run, a.out, ["O", "T", "M", "E", "F", "G", "H", "Ep", "Fp"], "mind_regret_curves_pm1")
        t1_profile(run, a.out)
        sensitivity(run, a.out)


if __name__ == "__main__":
    main()
