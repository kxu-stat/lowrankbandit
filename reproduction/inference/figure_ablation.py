#!/usr/bin/env python3
"""Summary and figures of the ablation of Step 3 (by hand).

    python3 figure_ablation.py OUT_DIR [--run RUN_DIR] [--vcr VCR_TSV] [--show BLOCK:TARGET:N,...] [--blocks I3,I1]
                               [--target dense] [--name N]
    e.g.  python3 figure_ablation.py ablation_results/tie --blocks "" --name ablation_tie         (the main text)
          python3 figure_ablation.py ablation_results/parts --run inference_results --name ablation_appendix \
              --show I2l:dense:40000,I2l:entry_max:40000,I2l:rank1:40000,I2l:entry_min:40000,I2l:dense:10000,I1:dense:10000,I1d32:dense:10000

Reads OUT_DIR/part_*.tsv (tie_ablation.py or ablation_step3.py) and prints the package versions that wrote them.
With --run (the run's directory, whose collected/ tables exist), it first checks the parts against the run -- every
record of the plan has its rows, and the arms ours and no_refit equal the run's design_75 and no_refit rows (mu,
relative 1e-6) -- and takes from the design_75 rows the reference of the sd panel: mean V_CR / mean V_pqv, the
Cramer-Rao standard error of the collected design over Algorithm 2's oracle standard error (what the cross-fitted one
step's sd_MC / that of Algorithm 2 should be if it is efficient).  --vcr takes that reference from another run's
collected/vcr.tsv instead (1 / VCR_ratio).

Every arm is standardized by its own Monte Carlo sd at each grid point, z = (mu_hat - mu*) / sd_MC: no arm's standard
error enters the figures (the one-step arms have no oracle standard error).  Writes
  OUT_DIR/summary.tsv  per block, target, n and arm: B; bias / sd_MC and its se; sd_MC relative to ours; the KS
                       distance of z to N(0, 1); the coverage of mu_hat +- 1.96 sd_MC and its two one-sided misses;
                       RMSE relative to ours; with the arm's own standard error V: mean V / sd_MC, the coverage of
                       mu_hat +- 1.96 V and its two one-sided misses; and the bias split into the mean of the noise
                       term (the adaptive-collection part: zero for the predictable weights) and the mean of the
                       remainder, both / sd_MC with their se;
  OUT_DIR/NAME_hist.pdf/png   one row per BLOCK:TARGET:N of --show (the displayed settings; default SHOW): the
                       histograms of z of the three arms of HIST with the N(0, 1) density, and their QQ plot (gray: the
                       95 % simultaneous KS band for Algorithm 2);
  OUT_DIR/NAME_trend.pdf/png  one row per block of --blocks (target --target): bias / sd_MC (+- 2 se), KS distance (null
                       95 % value dashed), coverage with sd_MC (0.95 +- 2 se), sd_MC relative to Algorithm 2 (dotted:
                       the reference V_CR / V_pqv), against n.
Color: the weights (ink: the predictable recursion; orange: one step); fill or line: cross-fitting (filled and solid:
yes; open and dashed: no).  numpy, scipy and matplotlib only (the package's requirements).
"""
import argparse, csv, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {  # arm: color, line style, marker face, label
    "ours":         ("#1a1a19", "-",  "full", "Algorithm 2 (predictable weights, cross-fitted)"),
    "nuc_tan_nocf": ("#eb6834", "--", "none", "one step on the policy's estimator, whole sample"),
    "nuc_tan":      ("#eb6834", "-",  "full", "one step on the policy's estimator, cross-fitted"),
}
HIST = (("ours", "Algorithm 2"), ("nuc_tan_nocf", "one step, whole sample"), ("nuc_tan", "one step, cross-fitted"))
SHOW = "I3:main_effect:10000"      # the main text's setting (tie_ablation.py)
CHECKS = {"ours": "design_75", "no_refit": "no_refit"}   # arm of the parts: arm of the run with the same numbers
BLABEL = {"I3": "adaptive tie", "I1": "identity link", "I2l": r"logistic, $n_b = 1$", "I1d32": r"identity link, $d = 32$"}
XLABEL = {"I3": "post-exploration rounds $N$"}
TLABEL = {"dense": "dense target", "entry_max": "entry, largest tangent part", "entry_min": "entry, smallest tangent part",
          "rank1": "rank-one target"}
Z95 = 1.959964
INK, INK2, GRID = "#1a1a19", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.0, "lines.markersize": 2.5, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def read_parts(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "part_*.tsv"))):
        rows += list(csv.DictReader(open(path), delimiter="\t"))
    if not rows:
        raise SystemExit("no part in %s" % out)
    print("%d rows, written by package %s" % (len(rows), ", ".join(sorted({str(r.get("package")) for r in rows}))))
    for r in rows:
        for k in ("rep", "n", "N", "n0"):
            r[k] = int(r[k])
        for k in ("mu", "V", "mu_star", "noise"):
            r[k] = float(r.get(k) or "nan")                          # noise: absent in the first pilot's parts
    return rows


def check_run(rows, run):
    """The parts against the run: completeness, and the arms of CHECKS equal the run's rows.  Returns {(block, n,
    target): mean V_CR / mean V_pqv over the matched design_75 rows} (I3 has no V_CR at its smallest N)."""
    per_record = {}
    for r in rows:
        per_record[r["record"]] = per_record.get(r["record"], 0) + 1
    acc, bad, matched, wanted = {}, [], 0, 0
    for block, regime in sorted({(r["block"], r["regime"]) for r in rows}):
        n_records = len(glob.glob(os.path.join(run, "records", "%s_%s_r*.npz" % (block, regime))))
        mine = {r["record"] for r in rows if r["block"] == block}
        counts = sorted({per_record[k] for k in mine})
        print("%s %s: %d of %d records in the parts, rows per record %s" % (block, regime, len(mine), n_records, counts))
        if len(mine) != n_records or len(counts) != 1:
            bad.append("%s: records missing or incomplete" % block)
        ours = {(CHECKS[r["arm"]], r["rep"], r["n"], r["target"]): r["mu"] for r in rows
                if r["block"] == block and r["arm"] in CHECKS}
        wanted += len(ours)
        key_n = "N" if block == "I3" else "n"
        for w in csv.DictReader(open(os.path.join(run, "collected", "%s_%s.tsv" % (block, regime))), delimiter="\t"):
            key = (w["arm"], int(w["rep"]), int(float(w[key_n])), w["target"])
            if key not in ours:
                continue
            matched += 1
            if abs(ours[key] - float(w["mu"])) > 1e-6 * max(1.0, abs(float(w["mu"]))):
                bad.append("%s %s rep %d n %d %s: ours %.12g, run %s" % ((block,) + key + (ours[key], w["mu"])))
            if w["arm"] == "design_75":
                acc.setdefault((block, key[2], key[3]), []).append((float(w["V_CR"]), float(w["V_pqv"])))
    print("ours and no_refit against the run's rows: %d of %d matched, %d differ" % (matched, wanted, len(bad)))
    if bad or matched != wanted:
        raise SystemExit("CHECK FAILED: " + "; ".join(bad[:5]))
    return {k: np.mean([a for a, _ in v]) / np.mean([b for _, b in v]) for k, v in acc.items()
            if np.all(np.isfinite(v))}


def vcr_reference(path, rows):
    """{(block, n, target): 1 / VCR_ratio} from a run's collected/vcr.tsv, in the regime of the parts."""
    regime = {r["block"]: r["regime"] for r in rows}
    return {(w["block"], int(w["n"]), w["target"]): 1.0 / float(w["VCR_ratio"])
            for w in csv.DictReader(open(path), delimiter="\t")
            if regime.get(w["block"]) == w["regime"] and np.isfinite(float(w["VCR_ratio"]))}


def summarise(rows, pred):
    G = {}
    for r in rows:
        G.setdefault((r["block"], r["target"], r["n"], r["arm"]), []).append(r)
    out = []
    for (block, target, n, arm), g in sorted(G.items()):
        e = np.array([r["mu"] - r["mu_star"] for r in g])
        V = np.array([r["V"] for r in g])
        noise = np.array([r["noise"] for r in g])
        ok = np.isfinite(e) & np.isfinite(V)
        e, V, noise = e[ok], V[ok], noise[ok]
        if len(e) < 5:
            continue
        sd = e.std(ddof=1)
        z, t = e / sd, e / V
        ref = np.array([r["mu"] - r["mu_star"] for r in G[(block, target, n, "ours")]])
        out.append(dict(block=block, target=target, n=n, arm=arm, B=len(e), bias_sd=e.mean() / sd, se=1 / np.sqrt(len(e)),
                        sd_rel=sd / ref.std(ddof=1), ks=stats.kstest(z, "norm").statistic,
                        ks_null95=stats.kstwo.ppf(0.95, len(e)), cover_mc=np.mean(np.abs(z) <= Z95),
                        lo_mc=np.mean(z < -Z95), hi_mc=np.mean(z > Z95),
                        rmse_rel=np.sqrt(np.mean(e ** 2) / np.mean(ref ** 2)), V_over_sd=V.mean() / sd,
                        cover_V=np.mean(np.abs(t) <= Z95), lo_V=np.mean(t < -Z95), hi_V=np.mean(t > Z95),
                        noise_bias_sd=noise.mean() / sd, noise_se=noise.std(ddof=1) / sd / np.sqrt(len(e)),
                        rem_bias_sd=(e - noise).mean() / sd, rem_se=(e - noise).std(ddof=1) / sd / np.sqrt(len(e)),
                        ref_sd_rel=pred.get((block, n, target), np.nan) if arm == "nuc_tan" else np.nan, z=z))
    return out


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote %s/%s.pdf, .png" % (out, name))


def hist_qq(S_, show, out, name):
    """Histograms of z of the arms of HIST and their QQ plot, one row per (block, target, n) of show."""
    fig, axes = plt.subplots(len(show), 4, figsize=(8.0, 1.85 * len(show) + 0.3), squeeze=False, layout="constrained")
    xx = np.linspace(-4.5, 4.5, 200)
    for i, (block, target, n) in enumerate(show):
        g = {s["arm"]: s for s in S_ if (s["block"], s["target"], s["n"]) == (block, target, n)}
        edges = np.linspace(-4.5, 4.5, 37 if g["ours"]["B"] >= 300 else 19)   # bins of 0.25 (0.5 below 300 replicates)
        for c, (arm, title) in enumerate(HIST):
            ax, (color, _, face, _), s = axes[i, c], STYLE[arm], g[arm]
            ax.hist(np.clip(s["z"], -4.49, 4.49), bins=edges, density=True, color=color, lw=1.0,   # |z| > 4.5: end bins
                    histtype="stepfilled" if face == "full" else "step", alpha=0.65 if face == "full" else 1.0)
            ax.plot(xx, stats.norm.pdf(xx), color=INK2, lw=0.9, ls="--")
            ax.axvline(0, color=INK2, lw=0.5)
            ax.text(0.03, 0.97, "bias %+.2f sd\nKS %.2f" % (s["bias_sd"], s["ks"]), transform=ax.transAxes, va="top",
                    fontsize=6, color=INK)
            ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 0.62)
            ax.set_xlabel("$z$")
            if i == 0:
                ax.set_title(title)
        ax = axes[i, 3]
        qs = {}
        for arm, _ in HIST:
            z = np.sort(g[arm]["z"]); B = len(z); j = np.arange(1, B + 1); qs[arm] = (stats.norm.ppf((j - 0.5) / B), z)
        q, z = qs["ours"]; B = len(z); j = np.arange(1, B + 1); dks = stats.kstwo.ppf(0.95, B)
        ax.fill_between(q, stats.norm.ppf(np.clip(j / B - dks, 1e-6, 1)), stats.norm.ppf(np.clip((j - 1) / B + dks, 0, 1 - 1e-6)),
                        color=GRID, lw=0)
        ax.plot([-3.5, 3.5], [-3.5, 3.5], color=INK2, lw=0.6, ls="--")
        for arm in ("nuc_tan_nocf", "nuc_tan", "ours"):                  # Algorithm 2 on top
            color, _, face, _ = STYLE[arm]
            q, z = qs[arm]
            ax.plot(q, np.clip(z, -4.9, 3.9), "o", ms=1.8, mec=color, mew=0.5, mfc=color if face == "full" else "white",
                    label=dict(HIST)[arm])
        if i == 0:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=5.5, handletextpad=0.2, borderaxespad=0.2)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-5, 4)
        ax.set_xlabel("N(0, 1) quantile")
        if i == 0:
            ax.set_title("QQ plot of $z$")
        axes[i, 0].set_ylabel("%s%s\n%s = %s" % (BLABEL[block], "" if block == "I3" else ", " + TLABEL.get(target, target),
                                                 "N" if block == "I3" else "n", format(n, ",")))
    save(fig, out, name)


TICKS = {"I3": (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)}
TICKS_N = (500, 1000, 2000, 3000, 5000, 10000, 20000, 60000)
SD_TICKS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10)


def n_axis(ax, ns, block):
    """The package's n axis (figures_inference.n_axis): log scale, a few plain ticks."""
    ax.set_xscale("log")
    ax.minorticks_off()
    t = [x for x in TICKS.get(block, TICKS_N) if min(ns) <= x <= max(ns)]
    ax.set_xticks(t)
    ax.set_xticklabels(["%d" % x if x < 1000 else "%gk" % (x / 1000) for x in t])
    ax.set_xlabel(XLABEL.get(block, "total rounds $n$"))


def trend(S_, blocks, target, out, name):
    fig, axes = plt.subplots(len(blocks), 4, figsize=(8.0, 1.75 * len(blocks) + 0.6), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        tgt = "main_effect" if block == "I3" else target
        ax = axes[i]
        sel = lambda arm: sorted((s for s in S_ if s["block"] == block and s["target"] == tgt and s["arm"] == arm),
                                 key=lambda s: s["n"])
        for arm, (color, ls, face, lab) in STYLE.items():
            g = sel(arm)
            x = np.array([s["n"] for s in g])
            kw = dict(color=color, ls=ls, marker="o", mfc=color if face == "full" else "white", mec=color, label=lab)
            ax[0].errorbar(x, [s["bias_sd"] for s in g], yerr=[2 * s["se"] for s in g], elinewidth=0.6, capsize=1.2, **kw)
            ax[1].plot(x, [s["ks"] for s in g], **kw)
            ax[2].plot(x, [s["cover_mc"] for s in g], **kw)
            ax[3].plot(x, [s["sd_rel"] for s in g], **kw)
            if arm == "nuc_tan" and np.any(np.isfinite([s["ref_sd_rel"] for s in g])):
                ax[3].plot(x, [s["ref_sd_rel"] for s in g], ":", color=INK2, lw=1.0,
                           label=r"$V_{CR}/V_{pqv}$: Cramér–Rao over Algorithm 2's oracle s.e.")
        g = sel("ours")
        x = np.array([s["n"] for s in g])
        ax[0].axhline(0, color=INK2, lw=0.6)
        ax[1].plot(x, [s["ks_null95"] for s in g], "--", color="#b2182b", lw=0.8)
        ax[1].set_ylim(bottom=0)
        se_c = np.sqrt(0.95 * 0.05 / g[-1]["B"])
        ax[2].axhspan(0.95 - 2 * se_c, 0.95 + 2 * se_c, color=GRID, lw=0)
        ax[2].axhline(0.95, color=INK2, lw=0.6)
        ax[3].axhline(1, color=INK2, lw=0.6)
        ax[3].set_yscale("log")
        ax[3].minorticks_off()
        lo, hi = ax[3].get_ylim()
        t = [v for v in SD_TICKS if lo <= v <= hi]
        ax[3].set_yticks(t); ax[3].set_yticklabels(["%g" % v for v in t])
        for a in ax:
            n_axis(a, x, block)
        ax[0].set_ylabel(BLABEL[block] + ("" if block == "I3" else "\n" + TLABEL.get(tgt, tgt)))
    titles = (r"bias / $\mathrm{sd}_{MC}$ ($\pm$2 se)", r"KS distance of $z$ to $N(0,1)$",
              r"coverage of $\hat\mu \pm 1.96\,\mathrm{sd}_{MC}$", r"$\mathrm{sd}_{MC}$ relative to Algorithm 2")
    for a, t in zip(axes[0], titles):
        a.set_title(t)
    h, l = [], []
    for a in axes.flat:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="outside upper center", ncol=2, fontsize=6)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", default=None)
    ap.add_argument("--vcr", default=None, help="another run's collected/vcr.tsv for the sd panel's reference")
    ap.add_argument("--show", default=SHOW, help="the histogram figure's rows, BLOCK:TARGET:N,...")
    ap.add_argument("--blocks", default="I3,I1,I2l,I1d32", help="the trend figure's rows")
    ap.add_argument("--target", default="dense")
    ap.add_argument("--name", default="ablation")
    a = ap.parse_args()
    rows = read_parts(a.out)
    pred = check_run(rows, a.run) if a.run else (vcr_reference(a.vcr, rows) if a.vcr else {})
    S_ = summarise(rows, pred)
    if not S_:
        print("no grid point has 5 or more replicates: no summary, no figure (the debug run)")
        return
    keys = [k for k in S_[0] if k != "z"]
    with open(os.path.join(a.out, "summary.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in S_)
    have = {(s["block"], s["target"], s["n"]) for s in S_}
    show = [(b, t, int(n)) for b, t, n in (x.split(":") for x in a.show.split(","))]
    show = [x for x in show if x in have]
    if show:
        hist_qq(S_, show, a.out, a.name + "_hist")
    else:
        print("none of --show is in the parts: no histogram figure")
    blocks = [b for b in a.blocks.split(",") if any(s["block"] == b for s in S_)]
    if blocks:                                                       # --blocks "": no trend figure
        trend(S_, blocks, a.target, a.out, a.name + "_trend")


if __name__ == "__main__":
    main()
