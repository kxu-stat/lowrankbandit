"""The two example figures of the README and the documentation (run by hand, a few minutes).

    python make_figures.py OUT_DIR

regret.png     cumulative regret of Algorithm 1 against the same greedy policy with a near-zero penalty (c = 1e-4),
               averaged over 30 simulated streams (d1 = d2 = 10, rank 2, K = 20 candidates, 3,000 rounds)
inference.png  Algorithm 2 on 200 simulated streams (d1 = d2 = 6, rank 1, K = 10, 3,000 rounds): the first 60
               95% intervals for m' Theta* x, and the histogram of (mu_hat - mu*) / se against N(0, 1)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lowrankbandit as lrb

OURS, OTHER = "#1f5fa8", "#d9822b"                      # ours blue, the comparison warm
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "savefig.bbox": "tight"})


def regret_figure(path, B=30, d=10, rank=2, K=20, n=3000):
    methods = (("Algorithm 1 (penalty by cross-validation)", "cv", OURS), ("near-zero penalty (c = 1e-4)", 1e-4, OTHER))
    curves = {label: [] for label, _, _ in methods}
    for b in range(B):
        Theta = lrb.low_rank_truth(d, d, rank, rng=100 + b)
        stream = lrb.simulate_stream(n, d, d, K, Theta, noise_sd=0.5, seed=200 + b)
        for label, c_lam, _ in methods:
            policy = lrb.GreedyPolicy(d, d, n, c_lam=c_lam, seed=300 + b)
            curves[label].append(np.cumsum(lrb.run(policy, stream)))
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    t = np.arange(1, n + 1)
    for label, _, color in methods:
        c = np.array(curves[label])
        mean, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(B)
        ax.plot(t, mean, color=color, lw=2, label=label)
        ax.fill_between(t, mean - 2 * se, mean + 2 * se, color=color, alpha=0.18, lw=0)
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative regret")
    ax.set_title("Cumulative regret, mean of %d streams (band: 2 standard errors)" % B, fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return {k: float(np.mean([c[-1] for c in v])) for k, v in curves.items()}


def inference_figure(path, B=200, d=6, rank=1, K=10, n=3000, shown=60):
    rng = np.random.default_rng(7)
    m, x = rng.standard_normal(d), rng.standard_normal(d)
    m, x = m / np.linalg.norm(m), x / np.linalg.norm(x)          # a new ad m and a new user x
    A = lrb.interaction_loading(np.outer(m, x), p0=0)           # the target m' Theta* x
    est, se, truth = np.zeros(B), np.zeros(B), np.zeros(B)
    for b in range(B):
        Theta = lrb.low_rank_truth(d, d, rank, signal=2.0, rng=1000 + b)
        stream = lrb.simulate_stream(n, d, d, K, Theta, noise_sd=0.5, seed=2000 + b)
        policy = lrb.TwoTrackPolicy(d, d, n, seed=3000 + b)
        lrb.run(policy, stream)
        res = lrb.two_track_inference(policy.data(), A, rank=rank)
        est[b], se[b], truth[b] = res["estimate"], res["se"], m @ Theta @ x
    T = (est - truth) / se
    covered = np.abs(T) <= 1.959964
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    for i in range(shown):
        color = OURS if covered[i] else OTHER
        ax1.plot([i, i], [est[i] - 1.96 * se[i] - truth[i], est[i] + 1.96 * se[i] - truth[i]], color=color, lw=1.6)
        ax1.plot(i, est[i] - truth[i], "o", color=color, ms=2.5)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xlabel("simulated stream")
    ax1.set_ylabel(r"interval $-\ \mu^*$")
    ax1.set_title("95%% intervals for $m^\\top\\Theta^* x$: %d of %d cover" % (covered[:shown].sum(), shown), fontsize=10)
    grid = np.linspace(-4, 4, 200)
    ax2.hist(T, bins=np.arange(-4, 4.01, 0.4), density=True, color=OURS, alpha=0.75, edgecolor="white")
    ax2.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color="black", lw=1.5, label="N(0, 1)")
    ax2.set_xlabel(r"$(\hat\mu - \mu^*)\, /\, \widehat{\mathrm{se}}$")
    ax2.set_title("coverage %.3f over %d streams" % (covered.mean(), B), fontsize=10)
    ax2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"coverage": float(covered.mean()), "mean T": float(T.mean()), "sd T": float(T.std(ddof=1))}


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    print("final mean regret:", regret_figure(os.path.join(out, "regret.png")))
    print("inference:", inference_figure(os.path.join(out, "inference.png")))
