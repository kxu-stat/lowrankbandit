#!/usr/bin/env python3
"""The oracle calibration martingale of every target on the stored MIND inference streams, and the control-variate
estimates of mean T.  Run after collect_mind_inference.py, by hand (about 3-10 s per stream and n):

    python3 oracle_mind_inference.py PHASE --setup SETUP_NPZ [--dir DIR] [--world down4] [--n N] [--arm ARM] [--kappa K]

--n defaults to the world's figure n (run_mind_inference.figure_n: 40,000 in the down-sampled world, 80,000 in the
main world).

Algorithm 2's correction is sum_t omega_t (y_t - p_t) with predictable weights omega_t from the calibration recursion
(inference/algorithms.calibrate_loadings).  Here the same function runs with the true parameter, the true tangent frame
and one information scale kappa for every stream (nothing estimated: n0 = 0), on the stored rows and clicks of the two
tracks (cells/KEY_stream.npz).  Its weights are predictable, so its mean is exactly 0, and
T_oracle = (mu - mu*) / V_pqv is the oracle martingale over its predictable standard error.  It shares the streams'
clicks with T, so mean(T) - beta mean(T_oracle), beta the least-squares slope of T on T_oracle across the streams, is a
control-variate estimate of E T with a much smaller Monte Carlo error (the variance of T - beta T_oracle in place of
that of T).  kappa defaults to the world's mean kappa_hat at the largest n of the grid (summary.tsv of the collect).
T is the choice ARM's (default the displayed choice, the design-stage split with cap 3/4).

Writes DIR/collected/WORLD/oracle_n<n>.tsv (stream x target T_oracle) and oracle_n<n>_cv.tsv (per target: mean T,
its se, the control-variate mean, its se, beta).
"""
import argparse, csv, glob, gzip, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inference"))
import algorithms as IA
import run_mind_inference as R
import mind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(R.PHASES))
    ap.add_argument("--setup", required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--world", default="down4", choices=sorted(R.WORLDS))
    ap.add_argument("--n", type=int, default=None, help="default: the world's figure n")
    ap.add_argument("--arm", default=R.IS.PRIMARY)            # the displayed choice of n0
    ap.add_argument("--kappa", type=float, default=None)
    a = ap.parse_args()
    folder = a.dir or "mind_inference_%s" % a.phase
    out_dir = os.path.join(folder, "collected", a.world)
    setup0 = dict(np.load(a.setup))
    data = mind.load(os.path.join(HERE, "data", "mind_small_prepared_glove.npz"), R.D, setup0)
    setup = R.world_setup(setup0, a.world)
    Dsg = R.MindDesign(data, setup)
    names, As = [str(s) for s in setup["target_names"]], setup["target_A"]
    grid = list(R.grid_of(a.phase, a.world))
    a.n = a.n or R.figure_n(a.phase, a.world)
    kappa = a.kappa
    if kappa is None:                                          # the world's mean kappa_hat at the largest n
        rows = [r for r in csv.DictReader(open(os.path.join(out_dir, "summary.tsv")), delimiter="\t")
                if r["arm"] == a.arm and int(float(r["n"])) == grid[-1]]
        kappa = float(np.mean([float(r["kap_hat"]) for r in rows]))
    F = dict(th_hat=[Dsg.theta, Dsg.theta], frames=[(Dsg.Us, Dsg.Vs), (Dsg.Us, Dsg.Vs)], kap=[kappa, kappa], n0=0)
    files = sorted(glob.glob(os.path.join(folder, "cells", "%s_%s_b*_stream.npz" % (a.phase, a.world))))
    orc = {}
    for path in files:
        rep = int(os.path.basename(path).split("_")[2][1:])
        st = np.load(path)
        chosen, clicks, tau = st["chosen"], st["clicks"].astype(float), int(st["tau"])
        N = a.n - tau
        n_tr = N // 2
        rec = dict(Z=[], Y=[])
        for j in (0, 1):
            idx = tau + j + 2 * np.arange(n_tr)                # track j's first n_tr rounds
            rec["Z"].append(Dsg.rows(chosen[idx]))
            rec["Y"].append(clicks[idx])
        res = IA.calibrate_loadings(Dsg, rec, N, As, F)
        orc[rep] = [(o["mu"] - o["mu_star"]) / o["V_pqv"] for o in res]
    with open(os.path.join(out_dir, "oracle_n%d.tsv" % a.n), "w") as f:
        f.write("\t".join(["rep"] + names) + "\n")
        for rep in sorted(orc):
            f.write("\t".join([str(rep)] + ["%.5f" % v for v in orc[rep]]) + "\n")
    T = {}
    with gzip.open(os.path.join(out_dir, "rows.tsv.gz"), "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["arm"] == a.arm and int(float(r["n"])) == a.n:
                T[(r["target"], int(r["rep"]))] = (float(r["mu"]) - float(r["mu_star"])) / float(r["V_pqv"])
    reps = sorted(set(orc) & {b for _, b in T})
    with open(os.path.join(out_dir, "oracle_n%d_cv.tsv" % a.n), "w") as f:
        f.write("target\tB\tmean_T\tse\tcv_mean_T\tcv_se\tbeta\tkappa\n")
        for k, name in enumerate(names):
            x = np.array([T[(name, b)] for b in reps]); y = np.array([orc[b][k] for b in reps])
            beta = float(np.cov(x, y)[0, 1] / np.var(y, ddof=1))
            d = x - beta * y
            f.write("%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
                name, len(reps), x.mean(), x.std(ddof=1) / np.sqrt(len(x)), x.mean() - beta * y.mean(),
                d.std(ddof=1) / np.sqrt(len(d)), beta, kappa))
    print("oracle martingale of %d streams of %s at n = %d (kappa %.4f): %s" % (len(reps), a.world, a.n, kappa,
                                                                               os.path.join(out_dir, "oracle_n%d_cv.tsv" % a.n)))


if __name__ == "__main__":
    main()
