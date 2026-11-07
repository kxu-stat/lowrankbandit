#!/usr/bin/env python3
"""The ablation of Step 3 in the adaptive-tie design, the main text's figure (by hand or
submit_tie_ablation.sge): Algorithm 2 against the one-step
debiasing of the policy's own estimator, whole sample and cross-fitted, at one N off the grid of the inference run.
The grid (50 to 12,800) has no N = 10,000, so the collections are drawn here from the package's seed of the replicate,
simulation.seed("I3", "tie", rep, N) = 74,000,000 + 100,000 rep + N.  For N = 10,000 these seeds are used by no other
collection, and they cannot equal a seed of another block (those are not multiples of 1,000).

    python3 tie_ablation.py N REP_FROM REP_TO OUT_DIR [--chunk k --chunks m]
    e.g.  python3 tie_ablation.py 10000 1 1000 ablation_results/tie --chunk 1 --chunks 10
          python3 tie_ablation.py 200 1 2 ablation_debug                     (the debug run, seconds)

--chunk k --chunks m: only the k-th of m contiguous blocks of REP_FROM..REP_TO.

Every replicate: algorithms.collect with I3's constants (12 exploration rounds, then two greedy tracks over N rounds),
then the seven arms of ablation_step3 (two Step 2's, Step 3 in three weights) with its two checks: every cross-fitted
predictable arm equals the package's calibrate_loadings, and onestep_full equals its comparator mu_ols (relative
1e-9).  In this design every one-step arm is the average of the two tracks' least squares, so they coincide.
Output: OUT_DIR/part_tie_N<N>_<FROM>-<TO>.tsv in ablation_step3's columns, stamped with the package version.  A
replicate already in the part is skipped, so a stopped run is completed by running it again; a part written by
another version is never completed (write to a new OUT_DIR).
"""
import argparse, math, os, time
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np
import ablation_step3 as T
S, A, R = T.S, T.A, T.R


def replicate(N, rep):
    D = S.TieDesign()
    _, c_lam, _ = S.constants("I3")
    rec = A.collect(D, S.TIE_TAU, N, np.random.default_rng(S.seed("I3", "tie", rep, N)), c_lam)
    n0 = R.split(dict(S.arms("I3"))[S.PRIMARY], N, D, rec)
    base = dict(record="I3_r%04d_N%05d" % (rep, N), block="I3", regime="tie", rep=rep, n=N, N=N, n0=n0)
    return T.arm_rows(D, rec, N, n0, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, help="post-exploration rounds")
    ap.add_argument("rep_from", type=int)
    ap.add_argument("rep_to", type=int)
    ap.add_argument("out", help="the directory of the parts")
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    a = ap.parse_args()
    reps = list(range(a.rep_from, a.rep_to + 1))
    size = math.ceil(len(reps) / a.chunks)                           # replicates per block
    mine = reps[(a.chunk - 1) * size: a.chunk * size]
    if not mine:
        raise SystemExit("chunk %d of %d has no replicate" % (a.chunk, a.chunks))
    os.makedirs(a.out, exist_ok=True)
    part = os.path.join(a.out, "part_tie_N%05d_%04d-%04d.tsv" % (a.N, mine[0], mine[-1]))
    done = {int(row["rep"]) for row in T.done_in(part)}
    print("tie_ablation: N = %d, replicates %d-%d -> %s (%d done; package %s)" % (a.N, mine[0], mine[-1], part, len(done),
                                                                                R.PACKAGE), flush=True)
    for rep in mine:
        if rep in done:
            continue
        t0 = time.time()
        T.append(part, replicate(a.N, rep))
        print("rep %d (%.1fs)" % (rep, time.time() - t0), flush=True)
    print("OK %s" % part, flush=True)


if __name__ == "__main__":
    main()
