#!/usr/bin/env python3
"""Replay the recorded collections of an inference run (by hand).

    python3 replay_inference.py RUN_DIR [--only NAME] [--max K] [--rows]

run_inference.py writes every collection as RUN_DIR/records/NAME.npz (NAME = BLOCK_REGIME_rNNNN, and for the tie
design I3, one collection per N, e.g. I3_tie_r0001_N00050): its seed, the arm played in every round (the uniform
draws of the exploration and the greedy choices of the two tracks), the rewards, the exploration pilot, the CV choices,
the information rule's kappa0 and the split of every arm at every N.

replay(record) regenerates the collection from the seed -- every round's candidates and context, the row of the arm
played, its reward -- without refitting anything (the greedy choices are the recorded arms), and returns it as
algorithms.collect returns it.  Steps 2-4 (algorithms.fold_estimates, calibrate_loadings, two_track_inference) can then
be run again on the same data under any other choice (another n0, C_ref, target, rank, ...).

The command checks a run's records: for each one the replayed exploration arms equal the recorded ones, the rewards
drawn again from the seed equal the recorded ones (relative 1e-12), and every arm's recorded n0 is what the rules give on
the replayed data; with --rows, Steps 2-4 of every arm at the record's largest N reproduce the run's rows (mu, V and T,
relative 1e-6: the self-test's standard for simulated runs, so a replay on another machine than the run's passes).
"""
import argparse, csv, glob, gzip, os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
import simulation as S
import algorithms as A
import run_inference as R

ROWS_TOL = 1e-6          # Steps 2-4 against the run's rows (selftest.py: simulated runs to 1e-6 across machines)


def load(path):
    """A record as a dict of arrays."""
    z = np.load(path)
    return {k: z[k] for k in z.files}


def design_of(r):
    return S.TieDesign() if str(r["regime"]) == "tie" else S.make_design(str(r["block"]), str(r["regime"]))


def replay(r):
    """(D, rec, y_drawn): the design, the collection in the form of algorithms.collect's output (with the recorded
    rewards), and the rewards drawn again from the seed (to be compared with the recorded ones).  The random numbers
    are consumed in collect's order: the CV fold seed (when the constant is chosen by CV), then per round the
    candidates, the context, the exploration's arm (exploration only) and the reward."""
    D = design_of(r)
    rng = np.random.default_rng(int(r["seed"]))
    if str(r["c_lam"]) == "cv":
        rng.integers(2 ** 31)                                  # the fold seeds of the CV, drawn before the exploration
    tau, d, p0, K = int(r["tau"]), D.d, A.main_len(D), D.K
    n_tr = r["arm"].shape[1]

    def candidates():
        arms = D.draw_arms(rng)
        x = D.context(rng)
        return A.stack_rows(arms, x, p0)

    G, b = np.zeros((d, d)), np.zeros(d)
    Zex, reg_ex = np.zeros((tau, d)), np.zeros(tau)
    y_ex = np.zeros(tau)
    for t in range(tau):
        Zc = candidates()
        a = int(rng.integers(K))
        assert a == r["arm_ex"][t], "the exploration's arm of round %d differs from the record" % t
        z = Zc[a]
        y_ex[t] = D.reward(rng, z @ D.theta)
        mean = A.mean_reward(Zc @ D.theta, D.family)
        reg_ex[t] = mean.max() - mean[a]
        G += np.outer(z, z); b += r["y_ex"][t] * z               # as collect accumulates them
        Zex[t] = z
    Z = [np.zeros((n_tr, d)), np.zeros((n_tr, d))]
    reg, y = [np.zeros(n_tr), np.zeros(n_tr)], np.zeros((2, n_tr))
    for t in range(n_tr):
        for j in (0, 1):
            Zc = candidates()
            a = int(r["arm"][j, t])
            z = Zc[a]
            y[j, t] = D.reward(rng, z @ D.theta)
            mean = A.mean_reward(Zc @ D.theta, D.family)
            reg[j][t] = mean.max() - mean[a]
            Z[j][t] = z
    cv = None
    if str(r["c_lam"]) == "cv":
        cv = [[(int(s), float(c)) for s, c in zip(r["cv_s"], r["cv_c"][j])] for j in (0, 1)]
    rec = dict(Z=Z, Y=[r["y"][0].copy(), r["y"][1].copy()], G_ex=G, b_ex=b, Z_ex=Zex, Y_ex=r["y_ex"].copy(),
               tau=tau, pilot=r["pilot"].copy(), c_lam="cv" if cv is not None else float(r["c_lam"]), cv=cv,
               solver=dict(zip(("iters", "calls", "caps", "refresh"), (int(v) for v in r["solver"]))),
               score=float(r["score"]), horizon=int(r["horizon"]), regret_ex=reg_ex, regret=reg,
               arm_ex=r["arm_ex"].astype(np.int64), arm=[r["arm"][0].astype(np.int64), r["arm"][1].astype(np.int64)])
    return D, rec, (y_ex, y)


def stored_rows(run, r):
    """The run's rows of a record's replicate (and N for I3), keyed by (arm, N, target)."""
    block, regime, rep = str(r["block"]), str(r["regime"]), int(r["rep"])
    out = {}
    for path in glob.glob(os.path.join(run, "parts", "%s_%s_r*.tsv.gz" % (block, regime))):
        with gzip.open(path, "rt", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if int(row["rep"]) == rep:
                    out[(row["arm"], int(float(row["N"])), row["target"])] = row
    return out


def check(path, run, rows):
    """Replay one record and compare; returns the list of the differences found (empty: identical)."""
    r = load(path)
    D, rec, (y_ex, y) = replay(r)
    bad = []
    scale = lambda u: np.maximum(1.0, np.abs(u))
    if np.any(np.abs(y_ex - r["y_ex"]) > 1e-12 * scale(r["y_ex"])) or np.any(np.abs(y - r["y"]) > 1e-12 * scale(r["y"])):
        bad.append("rewards drawn from the seed differ from the record")
    arms = dict(S.arms(str(r["block"])))
    for i, N in enumerate(r["N"]):
        for k, arm in enumerate(r["arms"]):
            how = arms[str(arm)]
            n0 = A.n0_rule(D, rec, int(N))[0] if how == "rule" else R.split(how, int(N), D, rec)
            if n0 != int(r["n0"][i, k]):
                bad.append("n0 of %s at N = %d: %d, record %d" % (arm, N, n0, r["n0"][i, k]))
    if rows:
        N = int(r["N"][-1])
        V_CR, kappa_A = A.cramer_rao(D, rec, N, D.As)
        regret = A.collection_regret(rec, N)
        want = stored_rows(run, r)
        for arm, how in S.arms(str(r["block"])):
            rule = A.rank_rule(D, S.RANK_ARMS[arm]) if arm in S.RANK_ARMS else None
            for row in R.inference_rows(D, rec, N, how, rule, {}, V_CR, kappa_A, regret):
                w = want.get((arm, N, row["target"]))
                if w is None:
                    bad.append("no stored row for %s, N = %d, %s" % (arm, N, row["target"]))
                    continue
                for k in ("mu", "V", "T"):
                    if abs(row[k] - float(w[k])) > ROWS_TOL * max(1.0, abs(float(w[k]))):
                        bad.append("%s at N = %d, %s: %s %.12g, stored %s" % (arm, N, row["target"], k, row[k], w[k]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the run's directory (run_inference.py --out)")
    ap.add_argument("--only", default=None, help="one record, e.g. I1_lowrank_r0001")
    ap.add_argument("--max", type=int, default=None, help="at most this many records (the first in name order)")
    ap.add_argument("--rows", action="store_true", help="also recompute Steps 2-4 at the largest N and compare")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.run, "records", "*.npz")))
    if a.only:
        paths = [p for p in paths if os.path.basename(p) == a.only + ".npz"]
    paths = paths[:a.max] if a.max else paths
    if not paths:
        raise SystemExit("no record found in %s/records" % a.run)
    failed = 0
    for path in paths:
        bad = check(path, a.run, a.rows)
        failed += bool(bad)
        print("%s %s%s" % ("DIFFERS" if bad else "same   ", os.path.basename(path)[:-4],
                           "" if not bad else ": " + "; ".join(bad[:3])), flush=True)
    print("\n%d records replayed%s: %d identical, %d differ" % (len(paths), " with Steps 2-4" if a.rows else "",
                                                                 len(paths) - failed, failed))


if __name__ == "__main__":
    main()
