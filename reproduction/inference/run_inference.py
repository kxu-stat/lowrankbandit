#!/usr/bin/env python3
"""Run one task of the inference experiment: one (block, regime, replicate range), written as one gzip-compressed TSV
part, OUT/parts/KEY.tsv.gz, with one row per (replicate, grid point, arm, target).  A part
whose file exists with every replicate is skipped; a partial part is completed.

    python3 run_inference.py --chunk k --chunks N [--debug] [--out DIR]

The task list is simulation.tasks() (B = 1,000): per covariate law, I1 10 tasks of 100 replicates, I2l 500 tasks of
2 and I1d32 100 tasks of 10, and I3 2 tasks of 500 (1,222 tasks); --debug runs the small plan simulation.DEBUG_BLOCKS
(7 tasks, minutes each).  Per replicate: one collection (Step 1) to the block's largest n, then at every n of the grid
-- the same run stopped at n -- and for every arm of simulation.arms(block): Step 2 once, and Steps 3-4 for every
target of the design (four targets, D.As; I3 one).  An arm whose split and rank rule are an earlier arm's at the same n
gets that arm's rows (it would compute the same numbers), and the nuisance fits are kept by (fold, n0) for the
replicate's other n and arms.  The first target's row is two_track_inference's (every column); the others' rows carry
the target-independent columns of that row (SHARED) and calibrate_loadings' per-target columns.  Every row also
records the collection's cumulative expected regret at n (algorithms.collection_regret) and the target's Cramer-Rao
standard error for the collected design (algorithms.cramer_rao: V_CR, kappa_A; with the prediction
VCR_ratio_pred = sqrt(C_cal kappa_A log n_tr / kappa_hat) of V_pqv / V_CR).  Every row records c_tau, c_lam and the
version of the computation, CELL_VERSION; a part written under another version is never completed (write to a new
--out).  Every
row also carries two numbers of the exploration sample S_o: kappa0_ex, the information per round and coordinate that
the information rule uses (algorithms.kappa0_exploration), and pilot_rank, the numerical rank of the exploration
pilot's interaction block.

Records.  Every collection is also written as OUT/records/BLOCK_REGIME_rNNNN.npz (I3, one collection per N:
e.g. I3_tie_r0001_N00050.npz), before the replicate's rows: the seed, the arm played in every round, the rewards, the
exploration pilot, the CV choices, the information rule's kappa0 and the split of every arm at every n (the function
`record`).  From a record the collection is regenerated exactly, without refitting (replay_inference.py), so Steps
2-4 can be run again on the same data under any other choice; and the records are the intermediate data of the run
(a few hundred MB for the whole array).
"""
import argparse, csv, gzip, os, time
from fractions import Fraction
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
import simulation as S
import algorithms as A

PACKAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VERSION")).read().split()[0]
CELL_VERSION = "2.40"    # the version of the computation a part's rows record; it changes only with the computation
                         # (code that gives the same numbers up to rounding keeps it), so finished parts and records
                         # stay usable; the records also keep PACKAGE, the package that ran


def split(how, N, D, rec):
    """n0 of one arm on the first N rounds: None (the information rule, applied inside two_track_inference), the
    whole track, the design-stage split with cap c ("design c"), or the fixed share s of the track ("s"), floor(s n_tr)
    computed exactly (s as a Fraction)."""
    if how == "rule":
        return None
    if how == "whole track":
        return N // 2
    if how.startswith("design"):
        return A.design_split(D, rec, N, cap=Fraction(how.split()[1]))
    return int(Fraction(how) * (N // 2))


# The columns of two_track_inference's row that do not depend on the target (the collection, Step 2, the constants):
# the rows of the other targets carry them.
SHARED = ("N", "n0", "n0_frac", "kappa0", "c0", "tau", "err_pre", "err_post", "err_tan", "err_nor", "theta_err",
          "theta_relerr", "kap_hat", "kap_hat_0", "kap_hat_1", "C_cal", "lam_cal", "psi2_bar", "nuis_iters",
          "nuis_caps", "c_lam", "c_nuis_0", "c_nuis_1", "score", "lam_nuis", "lam_over_score", "col_iters",
          "col_calls", "col_caps", "r0", "rank_0", "rank_1", "lam_ref", "C_ref", "p_T", "sigma_hat",
          "kap_ex_min", "kap_ex_avg", "kap_gr_min", "kap_gr_avg", "kap_gr_cond", "res2_u")


def record(D, rec, meta, Ns):
    """The record of one collection (stored as one compressed .npz): what replays it from its seed (the arm played in
    every round and the rewards, the exploration pilot, the CV choices) and the numbers the analysis of the splits
    reads (kappa0, kappa_D, the exploration's responses, the pilot's rank, every arm's n0 at every N of Ns)."""
    assert D.K < 128
    arms = S.arms(meta["block"])
    cv = rec["cv"] or [[], []]
    return dict(block=meta["block"], regime=meta["regime"], rep=meta["rep"], seed=meta["seed"], package=CELL_VERSION,
                package_run=PACKAGE, numpy=np.__version__, tau=rec["tau"], horizon=rec["horizon"],
                c_lam=str(rec["c_lam"]),
                arm_ex=rec["arm_ex"].astype(np.int8), y_ex=rec["Y_ex"], arm=np.array(rec["arm"], dtype=np.int8),
                y=np.array(rec["Y"]), pilot=rec["pilot"], score=rec["score"],
                cv_s=np.array([s for s, _ in cv[0]], dtype=np.int64), cv_c=np.array([[c for _, c in path] for path in cv]),
                solver=np.array([rec["solver"][k] for k in ("iters", "calls", "caps", "refresh")]),
                kappa0=A.kappa0_exploration(D, rec), kappa_D=D.kappa_D, y_ex_sum=float(np.sum(rec["Y_ex"])),
                pilot_rank=A.pilot_rank(D, rec), N=np.array(Ns), arms=np.array([a for a, _ in arms]),
                n0=np.array([[A.n0_rule(D, rec, N)[0] if how == "rule" else split(how, N, D, rec) for _, how in arms]
                             for N in Ns]))


def write_record(value, path):
    tmp = path + ".partial.%d.npz" % os.getpid()
    np.savez_compressed(tmp, **value)
    os.replace(tmp, path)


def inference_rows(D, rec, N, how, rule, meta, V_CR, kappa_A, regret, cache=None):
    """The rows of one arm at one N: Step 2 once (fold_estimates, with the collection's `cache` of nuisance fits), then
    Steps 3-4 for every target of D (the first by two_track_inference, the others by calibrate_loadings with the
    comparator), each with its Cramer-Rao standard error and the collection's regret."""
    n0 = split(how, N, D, rec)
    F = A.fold_estimates(D, rec, N, A.n0_rule(D, rec, N)[0] if n0 is None else n0, rule, cache)
    o = A.two_track_inference(D, rec, N, A=D.As[0], n0=n0, rank_rule=rule, F=F)
    per = [o] + (A.calibrate_loadings(D, rec, N, D.As[1:], F, comparator=True) if len(D.As) > 1 else [])
    shared = {k: o[k] for k in SHARED}
    n_tr = N // 2
    rows = []
    for l, name in enumerate(D.target_names):
        row = dict(meta, target=name, **(o if l == 0 else {**shared, **per[l]}))
        row.update(PT_A=float(np.linalg.norm(S.tangent_project(D.As[l], D.Us, D.Vs, D.d1, D.d2))),
                   A_norm=float(np.linalg.norm(D.As[l])),                       # PT_A / A_norm is reported
                   V_CR=float(V_CR[l]), kappa_A=float(kappa_A[l]), VCR_ratio=float(row["V_pqv"] / V_CR[l]),
                   VCR_ratio_pred=float(np.sqrt(A.C_CAL * kappa_A[l] * np.log(max(n_tr, 3)) / row["kap_hat"])),
                   **regret)
        rows.append(row)
    return rows


def replicate(task, rep, blocks, c_tau, c_lam, records):
    block, regime = task["block"], task["regime"]
    if regime == "tie":                                   # I3: one collection per N
        rows = []
        for N in task["N"]:
            D = S.TieDesign()
            tau = S.TIE_TAU                               # the tie design's own exploration (the tau rule
            sd = S.seed(block, regime, rep, N)            # would give 2-3 rounds: too few for the pilot)
            rec = A.collect(D, tau, N, np.random.default_rng(sd), c_lam)
            ex = dict(kappa0_ex=A.kappa0_exploration(D, rec), pilot_rank=A.pilot_rank(D, rec))
            write_record(record(D, rec, dict(block=block, regime=regime, rep=rep, seed=sd), [N]),
                         os.path.join(records, "%s_%s_r%04d_N%05d.npz" % (block, regime, rep, N)))
            V_CR, kappa_A = A.cramer_rao(D, rec, N, D.As)
            regret = A.collection_regret(rec, N)
            for arm, how in S.arms(block):
                meta = dict(block=block, regime=regime, rep=rep, arm=arm, seed=sd, c_tau="", package=CELL_VERSION, **ex)
                rows += inference_rows(D, rec, N, how, None, meta, V_CR, kappa_A, regret)
        return rows
    D = S.make_design(block, regime, blocks)
    n_max = max(task["n"])                                # the horizon, in total rounds
    tau = A.tau_rule(n_max, D.d1, D.d2, c_tau)               # the regret experiment's tau rule
    t0 = time.time()
    sd = S.seed(block, regime, rep)
    rec = A.collect(D, tau, n_max - tau, np.random.default_rng(sd), c_lam)
    t_collect = time.time() - t0
    ex = dict(kappa0_ex=A.kappa0_exploration(D, rec), pilot_rank=A.pilot_rank(D, rec))
    write_record(record(D, rec, dict(block=block, regime=regime, rep=rep, seed=sd), [n - tau for n in task["n"]]),
                 os.path.join(records, "%s_%s_r%04d.npz" % (block, regime, rep)))
    rows, cache = [], {}                                  # cache: the nuisance fits by (fold, n0)
    for n in task["n"]:                                   # the same run, stopped at n
        N = n - tau
        V_CR, kappa_A = A.cramer_rao(D, rec, N, D.As)     # the design's, for every target
        regret = A.collection_regret(rec, N)
        same = {}                                         # (n0, rank rule) -> the rows of the first such arm
        for arm, how in S.arms(block):
            rule = A.rank_rule(D, S.RANK_ARMS[arm]) if arm in S.RANK_ARMS else None
            meta = dict(block=block, regime=regime, rep=rep, arm=arm, n=n, nb=D.nb, seed=sd, c_tau=c_tau,
                        package=CELL_VERSION, t_collect=round(t_collect, 1), **ex)
            key = (split(how, N, D, rec), S.RANK_ARMS.get(arm))
            if key[0] is not None and key in same:        # the same split (not the information rule): the same rows
                rows += [dict(row, arm=arm) for row in same[key]]
            else:
                same[key] = inference_rows(D, rec, N, how, rule, meta, V_CR, kappa_A, regret, cache)
                rows += same[key]
    return rows


def provenance():
    """The versions a replicate's numbers depend on, stored in every row."""
    import platform, scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                node=os.uname().nodename)


def write_tsv(rows, path):
    """The part, gzip-compressed, written whole and then renamed, so a stopped task never leaves half a file."""
    keys = list(rows[0].keys())
    tmp = path + ".partial.%d" % os.getpid()
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="replay one task by key, e.g. I1_lowrank_r0001-0100")
    a = ap.parse_args()
    blocks = S.DEBUG_BLOCKS if a.debug else S.BLOCKS
    out = a.out or ("inference_debug" if a.debug else "inference_results")
    all_tasks = S.tasks(blocks)
    assert 1 <= a.chunk <= a.chunks
    mine = [t for t in all_tasks if t["key"] == a.only] if a.only else all_tasks[a.chunk - 1::a.chunks]
    if a.only and not mine:
        raise SystemExit("no task with key %s" % a.only)
    os.makedirs(os.path.join(out, "parts"), exist_ok=True)
    records = os.path.join(out, "records")
    os.makedirs(records, exist_ok=True)
    print("run_inference: tasks %d of %d -> %s (C_ref 1/%g, C_cal %g, c0 1/%g; package %s, computation %s)" % (
        len(mine), len(all_tasks), out, 1 / A.C_REF, A.C_CAL, 1 / A.C0, PACKAGE, CELL_VERSION), flush=True)
    for task in mine:
        c_tau, c_lam, source = S.constants(task["block"])
        print("%s: c_tau %s, c_lam %s (%s)" % (task["key"], c_tau, c_lam, source), flush=True)
        path = os.path.join(out, "parts", task["key"] + ".tsv.gz")
        if os.path.exists(path[:-3]):               # an uncompressed part of this key: compress, not recompute
            raise SystemExit("%s exists: compress the parts first, gzip %s/parts/*.tsv" % (path[:-3], out))
        rows, done = [], set()
        if os.path.exists(path):
            with gzip.open(path, "rt", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            done = {int(r["rep"]) for r in rows}
            if rows and rows[0].get("package") != CELL_VERSION:
                raise SystemExit("%s was written under another computation version: write to a new --out" % path)
        for rep in range(task["lo"], task["hi"] + 1):
            if rep in done:
                continue
            t0 = time.time()
            rows += [dict(r, **provenance()) for r in replicate(task, rep, blocks, c_tau, c_lam, records)]
            write_tsv(rows, path)                                      # after every replicate
            print("%s rep %d done (%.0fs)" % (task["key"], rep, time.time() - t0), flush=True)
        print("OK %s: %d replicates" % (task["key"], task["hi"] - task["lo"] + 1), flush=True)


if __name__ == "__main__":
    main()
