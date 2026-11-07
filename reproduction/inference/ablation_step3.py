#!/usr/bin/env python3
"""The ablation of Step 3 on the records of an inference run (by hand or submit_ablation.sge):
Algorithm 2 against the one-step debiasing of the policy's own estimator,
with and without cross-fitting, on the same collections.

    python3 ablation_step3.py RUN_DIR OUT_DIR [--chunk k --chunks m] [--every-n]
    e.g.  python3 ablation_step3.py inference_results ablation_results/parts --chunk 1 --chunks 20

RUN_DIR is the run's directory (run_inference.py --out).  Its records/*.npz are replayed (replay_inference.replay: the
same collection, nothing refitted).  For every record of the blocks in PLAN (each in its displayed covariate regime)
and every n of PLAN (--every-n: every n of the record, for the debug run), two Step 2's are run on track j's data, as
the run ran them:
    design   design_75's n0: the nuisance fit, truncation and predictable refit of Algorithm 2;
    nuc      n0 = n_tr: the policy's nuclear-norm estimator on S_o and the whole track, at the track's penalty level
             (its CV constant in force), truncated to rank r0 -- the run's no_refit arm, with no refit.
Then Step 3 in one of three weights, over the opposite track (cross-fitted) or over track j itself:
    predictable  Algorithm 2's calibration recursion;
    tangent      one step with the inverse empirical Hessian of the whole calibration track on the pilot's tangent
                 space, omega_t = q_t' (sum_s psi''_s q_s q_s')^{-1} Q'A, q_t = Q'z_t (debias and project);
    full         one step with the inverse empirical Hessian on all d coordinates (nan when n_tr < d).

    arm            pilot   cross-fitted  weights
    ours           design  yes           predictable   Algorithm 2
    nocf           design  no            predictable
    onestep        design  yes           tangent
    onestep_full   design  yes           full          the package's comparator
    no_refit       nuc     yes           predictable   the run's no_refit arm
    nuc_tan_nocf   nuc     no            tangent       whole-sample debiasing of the policy's estimator
    nuc_tan        nuc     yes           tangent       double-sample (cross-fitted) debiasing of it

Every arm: mu^(j) = <A, theta_hat^(j)> + sum_t omega_t (y_t - psi'(z_t' theta_hat^(j))), V_j^2 = sum_t omega_t^2 (y_t -
psi'(.))^2, mu_hat = (mu^(0) + mu^(1)) / 2, V = sqrt(V_0^2 + V_1^2) / 2.  Every row also carries noise = (1/2) sum_j
sum_t omega_t eps_t, eps_t = y_t - psi'(z_t' theta*) the true noise of the round (known in a simulation): with the
predictable weights its mean is zero whatever the collecting policy (a martingale); with the one-step weights its
mean is the bias the adaptive collection causes.  mu_hat - mu* - noise is the remainder (pilot and nonlinearity),
which every arm carries.  In the tie design I3 (d1 = d2 = r0 = 1, identity link) every one-step arm is the average of
the two tracks' least squares: they coincide.

Checks (the run stops at the first failure): the rewards drawn again from the seed equal the record's; design_75's n0
on the replay equals the recorded one; every n of PLAN is in the record; every cross-fitted predictable arm equals the
package's calibrate_loadings on the same Step 2, and onestep_full its comparator (relative 1e-9).
figure_ablation.py --run checks ours and no_refit against the run's own rows (design_75 and no_refit).

Output: OUT_DIR/part_K_of_M.tsv, one row per (record, n, target, arm), stamped with the package version.  A record
already in the part is skipped, so a stopped task is completed by running it again; a part written by another version
is never completed (write to a new OUT_DIR).
"""
import argparse, csv, glob, os, time
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np
import simulation as S
import algorithms as A
import run_inference as R
import replay_inference as P

PLAN = {  # block: (regime, the n of the ablation -- total rounds; I3 post-exploration rounds N, None = every N)
    "I3":    ("tie",     None),
    "I1":    ("lowrank", (500, 1000, 2000, 5000, 10000)),
    "I1d32": ("lowrank", (1000, 2000, 4000, 8000, 10000)),
    "I2l":   ("lowrank", (1000, 2000, 5000, 10000, 20000, 40000, 60000)),
}
ARMS = (  # arm, pilot, cross-fitted, weights
    ("ours", "design", True, "predictable"), ("nocf", "design", False, "predictable"),
    ("onestep", "design", True, "tangent"), ("onestep_full", "design", True, "full"),
    ("no_refit", "nuc", True, "predictable"), ("nuc_tan_nocf", "nuc", False, "tangent"), ("nuc_tan", "nuc", True, "tangent"))
COLS = ("record", "block", "regime", "rep", "n", "N", "n0", "target", "arm", "mu", "V", "noise", "mu_star", "package")


def step3(D, rec, N, As, F, cross, weights):
    """Steps 3-4 for every row of As in one arm, given Step 2's F: (mu_hat, V, noise), one entry per loading."""
    d1, d2, fam, nb = D.d1, D.d2, D.family, D.nb
    L, n_tr = len(As), N // 2
    mus, V2, noise = np.zeros((2, L)), np.zeros((2, L)), np.zeros((2, L))
    for j in (0, 1):
        k = 1 - j if cross else j                                    # the track the calibration runs over
        Z, Y = rec["Z"][k][:n_tr], rec["Y"][k][:n_tr]
        Uh, Vh = F["frames"][j]
        th = F["th_hat"][j]
        uh = Z @ th
        cc = A.psi2(uh, fam, nb) if fam != "gaussian" else np.ones(n_tr)
        if weights == "predictable":                                 # the calibration recursion of calibrate_loadings
            PZ = A.tangent_project_rows(Z, Uh, Vh, d1, d2)
            qq = np.einsum("ij,ij->i", PZ, PZ)
            rho = np.ascontiguousarray(A.tangent_project_rows(As, Uh, Vh, d1, d2).T)
            lam = n_tr / (2.0 * A.C_CAL * np.log(max(n_tr, 3))) * F["kap"][j]
            om = np.zeros((n_tr, L))
            for t in range(n_tr):
                w = (Z[t] @ rho) / (lam + cc[t] * qq[t])
                rho -= cc[t] * np.outer(PZ[t], w)
                om[t] = w
        else:                                                        # one step with the whole track's Hessian
            try:
                if weights == "tangent":
                    Q = A.tangent_basis(Uh, Vh, d1, d2, F["ranks"][j], A.main_len(D))
                    q = Z @ Q
                    om = q @ np.linalg.solve(q.T @ (q * cc[:, None]), Q.T @ As.T)
                else:                                                # as calibrate_loadings' comparator
                    if n_tr < Z.shape[1]:
                        raise np.linalg.LinAlgError
                    om = Z @ np.linalg.solve(Z.T @ (Z * cc[:, None]) / n_tr, As.T) / n_tr
            except np.linalg.LinAlgError:
                mus[j] = V2[j] = noise[j] = np.nan
                continue
        res = Y - A.psi1(uh, fam, nb)
        mus[j] = As @ th + res @ om
        V2[j] = (res ** 2) @ (om ** 2)
        noise[j] = (Y - A.psi1(Z @ D.theta, fam, nb)) @ om
    return 0.5 * (mus[0] + mus[1]), 0.5 * np.sqrt(V2[0] + V2[1]), 0.5 * (noise[0] + noise[1])


def arm_rows(D, rec, N, n0, base):
    """The seven arms on one collection at N post-exploration rounds, with the two checks against the package."""
    scale = lambda u: np.maximum(1.0, np.abs(u))
    F = {"design": A.fold_estimates(D, rec, N, n0), "nuc": A.fold_estimates(D, rec, N, N // 2)}
    pkg = {p: A.calibrate_loadings(D, rec, N, D.As, F[p], comparator=(p == "design")) for p in F}
    rows = []
    for arm, pilot, cross, weights in ARMS:
        mu, V, noise = step3(D, rec, N, D.As, F[pilot], cross, weights)
        ref = None                                                   # the package's numbers on the same Step 2
        if cross and weights == "predictable":
            ref = np.array([o["mu"] for o in pkg[pilot]])
        elif pilot == "design" and weights == "full":
            ref = np.array([o["mu_ols"] for o in pkg["design"]])
        if ref is not None:
            same = (np.abs(mu - ref) <= 1e-9 * scale(ref)) | (np.isnan(mu) & np.isnan(ref))
            assert np.all(same), "%s, N = %d: %s differs from the package" % (base["record"], N, arm)
        rows += [dict(base, target=t, arm=arm, mu=mu[l], V=V[l], noise=noise[l], mu_star=pkg["design"][l]["mu_star"],
                      package=R.PACKAGE) for l, t in enumerate(D.target_names)]
    return rows


def record_rows(path, every_n):
    r = P.load(path)
    D, rec, (y_ex, y) = P.replay(r)
    scale = lambda u: np.maximum(1.0, np.abs(u))
    assert np.all(np.abs(y_ex - r["y_ex"]) <= 1e-12 * scale(r["y_ex"])) and \
        np.all(np.abs(y - r["y"]) <= 1e-12 * scale(r["y"])), "%s: the rewards drawn from the seed differ" % path
    block, regime, rep, tau = str(r["block"]), str(r["regime"]), int(r["rep"]), int(r["tau"])
    ns = None if every_n else PLAN[block][1]
    k = [str(a) for a in r["arms"]].index(S.PRIMARY)
    how = dict(S.arms(block))[S.PRIMARY]
    name = os.path.basename(path)[:-4]
    rows = []
    for i, N in enumerate(int(v) for v in r["N"]):
        n = N if block == "I3" else tau + N
        if ns is not None and n not in ns:
            continue
        n0 = R.split(how, N, D, rec)
        assert n0 == int(r["n0"][i, k]), "%s: n0 at N = %d is %d, recorded %d" % (name, N, n0, r["n0"][i, k])
        rows += arm_rows(D, rec, N, n0, dict(record=name, block=block, regime=regime, rep=rep, n=n, N=N, n0=n0))
    assert ns is None or {row["n"] for row in rows} == set(ns), "%s: an n of the plan is not in the record" % name
    return rows


def done_in(part):
    """The records (or replicates) already in a part; a part written by another package version stops the run."""
    if not os.path.exists(part):
        return []
    rows = list(csv.DictReader(open(part), delimiter="\t"))
    other = {row.get("package") for row in rows} - {R.PACKAGE}
    if other:
        raise SystemExit("%s was written by package %s, this is %s: write to a new OUT_DIR" %
                         (part, ", ".join(sorted(str(v) for v in other)), R.PACKAGE))
    return rows


def append(part, rows):
    """Append one record's (or replicate's) rows whole."""
    new = not os.path.exists(part)
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the run's directory (run_inference.py --out)")
    ap.add_argument("out", help="the directory of the ablation's parts")
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--every-n", action="store_true", help="every n of each record (the debug run)")
    a = ap.parse_args()
    paths = sorted(p for block, (regime, _) in PLAN.items()
                   for p in glob.glob(os.path.join(a.run, "records", "%s_%s_r*.npz" % (block, regime))))
    if not paths:
        raise SystemExit("no record of the plan in %s/records" % a.run)
    mine = paths[a.chunk - 1::a.chunks]
    os.makedirs(a.out, exist_ok=True)
    part = os.path.join(a.out, "part_%03d_of_%03d.tsv" % (a.chunk, a.chunks))
    done = {row["record"] for row in done_in(part)}
    print("ablation_step3: %d of %d records -> %s (%d done; package %s)" % (len(mine), len(paths), part, len(done),
                                                                          R.PACKAGE), flush=True)
    for path in mine:
        name = os.path.basename(path)[:-4]
        if name in done:
            continue
        t0 = time.time()
        append(part, record_rows(path, a.every_n))
        print("%s (%.0fs)" % (name, time.time() - t0), flush=True)
    print("OK %s" % part, flush=True)


if __name__ == "__main__":
    main()
