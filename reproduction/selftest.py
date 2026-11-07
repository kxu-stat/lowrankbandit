#!/usr/bin/env python3
"""Self-test: does this installation compute what the authors' installation computed?

    python3 selftest.py                     # the install gate: about a minute, no scheduler
    python3 selftest.py --cells DIR         # check produced cells: values, errors, interpreter
    python3 selftest.py --write-reference   # regenerate the check values (authors only)
    python3 selftest.py --write-cells DIR   # record a run directory's cells as reference values

Every check is a number this package produces from a fixed seed, compared with the value
recorded in `selftest_reference.json`: the design constants exactly, the estimator to 1e-9,
the simulated runs to 1e-6.  A FAIL at 1e-6 is not noise: the checks are single
trajectories, and because a bandit chooses by argmax a last-bit difference can flip one
action and move a regret by a percent (measured: swapping one LAPACK routine for another
that solves the same system moved one of thirty baseline configurations by 1.2%).  So the
tolerance is there for the last-bit agreement of identical routines, and anything larger
means this machine's linear algebra differs from the authors' in a way that changes
results; send the output rather than starting a run.

The checks cover the design (case grids, seeds, exploration lengths, the tiers approx and d32), the estimator (FISTA
on a rank-deficient Gram), Algorithm 1 under both links with the penalty constant fixed and by cross-validation, both
baselines (ESTR with the authors' BM controls and its stage-1 subspace error), and Algorithm 2 under the identity and
logit links (the known rank and the appendix's rank rules, the design-stage split, several loadings at once, the four
targets, the collection's regret and the Cramer-Rao standard error, a collection's record and its replay from the
seed), i.e. every numerical path the experiment uses.
`--cells` additionally checks cells the arrays produced against the reference values of the debug cells recorded by the
authors (the regret debug arrays), and checks that the compute nodes used the interpreter that passed this gate on the
front end.
"""
import argparse, glob, json, os, sys, time
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "selftest_reference.json")
TOL = {"design constants": 1e-12, "estimator": 1e-9}     # every other check: 1e-6


def case(family, case_id):
    """A case of the package's own table (so its noise sd and link scale are the design's)."""
    sys.path.insert(0, os.path.join(HERE, "regret"))
    import simulation as RS
    return next(c for c in RS.cases(family, ("main",)) if c["case_id"] == case_id)


def gaussian_case():
    return case("gaussian", "sph_d16_r2_K16_rho020")


def binomial_case():
    return case("binomial", "sph_d16_r2_K16_nb01")


def checks():
    """(name, thunk) for every check; each thunk returns a dict of numbers."""
    sys.path.insert(0, os.path.join(HERE, "regret"))
    import simulation as RS, algorithms as RA, baselines as RB, run_regret as RT
    HALF, DOUBLE = RS.C_TAU_RUN[0], RS.C_TAU_RUN[-1]       # the sensitivity values of c_tau

    def tail_energy(kind):                                  # the truth's energy outside rank 2
        return float(np.sum(np.linalg.svd(RS.truth(16, 16, 2, kind), compute_uv=False)[2:] ** 2))

    def design():
        return dict(n_cases_gaussian=len(RS.cases("gaussian")), n_cases_binomial=len(RS.cases("binomial")),
                    case_index=RS.case_index("gaussian", 16, 2, 0.2, "gaussian"),
                    seed_eval=RS.environment_seed(10000, 38, 50, "evaluation", 64),
                    solver_seed=RS.solver_seeds(38, RS.FIRST_REPLICATION, RS.PHASE, 16)[0],
                    link_scale_sphere=RS.link_scale("sphere", 16, 2, 16),
                    case_index_normal=RS.case_index("normal", 16, 2, 1.0, "gaussian"),
                    link_scale_normal=RS.link_scale("normal", 16, 1, 16),
                    p_bar=RS.P_BAR,
                    tau_A_half=RA.tau_rule(10000, 16, 16, HALF),
                    tau_A=RA.tau_rule(10000, 16, 16, RS.C_TAU),
                    tau_A_double=RA.tau_rule(10000, 16, 16, DOUBLE),
                    n_arms=len(RS.ARMS), n_arms_pm1=len(RS.PM1_ARMS),
                    estr_tau_T1=RT.baseline_specs(gaussian_case(), "estr_tau", 10000)[0]["T1"],   # = tau_A
                    n_tasks_gaussian=len(RT.tasks("gaussian")), n_tasks_binomial=len(RT.tasks("binomial")),
                    time_grid=len(RS.time_grid()),
                    case_index_d16_r4=RS.case_index("sphere", 16, 4, 0.2, "gaussian"),
                    first_replication=RS.FIRST_REPLICATION,
                    seed_first=RS.environment_seed(10000, 38, RS.FIRST_REPLICATION, RS.PHASE, 16),
                    # the tiers approx (the spectra j^-2 and j^-1.5) and d32
                    n_tasks_gaussian_approx=len(RT.tasks("gaussian", ["approx"])),
                    n_tasks_binomial_approx=len(RT.tasks("binomial", ["approx"])),
                    n_tasks_gaussian_d32=len(RT.tasks("gaussian", ["d32"])),
                    tail_energy_decay2=tail_energy("decay2"), tail_energy_decay15=tail_energy("decay1.5"),
                    link_scale_decay2=RS.link_scale("sphere", 16, 2, 16, kind="decay2"),
                    link_scale_decay15=RS.link_scale("sphere", 16, 2, 16, kind="decay1.5"),
                    seed_approx=RS.environment_seed(10000, 38, RS.FIRST_REPLICATION, "approx", 16),
                    cv_first_checkpoint=RA.cv_checkpoints(24, 10000)[0], cv_folds=RA.CV_FOLDS)

    def estimator():
        """FISTA on a rank-deficient Gram (20 rows, 36 coefficients), and fit_nuclear on the same record,
        which returns the solver's minimizer, so the two must coincide."""
        rng = np.random.default_rng(7)
        d, s = 6, 20
        M = rng.standard_normal((s, d)); M /= np.linalg.norm(M, axis=1, keepdims=True)
        X = rng.standard_normal((s, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
        rec = RA.record_from_rows(M, X, 0.1 * rng.standard_normal(s))
        raw, iters, _ = RA.fista(rec.G / s, rec.cross / s, rec.y2 / s, 0.01, 2.0, np.zeros((d, d)), 20000, 1e-7)
        fit = RA.fit_nuclear(rec, 0.01, 2.0, np.zeros((d, d)), "gaussian", 1)
        return dict(fista_iters=iters, raw_frobenius=float(np.linalg.norm(raw)),
                    fit_frobenius=float(np.linalg.norm(fit["Theta"])),
                    fit_nuclear=float(RA.nuclear_norm(fit["Theta"])),
                    fit_equals_fista=int(np.allclose(fit["Theta"], raw, rtol=0, atol=1e-12)))

    def algorithm1_gaussian():
        """A fixed penalty constant, and the constant by 10-fold CV (checkpoints max(2 tau, 200) = 200 and 400 here, so
        the CV run has 450 rounds)."""
        env = RS.Environment(60, gaussian_case(), 12345)
        out = RA.run_algorithm1(env, 9, 1 / 16)
        cv = RA.run_algorithm1(RS.Environment(450, gaussian_case(), 12345), 9, "cv")
        return dict(regret=float(out["trace"]["regret"].sum()), score=float(out["score"]),
                    rank_n=int(out["trace"]["fitted_rank"][-1]), cv_regret=float(cv["trace"]["regret"].sum()),
                    cv_first=cv["cv"][0][1], cv_last=cv["cv"][-1][1], cv_checks=len(cv["cv"]))

    def algorithm1_binomial():
        env = RS.Environment(30, binomial_case(), 999)
        out = RA.run_algorithm1(env, 9, 1 / 16)
        cv = RA.run_algorithm1(RS.Environment(240, binomial_case(), 999), 9, "cv")     # first check at 200
        return dict(regret=float(out["trace"]["regret"].sum()), score=float(out["score"]),
                    cv_regret=float(cv["trace"]["regret"].sum()), cv_first=cv["cv"][0][1])

    def baselines():
        """150 rounds: ESTR's shortest T1 is 60 at (16, 2), so its stage 2 runs 90 rounds."""
        env = RS.Environment(150, gaussian_case(), 12345)
        c = gaussian_case()
        specs = {s["config_id"]: s for s in RB.specs(16, 2, 16, c["noise_sd"], c["amp"], 150, arm_scale=c["arm_scale"])}
        oful = RB.run_baseline(env, specs["OFUL_g1_m02"], None)
        estr = RB.run_baseline(env, specs["ESTR_BM_g1_t00_m02"], 777)
        b = binomial_case()                                  # binary rewards, the authors' treatment (2y - 1)
        env_b = RS.Environment(150, b, 12345)
        specs_b = {s["config_id"]: s for s in RB.specs(16, 2, 16, b["noise_sd"], b["amp"], 150, coding="pm1",
                                                       arm_scale=b["arm_scale"])}
        oful_b = RB.run_baseline(env_b, specs_b["OFUL_g1_m02"], None)
        estr_b = RB.run_baseline(env_b, specs_b["ESTR_BM_g1_t00_m02"], 777)
        specs_g = {s["config_id"]: s for s in RB.specs(16, 2, 16, b["noise_sd"], b["amp"], 150,
                                                       arm_scale=b["arm_scale"])}   # our GLM OFUL and GLM-ESTR
        oful_g = RB.run_baseline(env_b, specs_g["OFUL_g1_m02"], None)
        estr_g = RB.run_baseline(env_b, specs_g["ESTR_BM_g1_t00_m02"], 777)     # our GLM-ESTR
        return dict(oful_regret=float(oful["regret"].sum()), estr_regret=float(estr["regret"].sum()),
                    estr_T1=int(specs["ESTR_BM_g1_t00_m02"]["T1"]), estr_starts=RB.BM["starts"],
                    estr_maxIter=RB.BM["maxIter"], estr_bm_sin=estr["bm"]["bm_sin"], glm_estr_starts=RB.GLM_BM["starts"],
                    estr_best_start=estr["bm"]["bm_best_start"], estr_best_start_binary=estr_b["bm"]["bm_best_start"],
                    oful_regret_binary=float(oful_b["regret"].sum()), estr_regret_binary=float(estr_b["regret"].sum()),
                    oful_glm_regret_binary=float(oful_g["regret"].sum()),
                    estr_glm_regret_binary=float(estr_g["regret"].sum()),
                    estr_glm_best_start=estr_g["bm"]["bm_best_start"])

    def algorithm2():
        sys.path.insert(0, os.path.join(HERE, "inference"))
        for m in ("simulation", "algorithms"):
            sys.modules.pop(m, None)
        import simulation as IS, algorithms as IA
        out, collections = {}, []
        for label, D, N in (("tie", IS.TieDesign(), 200),
                            ("identity", IS.Design("normal", "gaussian", 1), 200),
                            ("logistic", IS.Design("lowrank", "binomial", 1), 120)):
            tau = 12 if label == "tie" else IA.tau_rule(N, D.d1, D.d2, 1 / 20)   # tie: its fixed exploration
            rec = IA.collect(D, tau, N, np.random.default_rng(4242), 1 / 16)
            collections.append((dict(tie="I3", identity="I1", logistic="I2l")[label], D, rec, N, 4242))
            o = IA.two_track_inference(D, rec, N)                 # n0 from the information rule
            out.update({label + "_tau": tau, label + "_mu": o["mu"], label + "_V": o["V"], label + "_T": o["T"],
                        label + "_n0": o["n0"], label + "_kappa0": o["kappa0"]})
            if label != "tie":                                    # the rank arms of the appendix
                for kind in ("gic", "aic"):
                    o = IA.two_track_inference(D, rec, N, rank_rule=IA.rank_rule(D, kind))
                    out.update({"%s_%s_mu" % (label, kind): o["mu"], "%s_%s_ranks" % (label, kind): 10 * o["rank_0"] + o["rank_1"]})
                for cap in (0.75, 0.9):                           # the design-stage split
                    out["%s_design_%d" % (label, round(100 * cap))] = IA.design_split(D, rec, N, cap)
                N_big = dict(identity=2000, logistic=10000)[label]  # the information rule where it is neither
                out["%s_rule_n0_N%d" % (label, N_big)] = IA.n0_rule(D, rec, N_big)[0]   # floor nor cap (S_o only)
                n0 = (N // 2) // 4                                # Steps 3-4 for two loadings at once ...
                As = np.vstack([D.A, np.random.default_rng(5).standard_normal(D.d)])
                many = IA.calibrate_loadings(D, rec, N, As, IA.fold_estimates(D, rec, N, n0))
                one = [IA.two_track_inference(D, rec, N, A=A, n0=n0) for A in As]   # ... and one at a time
                out.update({label + "_many_mu": many[1]["mu"], label + "_many_V": many[1]["V"],
                            label + "_many_equals_one": int(all(abs(m[k] - o[k]) <= 1e-12 * max(1.0, abs(o[k]))
                                                                for m, o in zip(many, one) for k in ("mu", "V", "V_pqv")))})
        D = IS.Design("lowrank", "gaussian", 1)                  # the collection with the constant by CV
        rec = IA.collect(D, IA.tau_rule(400, D.d1, D.d2, 1 / 20), 400, np.random.default_rng(4243), "cv")
        collections.append(("I1", D, rec, 400, 4243))
        o = IA.two_track_inference(D, rec, 400)
        out.update(cv_tau=rec["tau"], cv_choices=float(sum(c for path in rec["cv"] for _, c in path)), cv_mu=o["mu"],
                   cv_T=o["T"])
        # the four targets (their tangent norms and mu*), Steps 3-4 for all of them on one Step 2, the collection's
        # regret and the Cramer-Rao standard errors
        T_norm = [float(np.linalg.norm(IS.tangent_project(A, D.Us, D.Vs, 16, 16))) for A in D.As]
        F = IA.fold_estimates(D, rec, 400, IA.n0_rule(D, rec, 400)[0])
        rows = IA.calibrate_loadings(D, rec, 400, D.As, F, comparator=True)
        V_CR, kappa_A = IA.cramer_rao(D, rec, 400, D.As)
        reg = IA.collection_regret(rec, 400)
        out.update(targets_tangent_norms=float(sum(T_norm)), targets_mu_star=float(np.sum(np.abs(D.mus))),
                   targets_mu=float(sum(r["mu"] for r in rows)), targets_V=float(sum(r["V"] for r in rows)),
                   dense_equals_two_track=int(abs(rows[0]["mu"] - o["mu"]) <= 1e-12 * max(1.0, abs(o["mu"]))),
                   V_CR_sum=float(np.sum(V_CR)), kappa_A_sum=float(np.sum(kappa_A)), regret_two_track=reg["regret"])
        U32, _, V32t = np.linalg.svd(IS.truth(32, 32, 2)[0])                # the d = 32 block's targets
        out["d32_targets_tangent_norms"] = float(sum(np.linalg.norm(IS.tangent_project(A, U32[:, :2], V32t[:2].T, 32, 32))
                                                     for A in IS.targets(32, 32, 2)))
        # each collection above written as the run's record and replayed from its seed (inference/
        # replay_inference.py): the same rows, Gram, rewards and regrets, and every arm's n0 recomputed equal
        import tempfile
        import run_inference as IR, replay_inference as IP
        same = 1
        with tempfile.TemporaryDirectory() as tmp:
            for block, D, rec, N, sd in collections:
                path = os.path.join(tmp, "record.npz")
                IR.write_record(IR.record(D, rec, dict(block=block, regime=D.regime, rep=1, seed=sd), [N]), path)
                _, rp, _ = IP.replay(IP.load(path))
                same &= int(not IP.check(path, tmp, False)
                            and all(np.array_equal(rp[k], rec[k]) for k in ("Z_ex", "Y_ex", "G_ex", "b_ex", "regret_ex"))
                            and all(np.array_equal(rp[k][j], rec[k][j]) for k in ("Z", "Y", "regret") for j in (0, 1)))
        out["records_replay_identical"] = same
        for m in ("simulation", "algorithms", "run_inference", "replay_inference"):
            sys.modules.pop(m, None)
        return out

    return [("design constants", design), ("estimator", estimator),
            ("Algorithm 1, identity link", algorithm1_gaussian), ("Algorithm 1, logit link", algorithm1_binomial),
            ("baselines OFUL and ESTR", baselines), ("Algorithm 2, three designs", algorithm2)]


def versions():
    import platform, scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def compare(got, want, tol):
    """The names that differ by more than a relative `tol`, and the names this package computes that the reference
    does not hold (a reference from another package)."""
    bad = []
    for k, v in want.items():
        if k not in got:
            bad.append("%s missing" % k)
        elif isinstance(v, (int, float)) and abs(got[k] - v) > tol * max(1.0, abs(v)):
            bad.append("%s: got %.12g, reference %.12g" % (k, got[k], v))
    bad += ["%s: not in the reference" % k for k in got if k not in want]
    return bad


def run(write_reference):
    reference = json.load(open(REFERENCE)) if os.path.exists(REFERENCE) else {}
    print("versions here:     " + ", ".join("%s %s" % kv for kv in versions().items()))
    if not write_reference:
        print("reference from:    " + ", ".join("%s %s" % kv for kv in reference["versions"].items()))
    print()
    failures, out = 0, {}
    for name, thunk in checks():
        t0 = time.time()
        got = thunk()
        out[name] = got
        if write_reference:
            print("  recorded  %-28s (%4.0f s)  %s" % (name, time.time() - t0, list(got)[:3]))
            continue
        bad = compare(got, reference["checks"].get(name, {}), TOL.get(name, 1e-6))
        failures += len(bad) > 0
        print("  %-6s %-28s (%4.0f s)%s" % ("FAIL" if bad else "PASS", name, time.time() - t0,
                                            "" if not bad else "\n           " + "\n           ".join(bad)))
    if write_reference:
        json.dump(dict(versions=versions(), checks=out, cells=reference.get("cells", {})),
                  open(REFERENCE, "w"), indent=1, sort_keys=True)
        print("\nreference written:", REFERENCE)
        return 0
    print("\n%s" % ("ALL CHECKS PASS: this installation reproduces the authors' numbers."
                    if not failures else
                    "%d CHECK(S) FAILED: do not start a run; send this output." % failures))
    return 1 if failures else 0


def cell_key(path, z):
    """The reference key of a cell: its file name and its horizon (a debug cell and the production cell of the same
    case and replication share the file name, not the horizon)."""
    return "%s@n%s" % (os.path.basename(path)[:-5], z["task"].get("horizon"))


def write_cells(directory):
    """Record every finished cell of a run directory as a reference value (authors only).
    The cells block is keyed by `cell_key` and holds {config_id: regret}; `--cells`
    then checks exactly these cells and ignores the rest of the run."""
    reference = json.load(open(REFERENCE))
    cells = dict(reference["cells"])
    for path in sorted(glob.glob(os.path.join(directory, "cells", "*.json"))):
        z = json.load(open(path))
        if z["status"] != "ok":
            print("  skip (error) %s" % os.path.basename(path)[:-5])
            continue
        key = cell_key(path, z)
        cells[key] = {r["config_id"]: r["regret"] for r in z["rows"]}
        print("  recorded %-52s %d configuration(s)" % (key, len(cells[key])))
    reference["cells"] = cells
    json.dump(reference, open(REFERENCE, "w"), indent=1, sort_keys=True)
    print("\nreference written: %s (%d cell(s))" % (REFERENCE, len(cells)))
    return 0


def check_cells(directory):
    """Check produced cells: every one finished, the ones with a recorded reference value
    match it, and the compute nodes used the interpreter running this check (the one
    that passed the self-test on the front end)."""
    reference, here = json.load(open(REFERENCE))["cells"], versions()
    files = sorted(glob.glob(os.path.join(directory, "cells", "*.json")))
    if not files:
        print("no cells in %s/cells" % directory)
        return 1
    failures, errors, checked, envs = 0, 0, 0, {}
    for path in files:
        z = json.load(open(path))
        key = os.path.basename(path)[:-5]
        envs.setdefault(tuple(z.get(k, "?") for k in ("python", "numpy", "scipy")), []).append(z.get("node", "?"))
        if z["status"] != "ok":
            errors += 1
            print("  ERROR  %s: %s" % (key, z["error"].strip().split("\n")[-1][:100]))
            continue
        want = reference.get(cell_key(path, z))
        if want is None:
            continue
        checked += 1
        got = {r["config_id"]: r["regret"] for r in z["rows"]}
        bad = compare({k: got.get(k, float("nan")) for k in want}, want, 1e-6)
        failures += len(bad) > 0
        print("  %-6s %s%s" % ("FAIL" if bad else "PASS", key, "" if not bad else "\n           " + "\n           ".join(bad)))
    print()
    for used, nodes in sorted(envs.items()):                 # the compute nodes' interpreter vs this one
        same = list(used) == [here[k] for k in ("python", "numpy", "scipy")]
        print("  %-6s python %s, numpy %s, scipy %s on %d cell(s), e.g. %s" % (
            "PASS" if same else "DIFFER", *used, len(nodes), nodes[0]))
        if not same:
            print("           this interpreter is python %s, numpy %s, scipy %s" %
                  tuple(here[k] for k in ("python", "numpy", "scipy")))
            print("           " + ("these cells predate the version record (an older package)" if used[0] == "?"
                                   else "the compute nodes did not use the interpreter that ran selftest.py"))
            failures += used[0] != "?"
    print("\n%d cells, %d with an error, %d with a reference value%s" % (
        len(files), errors, checked,
        "" if not (failures or errors) else " -- send the failing cells"))
    if checked == 0:
        print("(no reference value for these cells; the completeness check is collect_regret.py)")
    return 1 if (failures or errors) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="a run directory produced by run_regret.py")
    ap.add_argument("--write-reference", action="store_true")
    ap.add_argument("--write-cells", default=None, help="record a run directory's cells as reference values")
    a = ap.parse_args()
    sys.exit(write_cells(a.write_cells) if a.write_cells else
             check_cells(a.cells) if a.cells else run(a.write_reference))
