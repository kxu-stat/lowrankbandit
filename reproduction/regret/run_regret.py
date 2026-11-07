#!/usr/bin/env python3
"""Run one slice of the regret experiment.  One task = one (case, replication, arm) cell, written as one JSON file; a
task whose file exists is skipped, so a re-run costs nothing.

    python3 run_regret.py FAMILY --chunk k --chunks N [options]

FAMILY   gaussian | binomial
--chunk k --chunks N   this array element runs tasks k-1, k-1+N, ... of the task list
--tiers main,karm      --levels 0.2,0.5,1.0   --horizon 10000   --replications B
                       (--tiers approx: the approximately low-rank truths, singular values ~ j^-2 and j^-1.5;
                       --tiers d32: d = 32; --tiers scale: the
                       regret against the horizon, each case at its own horizon, --horizon not used)
--designs sphere,normal   (each overrides simulation.PLAN for every design)
--out DIR              output root (default regret_FAMILY)
--trace                also store the per-round trajectory of every configuration
--only KEY             replay one cell

The arms of a (case, replication) are simulation.arms_of(case): simulation.ARMS (and, for binary rewards,
simulation.PM1_ARMS), or in the tiers approx and d32 simulation.TIER_ARMS (A_t2, oful, estr_k<k>; approx under
Gaussian rewards also estr_r4_k<k>, ESTR given rank 4):
  A_t<i>   our method, tau = algorithms.tau_rule(n, d, d, C_TAU_GRID[i]) for the values of C_TAU_RUN, the penalty
           constant by CV
  M_k<k>   the matched arm: our learner (constant by CV) with tau = ESTR's k-th T1
  oful, estr_k<k>          OFUL over its multipliers; ESTR at its k-th T1 over its multipliers (the authors' grid;
                           binary rewards: our GLM versions, coding "glm")
  estr_tau                 ESTR over its multipliers with our exploration length: T1 = the tau of A_t2 (main and karm
                           tiers)
  oful_pm1, estr_pm1_k<k>  binary rewards: the authors' own 2y - 1 treatment (the supplement)
The task list is a pure function of these options (see `tasks`), so run_regret.py and collect_regret.py agree on it
without a plan file.
"""
import argparse, hashlib, json, os, time, traceback
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
import simulation as S
import algorithms as A
import baselines as Bl

PACKAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VERSION")).read().split()[0]


def coding_of(case, arm):
    """The reward coding of a baseline cell: None (Gaussian), "glm" (our GLM OFUL / GLM-ESTR) or "pm1" (the authors'
    2y - 1 treatment)."""
    if case["family"] != "binomial" or arm.startswith(("A_", "M_")):
        return None
    return "pm1" if "pm1" in arm else "glm"


def estr_T1(case, horizon, rank=None):
    """ESTR's released T1 values for the case, k = 0..4 (baselines.specs: ceil(10^(k/4) r(2d - r))), at the case's
    rank or the given one."""
    specs = Bl.specs(case["d1"], case["rank"] if rank is None else rank, case["K"], case["noise_sd"], case["amp"],
                     horizon, arm_scale=case["arm_scale"])
    return [next(sp["T1"] for sp in specs if sp["config_id"] == "ESTR_BM_g1_t%02d_m00" % k) for k in range(S.T1_STEPS)]


def tasks(family, tiers=None, levels=None, horizon=S.HORIZON, replications=None, designs=None):
    """Every (case, replication, arm) of the study, with its exploration length and seeds."""
    out = []
    for design, design_tiers, B in S.plan(tiers, designs, replications):
        for case in S.cases(family, design_tiers, levels, [design]):
            d, K = case["d1"], case["K"]
            n = case.get("horizon", horizon)                 # the tier scale: the case's own horizon
            T1 = estr_T1(case, n)
            T1_r4 = estr_T1(case, n, 4) if any(a.startswith("estr_r4") for a in S.arms_of(case)) else None
            for b in range(S.FIRST_REPLICATION, S.FIRST_REPLICATION + B):
                seed = S.environment_seed(n, case["case_index"], b, case["seed_phase"], K)
                solver = S.solver_seeds(case["case_index"], b, case["seed_phase"], K)
                for arm in S.arms_of(case):
                    c_tau = tau = None
                    if arm.startswith("A_t"):
                        c_tau = S.C_TAU_GRID[int(arm[3:])]
                        tau = A.tau_rule(n, d, d, c_tau)
                    elif arm.startswith("M_k"):
                        tau = T1[int(arm[3:])]
                    elif arm == "estr_tau":
                        tau = A.tau_rule(n, d, d, S.C_TAU)          # ESTR's T1 = our tau
                    if tau is not None and not A.tau_feasible(tau, n):
                        continue                         # a value the cell cannot run (tau infeasible)
                    if "_k" in arm and arm.startswith("estr") and \
                            (T1_r4 if arm.startswith("estr_r4") else T1)[int(arm.split("_k")[1])] >= n:
                        continue                         # T1 beyond the horizon (short debug horizons)
                    out.append(dict(case=case, replication=b, arm=arm, c_tau=c_tau, tau=tau, horizon=n,
                                    seed=seed, solver_seeds=solver, key="%s_b%03d_%s" % (case["case_id"], b, arm)))
    return out


def provenance():
    """The versions a cell's numbers depend on, stored with every cell."""
    import platform, scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                node=os.uname().nodename)


def curve(regret, exploration, grid, extra=None):
    """Cumulative regret (total and during exploration) on the shared time grid."""
    cum, cum_ex = np.cumsum(regret), np.cumsum(np.where(exploration, regret, 0.0))
    z = dict(t=grid.tolist(), cumulative_regret=cum[grid - 1].tolist(), exploration_regret=cum_ex[grid - 1].tolist())
    for k, v in (extra or {}).items():
        z[k] = np.asarray(v)[grid - 1].tolist()
    return z


def fingerprint(env):
    """A short signature of the environment, so a replay can be checked to have
    regenerated exactly the environment the cell was scored on."""
    return dict(x_sum=float(env.contexts.sum()), a_sum=float(env.candidates.sum()),
                e_sum=float(env.noise.sum()), arm_sum=int(env.exploration_arms.sum()),
                theta_fro=float(np.linalg.norm(env.Theta)))


class OneRecovery:
    """ESTR's stage-1 recovery computed once per stage-1 sample: the five multipliers of one T1 see the same rows and
    seed, hence the same recovery."""

    def __init__(self, recover):
        self.recover, self.store = recover, {}

    def __call__(self, M, X, y, r, R, seed, family, nb, s=1.0, a=1.0):
        key = (seed, r, R, family, nb, s, a, hashlib.sha256(M.tobytes() + X.tobytes() + y.tobytes()).hexdigest())
        if key not in self.store:
            self.store[key] = self.recover(M, X, y, r, R, seed, family, nb, s, a)
        return self.store[key]


PACKAGE_RECOVER = Bl.recover


def baseline_specs(case, arm, horizon):
    """The configurations of a baseline cell: OFUL's five multipliers, ESTR's five multipliers at the k-th T1 (rank
    simulation.estr_rank), or (estr_tau) ESTR's five multipliers at T1 = our tau, every other setting as at the first
    T1 (its stage-1 solver seed included).  Under the approximately low-rank truth S_F and S2 are the truth's own
    norms."""
    coding = coding_of(case, arm)
    S_F = S2 = None
    if case.get("truth", "exact") != "exact":
        S_F, S2 = case["amp"] * case["truth_fro"], case["amp"] * case["truth_op"]
    specs = Bl.specs(case["d1"], S.estr_rank(case, arm), case["K"], case["noise_sd"], case["amp"], horizon,
                     S_F=S_F, S2=S2, coding=coding, arm_scale=case["arm_scale"])
    if arm.startswith("oful"):
        return [sp for sp in specs if sp["method"] == "OFUL"]
    if arm == "estr_tau":
        tau = A.tau_rule(horizon, case["d1"], case["d2"], S.C_TAU)
        return [dict(sp, T1=tau, C_T1=None, config_id=sp["config_id"].replace("_t00_", "_tau_"), eligible=tau < horizon)
                for sp in specs if sp["config_id"].startswith("ESTR_BM_g1_t00_")]
    k = int(arm.split("_k")[1])
    return [sp for sp in specs if sp["config_id"].startswith("ESTR_BM_g1_t%02d_" % k)]


def run_task(tk, grid, store_trace):
    case, n, arm = tk["case"], tk["horizon"], tk["arm"]
    env = S.Environment(n, case, tk["seed"])
    rows, curves, traces = [], {}, {}
    if arm.startswith(("A_", "M_")):                               # our method / the matched arm
        out = A.run_algorithm1(env, tk["tau"], "cv")
        tr = out["trace"]
        method = "Proposed" if arm.startswith("A_") else "Matched"
        rows.append(dict(method=method, config_id=arm, tau=tk["tau"], c_tau=tk["c_tau"], regret=tr["regret"].sum(),
                         exploration_regret=tr["regret"][tr["exploration"]].sum(),
                         post_regret=tr["regret"][~tr["exploration"]].sum(), score=out["score"],
                         sigma_hat=out["sigma_hat"], lambda_n=out["levels"][-1], rank_n=int(tr["fitted_rank"][-1]),
                         err_n=tr["theta_error"][-1], cv=json.dumps(out["cv"])))
        curves[arm] = curve(tr["regret"], tr["exploration"], grid,
                            dict(theta_error=tr["theta_error"], fitted_rank=tr["fitted_rank"]))
        tr["lam"] = out["levels"]
        traces[arm] = {k: np.asarray(v).tolist() for k, v in tr.items()}
    else:                                                          # OFUL / ESTR over their multipliers
        Bl.recover = OneRecovery(PACKAGE_RECOVER)                  # a fresh store per cell
        suffix = ("_pm1" if coding_of(case, arm) == "pm1" else "") + ("_r4" if arm.startswith("estr_r4") else "")
        for sp in baseline_specs(case, arm, n):
            if not sp["eligible"]:
                continue
            seed = tk["solver_seeds"][sp["source_parameter_index"] // 5] if sp["method"] == "ESTR" else None
            tr = Bl.run_baseline(env, sp, seed)
            cid = sp["config_id"] + suffix
            method = "ESTR_tau" if arm == "estr_tau" else sp["method"] + suffix
            rows.append(dict(method=method, config_id=cid, tau=sp["T1"],
                             multiplier=sp["multiplier"], regret=tr["regret"].sum(),
                             exploration_regret=tr["regret"][tr["exploration"]].sum(),
                             post_regret=tr["regret"][~tr["exploration"]].sum(), **tr.get("bm", {})))
            curves[cid] = curve(tr["regret"], tr["exploration"], grid)
            traces[cid] = {k: np.asarray(v).tolist() for k, v in tr.items() if k != "bm"}
    return dict(rows=rows, curves=curves, fingerprint=fingerprint(env), traces=traces if store_trace else None)


def stamp(tk):
    """The package version a cell was made with; a cell is reused only under the same version."""
    return dict(package=PACKAGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["gaussian", "binomial"])
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--tiers", default=None, type=lambda v: v.split(","))
    ap.add_argument("--designs", default=None, type=lambda v: v.split(","))
    ap.add_argument("--levels", default=None)
    ap.add_argument("--horizon", type=int, default=S.HORIZON)
    ap.add_argument("--replications", type=int, default=None, help="override the plan's B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--only", default=None, help="replay one cell by key, e.g. sph_d16_r2_K16_rho020_b601_A_t2")
    a = ap.parse_args()
    assert 1 <= a.chunk <= a.chunks
    levels = None if a.levels is None else [float(v) if a.family == "gaussian" else int(v) for v in a.levels.split(",")]
    out = a.out or "regret_%s" % a.family
    all_tasks = tasks(a.family, a.tiers, levels, a.horizon, a.replications, a.designs)
    mine = [t for t in all_tasks if t["key"] == a.only] if a.only else all_tasks[a.chunk - 1::a.chunks]
    if a.only and not mine:
        raise SystemExit("no cell with key %s" % a.only)
    os.makedirs(os.path.join(out, "cells"), exist_ok=True)
    print("run_regret: %s tasks %d of %d -> %s" % (a.family, len(mine), len(all_tasks), out), flush=True)
    for tk in mine:
        path = os.path.join(out, "cells", tk["key"] + ".json")
        if a.only and os.path.exists(path):
            os.replace(path, path + ".previous")            # a replay never overwrites the recorded cell
        done = json.load(open(path)) if os.path.exists(path) else None
        if done is not None and done["status"] == "ok":
            if done.get("stamp") != stamp(tk):
                raise SystemExit("%s was made by another package version: write to a new --out" % path)
            print("skip", tk["key"], flush=True)
            continue
        t0 = time.time()
        try:
            value = run_task(tk, S.time_grid(tk["horizon"]), a.trace)
            value.update(status="ok", error=None)
        except Exception:
            value = dict(status="error", error=traceback.format_exc(), rows=[], curves={}, traces=None)
        value.update(task=tk, elapsed=time.time() - t0, stamp=stamp(tk), **provenance())
        tmp = path + ".partial.%d" % os.getpid()                 # atomic write
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, path)
        print("%s %s tau=%s regret=%s (%.0fs)" % (value["status"].upper(), tk["key"],
              ",".join(str(r["tau"]) for r in value["rows"]) or "-",
              ",".join("%.3f" % r["regret"] for r in value["rows"]) or value["error"].strip().split("\n")[-1][:120],
              value["elapsed"]), flush=True)


if __name__ == "__main__":
    main()
