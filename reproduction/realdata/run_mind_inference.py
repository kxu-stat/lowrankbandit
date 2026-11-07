#!/usr/bin/env python3
"""Run one slice of the MIND inference study.  One task = one stream of one world: one collection to the world's
largest n (Step 1 of Algorithm 2; 80,000 rounds in both worlds), then Steps
2-4 at every n of the world's grid -- the same collection stopped at n -- for every choice of n0 and every target of the
setup file, written as one JSON cell.  A task whose cell exists is skipped.

    python3 run_mind_inference.py PHASE --setup SETUP_NPZ --chunk k --chunks N [--out DIR]

The design:
  worlds    "main": the semi-synthetic world of the regret study (mind.py): the click of a displayed article is
            Bernoulli(expit(c' gamma* + m' Theta* x)), (gamma*, Theta*) the rank-2 truth of setup_mind.py.
            "down4" (mind.downsampled): the click model that a logistic fit estimates from MIND's logs after keeping
            every click and a quarter of the non-clicks -- the main truth with every log-odds raised by log 4 (the
            intercept), Theta* and the other coefficients unchanged, so every target's mu* is the same as in main --
            on the same rounds.  Only the truth changes: a run keeps every round and discards nothing.  Algorithm 2
            is given r0 = 2 in both.  Stream b has the same seed in both worlds and both collect to 80,000 rounds
            (tau = 57), so the two see the same impressions, displayed articles, exploration arms and uniforms in
            every round; only the articles chosen after the exploration differ
  rounds    each round draws one of the 86,045 eligible K = 10 impressions uniformly WITH replacement, and K of its
            displayed articles uniformly without replacement: rounds i.i.d. from the empirical distribution of real
            rounds, as the theory assumes
  Step 1    inference/algorithms.collect, the regret study's Algorithm 1 on the cmx model: tau = ceil(c_tau (2D + Q)
            log(n D^2)) with c_tau = 1/20 and Q = 35 (tau = 57 at n = 80,000), the penalty constant by
            10-fold CV inside each track
  Steps 2-4 at n = 10,000, 20,000, 30,000, 40,000, 60,000, 80,000 in both worlds for every
            choice of n0 in ARMS (the inference simulations' choices: the design-stage
            split with cap 3/4, displayed, the information rule, the fixed shares, no refit and the
            design-stage split with cap 0.9), with C_ref = 1/15 and
            C_cal = 1.  The rows are in the paper's normalization (mind.py), the units of the design-stage
            split's planning value kappa_D (the world's) and of the information rule's kappa0
  targets   mu* = <A, theta*> for the 32 loadings of the setup file (setup_mind.targets: for each of the 8 largest
            categories the readers' log odds ratio "or", its user-vector part "orx", its interaction part "ort", and
            the readers' content log-odds "lvl")
Step 2 does not involve the loading, so it runs once per n and distinct n0 (algorithms.fold_estimates), and Steps 3-4
run for the 32 loadings together (algorithms.calibrate_loadings).  Every row also carries the collection's cumulative
expected regret at n (algorithms.collection_regret) and the target's Cramer-Rao standard error for the collected design
(algorithms.cramer_rao: V_CR, kappa_A, the ratio V_pqv / V_CR and its prediction sqrt(C_cal kappa_A log n_tr /
kappa_hat)).  Phases: "evaluation" (500 streams per world) and "debug" (two streams
per world of 4,000 rounds; minutes each; no claim).  The paper's figures are drawn at FIGURE_N, 40,000 in the
down-sampled world (the main text) and 80,000 in the main world (the appendix), fixed
before the run; every n is in the tables.

Besides the cell, each task writes cells/KEY_stream.npz: the displayed pair chosen in every round (an index into the
prepared file's pair arrays) and its click, in the order of the collection (tau exploration rounds, then the two
tracks alternating, track 0 first), and the exploration pilot fit.  The rows of a round are a function of its pair
(MindDesign.rows), so with the cell's CV choices and pilot score any other target can be evaluated later on the same
streams without collecting again.  The stream file also holds the seed and the arm played in every round (its
index among the round's K displayed articles), and the cell records the exploration sample's numbers (the information
rule's kappa0, the exploration clicks, the pilot's rank) and every choice's n0 at every n.
"""
import argparse, hashlib, json, os, sys, time, traceback
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inference"))
import algorithms as IA                       # inference/algorithms.py: Algorithm 2 (NOT regret/algorithms.py)
import simulation as IS                       # inference/simulation.py: the choices of n0 (N0_CHOICES)
from run_inference import split, PACKAGE      # inference/run_inference.py: n0 of a fixed share or design-stage split
import mind

D, K = 16, 10                                 # the setup's D, the displayed articles per round
C_TAU, C_LAM = 1 / 20, "cv"                   # the regret study's constants (regret/simulation.C_TAU; CV)
WORLDS = {"main": 1.0, "down4": 1 / 4}        # the share of the non-clicks kept in the logs of each world's click model
GRID = {"down4": (10000, 20000, 30000, 40000, 60000, 80000),          # n, in total rounds, per world,
        "main": (10000, 20000, 30000, 40000, 60000, 80000)}           # the same in both
FIGURE_N = {"down4": 40000, "main": 80000}    # the n of the paper's figures (debug: the largest n)
ARMS = IS.N0_CHOICES                          # the choices of n0, as in the inference simulations
PHASES = {                                    # streams per world, first seed, the grid of n per world
    "evaluation": dict(B={"main": 500, "down4": 500}, seed=44_000_000, grid=GRID),
    "debug":      dict(B={"main": 2, "down4": 2}, seed=30_000_000, grid={"down4": (2000, 4000), "main": (2000, 4000)}),
}


def grid_of(phase, world):
    """The n of a world's grid in a phase."""
    return PHASES[phase]["grid"][world]


def figure_n(phase, world):
    """The n of a world's paper figures: FIGURE_N, or the grid's largest n when it is not in the grid (debug)."""
    grid = grid_of(phase, world)
    return FIGURE_N[world] if FIGURE_N[world] in grid else grid[-1]


def planned_B(phase, world):
    """The streams of a world in a phase."""
    return PHASES[phase]["B"][world]


class MindDesign:
    """One stream of a world as a design of inference/algorithms: it supplies each round's rows (next_round) and the
    chosen article's click (reward_arm).  The row of a displayed pair is z = (c, m, x, vec(m x')) in design units
    (mind.covariates: the paper's normalization), vec column-major (inference/algorithms' convention), so that
    z' theta* = c' gamma* + m' Theta* x with theta* = (gamma*, vec(Theta*))."""

    family, nb = "binomial", 1
    anchor_at_main_fit = True                 # the pilot score at the main-effects-only fit, as in Algorithm 1

    def __init__(self, data, setup):
        self.data = data
        self.d1 = self.d2 = D
        self.K, self.r0 = K, int(setup["rank"])
        self.main_sizes = mind.group_sizes("cmx", D)                     # (3, 16, 16): Q = 35 coefficients
        self.d = sum(self.main_sizes) + D * D
        self.radius = tuple(setup["radii_main"]) + (float(setup["radius_Theta"]),)   # the regret study's balls
        Theta = setup["truth_Theta"]
        self.theta = np.r_[setup["truth_gamma"], IA.vec(Theta)]
        U, _, Vt = np.linalg.svd(Theta)
        self.Us, self.Vs = U[:, :self.r0], Vt[:self.r0].T                  # the truth's frame (algorithms.cramer_rao)
        self.kappa_D = float(setup["kappa_D"])                           # the design-stage split's planning value
        self.pool = np.flatnonzero(data["eligible_K10"])                 # the eligible K = 10 impressions
        self.pairs = self.p = None                                       # this round's displayed pairs and click probabilities
        self.chosen, self.clicks = [], []                                # the stream's record

    def rows(self, pairs):
        """The rows z of displayed pairs."""
        data = self.data
        C = mind.covariates(data, pairs, "cmx")                          # (c, m, x)
        m = data["M"][data["c_news"][pairs]]
        x = data["X"][data["imp_user"][data["c_imp"][pairs]]]
        inter = np.einsum("kj,ki->kji", x, m).reshape(len(pairs), D * D)  # row k: vec(m_k x_k'), column-major
        return np.concatenate([C, inter], axis=1)

    def next_round(self, rng):
        i = self.pool[rng.integers(len(self.pool))]                     # one impression, with replacement
        self.pairs = self.data["first"][i] + rng.choice(int(self.data["imp_size"][i]), self.K, replace=False)
        Z = self.rows(self.pairs)
        self.p = mind.expit(Z @ self.theta)
        return Z

    def reward_arm(self, rng, a):
        y = float(rng.random() < self.p[a])
        self.chosen.append(int(self.pairs[a]))
        self.clicks.append(int(y))
        return y


def world_setup(setup, world):
    """The setup of a world: the setup file's for main, mind.downsampled for down4."""
    if world == "main":
        return setup
    return mind.downsampled(setup, WORLDS[world])


def tasks(phase):
    """Every stream of the phase in both worlds: task 1 is stream 1 of main, task 2 stream 1 of down4, task 3 stream 2
    of main, ...; past the main world's last stream, down4's streams alone (an interrupted run leaves the worlds even
    over their common streams).  Stream b has the same seed in both worlds."""
    ph = PHASES[phase]
    out = []
    for b in range(1, max(ph["B"].values()) + 1):
        for world in WORLDS:
            if b <= ph["B"][world]:
                out.append(dict(phase=phase, world=world, replication=b, seed=ph["seed"] + b,
                                key="%s_%s_b%03d" % (phase, world, b)))
    return out


def n0_of(how, N, Dsg, rec):
    """Each track's n0 under one choice, on the first N rounds after the exploration: the information rule
    (algorithms.n0_rule), or a fixed share or design-stage split (run_inference.split)."""
    return IA.n0_rule(Dsg, rec, N)[0] if how == "rule" else split(how, N, Dsg, rec)


def run_task(tk, data, setup):
    """One stream: the collection, then Steps 2-4 at every n for every choice of n0 and every target.  Returns the
    cell's value and the stream's record."""
    ph = PHASES[tk["phase"]]
    setup = world_setup(setup, tk["world"])
    names, As = [str(s) for s in setup["target_names"]], setup["target_A"]
    Dsg = MindDesign(data, setup)
    grid = ph["grid"][tk["world"]]
    n_max = max(grid)
    tau = IA.tau_rule(n_max, D, D, C_TAU, q=sum(Dsg.main_sizes))         # 57 at n_max = 80,000 (both worlds)
    t0 = time.time()
    rec = IA.collect(Dsg, tau, n_max - tau, np.random.default_rng(tk["seed"]), C_LAM)
    t_collect, t0 = time.time() - t0, time.time()
    results, n0_all = [], {}
    for n in grid:                                                       # the same collection, stopped at n
        N = n - tau
        n0 = n0_all[str(n)] = {arm: n0_of(how, N, Dsg, rec) for arm, how in ARMS}
        V_CR, kappa_A = IA.cramer_rao(Dsg, rec, N, As)                   # the design's, every target
        regret = IA.collection_regret(rec, N)                            # the collection's regret at n
        PT_A = [float(np.linalg.norm(IA.tangent_project(A, Dsg.Us, Dsg.Vs, D, D))) for A in As]
        A_norm = [float(np.linalg.norm(A)) for A in As]                  # PT_A / A_norm is reported
        for v in sorted(set(n0.values())):
            F = IA.fold_estimates(Dsg, rec, N, v)                        # Step 2, once per distinct n0
            out = IA.calibrate_loadings(Dsg, rec, N, As, F)              # Steps 3-4, every target
            for arm in [arm for arm, _ in ARMS if n0[arm] == v]:
                for l in range(len(names)):                              # one row per target
                    o = out[l]
                    row = dict(n=n, arm=arm, target=names[l], nuis_caps=F["nuis_cap"])
                    row.update(o)
                    row.update(PT_A=PT_A[l], A_norm=A_norm[l], V_CR=float(V_CR[l]), kappa_A=float(kappa_A[l]),
                               VCR_ratio=float(o["V_pqv"] / V_CR[l]),
                               VCR_ratio_pred=float(np.sqrt(IA.C_CAL * kappa_A[l] * np.log(N // 2) / o["kap_hat"])))
                    row.update(regret)
                    results.append(row)
    value = dict(results=results, tau=tau, n_max=n_max, pool=int(len(Dsg.pool)), kappa_D=Dsg.kappa_D,
                 intercept=float(Dsg.theta[0]), cv=rec["cv"], score=rec["score"], solver=rec["solver"],
                 n0=n0_all, kappa0=IA.kappa0_exploration(Dsg, rec), clicks_ex=int(np.sum(rec["Y_ex"])),
                 pilot_rank=IA.pilot_rank(Dsg, rec), t_collect=round(t_collect, 1), t_infer=round(time.time() - t0, 1))
    stream = dict(chosen=np.array(Dsg.chosen, dtype=np.int32), clicks=np.array(Dsg.clicks, dtype=np.int8),
                  pilot=rec["pilot"], tau=tau, seed=tk["seed"], arm_ex=rec["arm_ex"].astype(np.int8),
                  arm=np.array(rec["arm"], dtype=np.int8))
    return value, stream


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def stamp(setup_sha):
    """What a finished cell must have been made under to be reused (checked here and by collect_mind_inference.py)."""
    return dict(setup_sha=setup_sha, package=PACKAGE, worlds=WORLDS, c_tau=C_TAU, c_lam=C_LAM, C_ref=IA.C_REF,
                C_cal=IA.C_CAL, c0=IA.C0)


def provenance():
    import platform, scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                node=os.uname().nodename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--setup", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "mind_small_prepared_glove.npz"))
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert 1 <= a.chunk <= a.chunks
    out = a.out or "mind_inference_%s" % a.phase
    os.makedirs(os.path.join(out, "cells"), exist_ok=True)
    setup = dict(np.load(a.setup))
    assert str(setup["package"]) == mind.SETUP_VERSION, "a setup file of setup_mind.py %s is needed" % mind.SETUP_VERSION
    setup_sha = sha16(a.setup)
    mine = tasks(a.phase)[a.chunk - 1::a.chunks]
    print("run_mind_inference: %s, tasks %d -> %s" % (a.phase, len(mine), out), flush=True)
    data = None
    for tk in mine:
        path = os.path.join(out, "cells", tk["key"] + ".json")
        done = json.load(open(path)) if os.path.exists(path) else None
        if done is not None and done["status"] == "ok":
            if done.get("stamp") != stamp(setup_sha):
                raise SystemExit("%s was made under another setup or package version: write to a new --out" % path)
            print("skip", tk["key"], flush=True)
            continue
        if data is None:
            data = mind.load(a.data, D, setup)
            assert str(setup["data_sha"]) == hashlib.sha256(open(a.data, "rb").read()).hexdigest(), \
                "the prepared file is not the one the setup was made from"
        t0 = time.time()
        try:
            value, stream = run_task(tk, data, setup)
            np.savez_compressed(os.path.join(out, "cells", tk["key"] + "_stream.npz"), **stream)
            value.update(status="ok", error=None)
        except Exception:
            value = dict(status="error", error=traceback.format_exc(), results=[])
        value.update(task=tk, elapsed=round(time.time() - t0, 1), stamp=stamp(setup_sha), **provenance())
        tmp = path + ".partial.%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, path)
        print("%s %s (%s)" % (value["status"].upper(), tk["key"],
              "collection %.0fs, inference %.0fs" % (value["t_collect"], value["t_infer"]) if value["status"] == "ok"
              else value["error"].strip().split("\n")[-1][:160]), flush=True)


if __name__ == "__main__":
    main()
