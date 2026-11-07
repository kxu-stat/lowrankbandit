#!/usr/bin/env python3
"""Run one slice of the MIND regret study.  One task = one (replication, arm) cell, written as one JSON file; a task
whose file exists is skipped, so a re-run costs nothing.  Mirrors regret/run_regret.py.

    python3 run_mind.py --setup SETUP_NPZ --chunk k --chunks N [--block down4 | main] [--best BEST_TSV]
                        [--horizon n] [--replications B] [--out DIR] [--only KEY]

The design.  D = 16, K = 10, the cmx learner (the paper's model: c,
m, x main effects and Theta), kept users, rounds drawn i.i.d. from the real rounds (each round an eligible impression
WITH replacement, K of its displayed articles without replacement: mind.draw_rounds), B = 50 streams.  Clicks are
drawn from the rank-r truth of setup_mind.py (the semi-synthetic world, mind.py); the regret is the expected regret.
No method sees the truth.  Our constants are the simulations' (regret/simulation.py), carried over without tuning;
every constant a baseline is given comes from the truth, as in the simulations (setup_mind.py).  Two blocks (BLOCKS):
  down4   the down-sampled world (the inference's main text: every log-odds plus log 4), horizon 40,000, every arm
          below and every configuration of every grid; the predictions M1-M6 (collect_mind.py)
  main    the main world (appendix; reported, no prediction), horizon 80,000, five arms: ours
          (A_t2), the two-track collection, logistic OFUL and GLM-ESTR each at the configuration that was best in the
          down4 block (read from that block's collected/best.tsv, --best), and the auxiliary-covariate greedy over its
          tau grid

The arms of a replication, each one cell (cf. regret/simulation.ARMS):
  A_t<i>         our method, tau = algorithms.tau_rule(n, D, D, C_TAU_GRID[i], Q) = ceil(c_tau (2D + Q) log(n D^2))
                 for the values of C_TAU_RUN, with Q = 35 main-effect coefficients (A_t2: c_tau = C_TAU = 1/20, the
                 reported configuration; A_t3: 1/10, the sensitivity analysis; A_t1: 1/40 gives tau <= Q and is
                 infeasible), the penalty constant by cross-validation inside the run
  two_track      the data collection of the inference (Algorithm 2's Step 1, inference/algorithms.collect, as the MIND
                 inference runs it): tau = the tau of A_t2 rounds of shared exploration, then two greedy tracks in
                 alternation, each refitting Algorithm 1's estimator on (S_o, its own rounds) with the penalty constant
                 by CV inside the track.  It sees the stream's rounds and uniforms (TwoTrackStream); its regret is the
                 sum over the exploration and both tracks.  When n - tau is odd the last round belongs to no track; it
                 is played as an exploration round (the stream's uniform arm).  Prediction M6
  M_k<k>         the matched arm: our learner with tau = ESTR's k-th T1
  oful           logistic OFUL on (c, m, x, vec(m x')), the authors' 5 multipliers (the main competitors, as in the
  estr_k<k>      binary simulations); GLM-ESTR at its k-th T1 on the augmented design (m, c) x (x, 1) of rank r + 1,
                 the authors' 5 multipliers
  estr_tau       GLM-ESTR as estr_k0 but with our exploration length, T1 = the tau of A_t2, the authors'
                 5 multipliers
  oful_pm1, estr_pm1_k<k>   the authors' own binary treatment (linear model on 2y - 1): the supplement
  aux_greedy     the auxiliary-covariate greedy: the logistic fit of c alone, greedy after tau uniform rounds, over
                 the tau grid GREEDY_TAUS (one cell; its fit: mind._fit)
  cm_greedy      the main-effect greedy: the logistic fit of c and m (the main effects that can change which article
                 is best: x's is the same for the K candidates of a round), the same tau grid, in Algorithm 1's balls
                 for these blocks (reported)
  oful_best, estr_best   (main block only) logistic OFUL, GLM-ESTR at the one configuration given by --best
Every configuration of every grid is reported; the competitors' best configuration is chosen in hindsight
(collect_mind.py).  The task list is a pure function of the options (`tasks`), so run_mind.py and collect_mind.py agree
on it without a plan file.

Units (mind.py).  Our learner sees the rows in the paper's normalization (scale "design": E[m m'] = I, x of length
sqrt(D), c standardized), its solver tests read in the units of unit-length arms (MindEnv.arm_scale = D), so that the
change of units changes no iterate.  The competitors see m and x divided by sqrt(D), the unit-length arms and contexts
their constants are written for (as regret/baselines.py converts them in the simulations), and c standardized; their
declared norms are in those units (setup_mind.py).
"""
import argparse, hashlib, json, os, sys, time, traceback
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per process
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "regret"))
import algorithms as A
import baselines as Bl
import simulation as S                       # the time grid, our constants (C_TAU, C_TAU_GRID), T1_STEPS
import run_regret as RR                       # OneRecovery: ESTR's stage 1 once per T1 cell
import mind

D, K = 16, 10
MAIN = "cmx"                                  # the learner's main effects (the paper's model)
Q = mind.N_COV + 2 * D                        # its number of main-effect coefficients
C_TAU, C_TAU_GRID, C_TAU_RUN, T1_STEPS = S.C_TAU, S.C_TAU_GRID, S.C_TAU_RUN, S.T1_STEPS
GREEDY_TAUS = (10, 20, 50, 100, 200)          # the greedy comparators' exploration grid
B = 50                                        # streams, as every simulation case
WORLDS = {"down4": 1 / 4, "main": 1.0}        # the share of the non-clicks kept in the logs of each world's click model
OURS = "A_t%d" % C_TAU_GRID.index(C_TAU)     # our reported configuration, c_tau = 1/20
ARMS = (tuple("A_t%d" % C_TAU_GRID.index(c) for c in C_TAU_RUN) + ("two_track",)
        + tuple("M_k%d" % k for k in range(T1_STEPS))
        + ("oful",) + tuple("estr_k%d" % k for k in range(T1_STEPS)) + ("estr_tau",)
        + ("oful_pm1",) + tuple("estr_pm1_k%d" % k for k in range(T1_STEPS)) + ("aux_greedy", "cm_greedy"))
MAIN_ARMS = (OURS, "two_track", "oful_best", "estr_best", "aux_greedy")
BLOCKS = {
    "down4": dict(horizon=40000, B=B, arms=ARMS),          # the main text: every arm and configuration, M1-M6
    "main": dict(horizon=80000, B=B, arms=MAIN_ARMS),      # the appendix: reported
}
SEED_BASE = 20_261_000                        # stream b: SEED_BASE + 10 (b - 1) + 1, 2, 3 (seeds)


def seeds(b):
    """Stream, environment and ESTR solver seeds of replication b: SEED_BASE + 10 (b - 1) + 1, 2, 3, i.e. 20,261,001,
    002, 003 for b = 1, 20,261,011, 012, 013 for b = 2, ..., 20,261,491, 492, 493 for b = 50 (the last digit names the
    generator).  They depend on the replication only, so every arm of a replication sees the same rounds, exploration
    arms and uniforms (paired comparisons)."""
    s = SEED_BASE + 10 * (b - 1)
    return dict(stream=s + 1, env=s + 2, solver=s + 3)


def world_setup(setup, block):
    """The setup of a block's world.  main: the setup file's.  down4: the down-sampled click model of the inference
    study (mind.downsampled: the truth's intercept raised by log(1 / w)), with every constant the methods are given
    recomputed from that truth by the rules of setup_mind.py (the setup file's keys "w4_*", written by
    add_world_mind.py)."""
    if block == "main":
        return setup
    import add_world_mind
    w = WORLDS[block]
    out = mind.downsampled(setup, w)
    tag = "w%d_" % round(1 / w)
    for k in add_world_mind.WORLD_KEYS:
        out[k] = setup[tag + k]
    return out


def setting_name(block):
    return "D%02d_K%02d_%s" % (D, K, block)


def is_alg1(arm):
    return arm.startswith(("A_", "M_"))


def coding_of(arm):
    """The reward coding of a baseline cell: "pm1" (the authors' 2y - 1) or "glm" (our logistic versions)."""
    return "pm1" if "pm1" in arm else "glm"


def estr_T1(rank, horizon):
    """ESTR's five T1 at the declared rank on the augmented design (side D + N_COV, rank r + 1): ceil(10^(k/4) k0)."""
    specs = Bl.specs(D, rank, K, 1.0, 1.0, horizon, main_dim=mind.N_COV, coding="pm1")
    return [next(sp["T1"] for sp in specs if sp["config_id"] == "ESTR_BM_g1_t%02d_m00" % k) for k in range(T1_STEPS)]


def best_configs(path):
    """The configuration of logistic OFUL and of GLM-ESTR (our logistic versions) with the smallest mean regret in the
    down4 block, from that block's collected/best.tsv (collect_mind.py): {"oful_best": config_id, "estr_best":
    config_id}.  The main block runs these two."""
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f]
    pick = {r["method"]: r["config_id"] for r in rows if r["block"] == "down4"}
    return {"oful_best": pick["OFUL"], "estr_best": pick["ESTR"]}


def tasks(setup, block="down4", horizon=None, replications=None, best=None):
    """Every cell of a block.  An infeasible tau drops its arm, as in the simulations (regret/algorithms.tau_feasible:
    here tau <= Q, c_tau = 1/40, or tau >= the horizon); an ESTR T1 at or beyond the horizon drops its
    configurations (baselines.specs `eligible`), so a short debug horizon runs fewer ESTR cells.  best (the main
    block): the path of the down4 block's collected/best.tsv."""
    out = []
    spec = BLOCKS[block]
    horizon = horizon or spec["horizon"]
    rank = int(setup["rank"])                                # the declared rank: the truth's and ESTR's
    T1 = estr_T1(rank, horizon)
    chosen = best_configs(best) if block == "main" else {}
    for b in range(1, (replications or spec["B"]) + 1):
        for arm in spec["arms"]:
            c_tau = tau = None
            if arm.startswith("A_t"):
                c_tau = C_TAU_GRID[int(arm[3:])]
                tau = A.tau_rule(horizon, D, D, c_tau, Q)
            elif arm == "two_track":
                tau = A.tau_rule(horizon, D, D, C_TAU, Q)            # the tau of A_t2 (the inference's rule)
            elif arm.startswith("M_k"):
                tau = T1[int(arm[3:])]
            elif arm == "estr_tau":
                tau = A.tau_rule(horizon, D, D, C_TAU, Q)        # GLM-ESTR's T1 = our tau
            elif arm.startswith("estr") and arm != "estr_best":
                tau = T1[int(arm.split("_k")[1])]
            alg1 = is_alg1(arm) or arm == "two_track"           # Algorithm 1's estimator after tau rounds
            if tau is not None and not (A.tau_feasible(tau, horizon, Q) if alg1 else tau < horizon):
                continue
            out.append(dict(block=block, setting=setting_name(block), D=D, K=K, main=MAIN, replication=b, arm=arm,
                            c_tau=c_tau, tau=tau, rank=rank, horizon=horizon, seeds=seeds(b), config=chosen.get(arm),
                            key="%s_b%03d_%s" % (setting_name(block), b, arm)))
    return out


def provenance():
    import platform, scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                node=os.uname().nodename)


def curve(tr, grid):
    """Cumulative regret, expected reward (the chosen arms' means) and realized reward on the shared time grid."""
    return dict(t=grid.tolist(),
                cumulative_regret=np.cumsum(tr["regret"])[grid - 1].tolist(),
                cumulative_expected=np.cumsum(tr["expected"])[grid - 1].tolist(),
                cumulative_reward=np.cumsum(tr["reward"])[grid - 1].tolist())


def row(method, config_id, tau, tr, **extra):
    """One configuration's result on one stream; last_quarter = the regret of the last quarter of the horizon."""
    explore = tr["exploration"] if "exploration" in tr else np.arange(len(tr["regret"])) < (tau or 0)
    return dict(method=method, config_id=config_id, tau=tau, regret=float(tr["regret"].sum()),
                exploration_regret=float(tr["regret"][explore].sum()), post_regret=float(tr["regret"][~explore].sum()),
                last_quarter=float(tr["regret"][-(len(tr["regret"]) // 4):].sum()),
                expected=float(tr["expected"].sum()), clicks=float(tr["reward"].sum()), **extra)


def expected_of(env, tr):
    return np.array([env.means(t)[a] for t, a in enumerate(tr["arm"])])


def baseline_specs(setup, rank, arm, n, config=None):
    """The configurations of a baseline cell.  OFUL sees (c, m, x, vec(m x')) (main_dim Q), ESTR the augmented design
    of the covariates c (main_dim N_COV); the declared S_F and S2 of the cell's coding come from the truth
    (setup_mind.py): its logistic coefficients ("glm") or the least-squares fit of 2 p* - 1 ("pm1"); ESTR's are the
    norms of its augmented truth, which carries the m and x main effects (mind.declared_estr).  oful_best and
    estr_best (the main block): the one configuration `config`."""
    coding = coding_of(arm)
    if arm.startswith("oful"):
        specs = Bl.specs(D, rank, K, 0.5, 1.0, n, main_dim=Q, S_F=float(setup["%s_S_F_cmx" % coding]),
                         S2=float(setup["%s_S2_cmx" % coding]), coding="pm1" if coding == "pm1" else None)
        return [sp for sp in specs if sp["method"] == "OFUL" and (config is None or sp["config_id"] == config)]
    specs = Bl.specs(D, rank, K, 0.5, 1.0, n, main_dim=mind.N_COV, S_F=float(setup["%s_S_F_estr" % coding]),
                     S2=float(setup["%s_S2_estr" % coding]), coding="pm1" if coding == "pm1" else None)
    if arm == "estr_best":
        return [sp for sp in specs if sp["config_id"] == config and sp["eligible"]]
    if arm == "estr_tau":                                    # the first T1's settings with T1 = our tau
        tau = A.tau_rule(n, D, D, C_TAU, Q)
        return [dict(sp, T1=tau, C_T1=None, config_id=sp["config_id"].replace("_t00_", "_tau_"), eligible=tau < n)
                for sp in specs if sp["config_id"].startswith("ESTR_BM_g1_t00_")]
    k = int(arm.split("_k")[1])
    return [sp for sp in specs if sp["config_id"].startswith("ESTR_BM_g1_t%02d_" % k) and sp["eligible"]]


def inference_algorithms():
    """inference/algorithms.py, the code of Algorithm 2 (its Step 1 is the two_track arm).  It and the
    inference/simulation.py it imports share their file names with the regret/ modules this script imported first, so
    they are imported from the inference directory with those two names cleared, and the regret modules are put back
    afterward (as selftest.py does)."""
    saved = {m: sys.modules.pop(m) for m in ("algorithms", "simulation") if m in sys.modules}
    sys.path.insert(0, os.path.join(HERE, "..", "inference"))
    try:
        import algorithms as IA
    finally:
        sys.path.pop(0)
        for m in ("algorithms", "simulation"):
            sys.modules.pop(m, None)
        sys.modules.update(saved)
    return IA


class TwoTrackStream:
    """The stream as a design of inference/algorithms.collect (the interface of run_mind_inference.MindDesign): round g
    supplies the K rows z = (c, m, x, vec(m x')) of its displayed pairs in design units (vec column-major), and the
    click of the chosen one is uniforms[g] < p_g(a), as for every arm of the stream.  collect calls next_round and
    reward_arm once per round, in the order of the stream; the chosen arms are recorded."""

    family, nb = "binomial", 1
    anchor_at_main_fit = True                 # the pilot score at the main-effects-only fit, as in Algorithm 1

    def __init__(self, env, setup):
        self.env, self.g, self.arms = env, 0, []
        self.d1 = self.d2 = D
        self.K, self.r0 = K, int(setup["rank"])
        self.main_sizes = mind.group_sizes(MAIN, D)                      # (3, 16, 16)
        self.d = sum(self.main_sizes) + D * D
        self.radius = tuple(setup["radii_main"]) + (float(setup["radius_Theta"]),)   # Algorithm 1's balls
        self.theta = np.r_[setup["truth_gamma"], setup["truth_Theta"].reshape(-1, order="F")]

    def next_round(self, rng):
        e, g = self.env, self.g
        inter = np.einsum("j,ki->kji", e.contexts[g], e.candidates[g]).reshape(self.K, D * D)   # row k: vec(m_k x')
        return np.concatenate([e.main[g], inter], axis=1)

    def reward_arm(self, rng, a):
        e, g = self.env, self.g
        self.arms.append(int(a))
        self.g += 1
        return e.reward(g, a, e.means(g))


def two_track(e, setup, tau, seed):
    """The two_track arm on the stream of env e (design units): inference/algorithms.collect with tau shared
    exploration rounds and (n - tau) // 2 rounds per track, the penalty constant by CV inside each track (the MIND
    inference's Step 1); its own generator (seed) draws the CV fold seeds and the tau exploration arms.  Returns the
    trace of the n rounds, the last one (when n - tau is odd) played as an exploration round."""
    IA = inference_algorithms()
    Dsg = TwoTrackStream(e, setup)
    rec = IA.collect(Dsg, tau, e.n - tau, np.random.default_rng(seed), "cv")
    arms = Dsg.arms + [int(e.exploration_arms[g]) for g in range(Dsg.g, e.n)]      # at most one leftover round
    tr = dict(arm=np.array(arms), reward=np.zeros(e.n), regret=np.zeros(e.n), expected=np.zeros(e.n))
    for g, a in enumerate(arms):
        means = e.means(g)
        tr["reward"][g] = e.reward(g, a, means)
        tr["regret"][g], tr["expected"][g] = means.max() - means[a], means[a]
    played = np.concatenate([rec["regret_ex"], np.ravel(np.column_stack(rec["regret"]))])   # collect's own count
    assert np.allclose(played, tr["regret"][:len(played)], rtol=0, atol=1e-9), "the stream and the collection disagree"
    tr["exploration"] = (np.arange(e.n) < tau) | (np.arange(e.n) >= Dsg.g)
    return tr, rec


def run_task(tk, data, setup, grid):
    setup = world_setup(setup, tk["block"])
    n = tk["horizon"]
    pairs = mind.draw_rounds(data, K, n, tk["seeds"]["stream"])
    radii = {"Theta": float(setup["radius_Theta"]), "main": tuple(setup["radii_main"])}
    truth = (setup["truth_gamma"], setup["truth_Theta"])

    def env(scale, spec=MAIN):
        return mind.MindEnv(data, pairs, spec, tk["seeds"]["env"], radii, truth, scale=scale)
    rows, curves = [], {}
    arm = tk["arm"]
    if is_alg1(arm):                                         # our method / the matched arm, constant by CV
        e = env("design")
        out = A.run_algorithm1(e, tk["tau"], "cv")
        tr = out["trace"]
        tr["expected"] = expected_of(e, tr)
        rows.append(row("Proposed" if arm.startswith("A_") else "Matched", arm, tk["tau"], tr, c_tau=tk["c_tau"],
                        score=out["score"], sigma_hat=out["sigma_hat"], lambda_n=float(out["levels"][-1]),
                        rank_n=int(tr["fitted_rank"][-1]), theta_error_n=float(tr["theta_error"][-1]),
                        gamma_n=out["Gamma"].tolist(), cv=json.dumps(out["cv"])))
        curves[arm] = curve(tr, grid)
    elif arm == "two_track":                                 # the inference's data collection (Step 1)
        e = env("design")
        tr, rec = two_track(e, setup, tk["tau"], tk["seeds"]["solver"])
        rows.append(row("TwoTrack", arm, tk["tau"], tr, score=rec["score"], cv=json.dumps(rec["cv"]),
                        rounds_tracks=int(len(rec["regret"][0]) + len(rec["regret"][1]))))
        curves[arm] = curve(tr, grid)
    elif arm.startswith(("oful", "estr")):                   # a baseline over its multipliers
        Bl.recover = RR.OneRecovery(RR.PACKAGE_RECOVER)        # a fresh store per cell: stage 1 once per T1
        suffix = "_pm1" if coding_of(arm) == "pm1" else ""
        for sp in baseline_specs(setup, tk["rank"], arm, n, tk.get("config")):
            method = sp["method"]
            e = env("unit", "c" if method == "ESTR" else MAIN)         # ESTR sees the covariates c
            tr = Bl.run_baseline(e, sp, tk["seeds"]["solver"] if method == "ESTR" else None)
            tr["expected"] = expected_of(e, tr)
            rows.append(row("ESTR_tau" if arm == "estr_tau" else method + suffix, sp["config_id"] + suffix, sp["T1"], tr,
                            multiplier=sp["multiplier"],
                            coding=coding_of(arm), S_F=sp["S_F"], S2=sp["S2"], **tr.get("bm", {})))
            curves[sp["config_id"] + suffix] = curve(tr, grid)
    elif arm in ("aux_greedy", "cm_greedy"):               # the greedy comparators without the interaction
        if arm == "aux_greedy":                            # c alone, in the ball of its best fit to the truth
            e, blocks, method = env("unit", "c"), ((mind.N_COV, float(setup["main_only_ball"])),), "AuxGreedy"
        else:                                              # c and m, in Algorithm 1's balls for these blocks
            e, blocks, method = env("design"), ((mind.N_COV, radii["main"][0]), (D, radii["main"][1])), "CMGreedy"
        for tau in GREEDY_TAUS:
            tr = mind.run_greedy(e, tau, blocks)
            cid = "%s_tau%03d" % (arm, tau)
            rows.append(row(method, cid, tau, tr))
            curves[cid] = curve(tr, grid)
    else:
        raise ValueError("unknown arm %s" % arm)
    stream = dict(n=n, K=K, pairs_sum=int(pairs.sum()), **env("unit", "c").stream_constants())
    return dict(rows=rows, curves=curves, stream=stream)


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def stamp(tk, setup_sha):
    """What a finished cell must have been made under to be reused (checked by run_mind and collect_mind); a main-block
    competitor cell also records the configuration it ran."""
    bm = (Bl.BM if coding_of(tk["arm"]) == "pm1" else Bl.GLM_BM) if tk["arm"].startswith("estr") else None
    return dict(setup_sha=setup_sha, estr_bm=None if bm is None else dict(bm), package=RR.PACKAGE,
                config=tk.get("config"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "mind_small_prepared_glove.npz"))
    ap.add_argument("--setup", required=True)
    ap.add_argument("--block", default="down4", choices=sorted(BLOCKS))
    ap.add_argument("--best", default=None, help="main block: the down4 block's collected/best.tsv")
    ap.add_argument("--horizon", type=int, default=None, help="default: the block's (40,000; main 80,000)")
    ap.add_argument("--replications", type=int, default=None)
    ap.add_argument("--out", default="mind_regret")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    assert 1 <= a.chunk <= a.chunks
    assert a.block != "main" or a.best, "the main block runs the down4 block's best configurations: give --best"
    setup = dict(np.load(a.setup))
    assert str(setup["package"]) == mind.SETUP_VERSION, "a setup file of setup_mind.py %s is needed" % mind.SETUP_VERSION
    all_tasks = tasks(setup, a.block, a.horizon, a.replications, a.best)
    mine = [t for t in all_tasks if t["key"] == a.only] if a.only else all_tasks[a.chunk - 1::a.chunks]
    if a.only and not mine:
        raise SystemExit("no cell with key %s" % a.only)
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    grid = S.time_grid(all_tasks[0]["horizon"])
    data, setup_sha = None, sha16(a.setup)
    print("run_mind: tasks %d of %d -> %s" % (len(mine), len(all_tasks), a.out), flush=True)
    for tk in mine:
        path = os.path.join(a.out, "cells", tk["key"] + ".json")
        done = json.load(open(path)) if os.path.exists(path) else None
        if done is not None and done["status"] == "ok":
            if done.get("stamp") != stamp(tk, setup_sha):
                raise SystemExit("%s was made under another setup or package version: write to a new --out" % path)
            print("skip", tk["key"], flush=True)
            continue
        if data is None:
            data = mind.load(a.data, D, setup)
            assert str(setup["data_sha"]) == hashlib.sha256(open(a.data, "rb").read()).hexdigest(), \
                "the prepared file is not the one the setup was made from"
        t0 = time.time()
        try:
            value = run_task(tk, data, setup, grid)
            value.update(status="ok", error=None)
        except Exception:
            value = dict(status="error", error=traceback.format_exc(), rows=[], curves={})
        value.update(task=tk, elapsed=time.time() - t0, stamp=stamp(tk, setup_sha), data_sha=str(setup["data_sha"]),
                     **provenance())
        tmp = path + ".partial.%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, path)
        print("%s %s %s (%.0fs)" % (value["status"].upper(), tk["key"],
              ", ".join("%s regret %.1f" % (r["config_id"], r["regret"]) for r in value["rows"][:5])
              or value["error"].strip().split("\n")[-1][:160], value["elapsed"]), flush=True)


if __name__ == "__main__":
    main()
