"""Design of the regret experiment: the case grid, seeds, environments and rewards.

Every cell is a function of (case, replication): the same pair regenerates the same environment, and all arms of a
cell are scored on it.  Round t shows K candidate arms m_{t,1..K} in R^d1 and one context x_t in R^d2, drawn with
E[m m'] = I from one of two laws:

    sphere    uniform on the sphere of radius sqrt(d) (bounded: every method's assumptions hold);
    normal    N(0, I_d), untruncated (sub-Gaussian: our assumptions hold, the baselines' bounded-arm ones do not).

Rewards: Gaussian y = <m_a, Theta* x_t> + eps, eps ~ N(0, rho^2), or binomial y ~ Bin(nb, sigmoid(amp <m_a, Theta* x_t>)),
with Theta* of rank r and ||Theta*||_F = 1 (tier "approx": approximately low rank, singular values decaying like j^-2
and j^-1.5, see `truth`).  The linear predictor
has standard deviation 1, so rho is the noise-to-signal ratio and regret is in signal units; amp is solved so that the
greedy arm's mean success probability is P_BAR (`link_scale`).  The baselines receive their constants converted to
these units (baselines.py, Units; case["arm_scale"] = d).
"""
import numpy as np

# ---------------------------------------------------------------- the grid
HORIZON = 10000
RHO_LEVELS = (0.2, 0.5, 1.0)          # Gaussian family
NB_LEVELS = (1,)                      # binomial family: nb = 1, where the baselines' own binary treatment applies
DESIGNS = ("sphere", "normal")
P_BAR = 0.80                          # binomial: mean success probability at the greedy arm
TIERS = {                              # tier -> ((d, r) pairs, K values)
    "main": ([(16, 4), (16, 2), (16, 1)], (16,)),        # d = 16, d/r = 4, 8, 16
    "karm": ([(16, 2)], (32, 64)),
    "debug": ([(16, 1)], (16,)),       # the debug arrays: the cheapest main case, at their horizon of 2,000 rounds
    "approx": ([(16, 2)], (16,)),      # approximately low-rank truths (TIER_TRUTH); r = 2 the rank ESTR is given
    "d32": ([(32, 2), (32, 4)], (16,)),
    "scale": ([(16, 2)], (16,)),       # the regret against the horizon n (TIER_HORIZONS)
    "scale_debug": ([(16, 2)], (16,)), # its debug array: two short horizons (its cells are cells of the tier scale)
}
DEFAULT_TIERS = ("main", "karm")       # "approx", "d32" and "scale" run as their own arrays (--tiers)
TIER_TRUTH = {"approx": ("decay2", "decay1.5")}    # a tier's truths (`truth`): singular values ~ j^-2 and j^-1.5
                                                  # every other tier: "exact"
TIER_LEVELS = {"d32": {"gaussian": (0.5,), "binomial": ()},    # a tier's noise levels, if not the family's
               "scale": {"gaussian": (0.5, 1.0), "binomial": ()}, "scale_debug": {"gaussian": (0.5,), "binomial": ()}}
# The tier "scale": the regret at the horizon n for several n, each run tuned to its own n (our
# tau and penalty levels, the competitors' horizon-dependent constants).  One case per horizon, case id suffix
# _n<6 digits>; the seeds are the main grid's at that horizon (`environment_seed`), so its cells at n = 10,000 are
# the main grid's cells of the same case.
TIER_HORIZONS = {"scale": (1000, 2000, 5000, 10000, 20000, 50000, 100000), "scale_debug": (1000, 2000)}
K_BLOCKS = (16, 32, 64)
DR_REF = 8.0          # the d/r above which the low-rank structure is predicted to pay (P3)
DR_LOW = 4.0          # the d/r at or below which OFUL is expected to beat ESTR (P3)
P3_FAMILY = "gaussian"   # P3 is declared for Gaussian rewards; binary rewards are reported

# Our method: tau = ceil(C_TAU (d1 + d2) log(n d1 d2)) with C_TAU fixed in every setting (set before any run of this
# design) and the penalty constant chosen inside each run by
# CV; the other values of C_TAU_RUN (1/40 and 1/10) are the sensitivity analysis.  The competitors run on
# the authors' released grids, and every configuration is reported.
C_TAU = 1 / 20
C_TAU_GRID = (1 / 80, 1 / 40, 1 / 20, 1 / 10, 1 / 5)     # the names: arm A_t<i> runs c_tau = C_TAU_GRID[i]
C_TAU_RUN = (1 / 40, 1 / 20, 1 / 10)                     # the values run: half, C_TAU, double
T1_STEPS = 5                          # ESTR's released T1 grid, C_T1 = 10^(k/4), k = 0..4 (baselines.specs)

# The arms of a (case, replication); each arm is one cell.
#   A_t<i>        our method at C_TAU_GRID[i], for the values of C_TAU_RUN: A_t1, A_t2 (= C_TAU, the reported
#                 configuration) and A_t3; c by CV
#   M_k<k>        the matched arm: our learner (c by CV) with tau = ESTR's k-th T1
#   oful          OFUL over its 5 multipliers;            estr_k<k>      ESTR at its k-th T1, 5 multipliers
#                 (binary rewards: our GLM versions, coding "glm", the main binary competitors)
#   estr_tau      ESTR with our exploration length, T1 = tau of our rule at C_TAU, 5 multipliers (is our
#                 advantage more than the shorter exploration?  prediction P7)
#   oful_pm1      binary rewards only: the authors' own treatment (linear model on 2y - 1), the supplement;
#   estr_pm1_k<k> the same for ESTR
ARMS = (tuple("A_t%d" % C_TAU_GRID.index(c) for c in C_TAU_RUN) + tuple("M_k%d" % k for k in range(T1_STEPS))
        + ("oful",) + tuple("estr_k%d" % k for k in range(T1_STEPS)) + ("estr_tau",))
PM1_ARMS = ("oful_pm1",) + tuple("estr_pm1_k%d" % k for k in range(T1_STEPS))
PM1_FAMILY = "binomial"               # the family in which the authors' 2y - 1 versions also run
# The tiers "approx", "d32" and "scale": our method at C_TAU, OFUL and ESTR at its five T1; in "approx" under Gaussian
# rewards also ESTR given rank 4 (estr_r4_k<k>).
_ESTR = tuple("estr_k%d" % k for k in range(T1_STEPS))
TIER_ARMS = {"approx": {"gaussian": ("A_t2", "oful") + _ESTR + tuple("estr_r4_k%d" % k for k in range(T1_STEPS)),
                        "binomial": ("A_t2", "oful") + _ESTR},
             "d32": {"gaussian": ("A_t2", "oful") + _ESTR, "binomial": ()},
             "scale": {"gaussian": ("A_t2", "oful") + _ESTR, "binomial": ()},
             "scale_debug": {"gaussian": ("A_t2", "oful") + _ESTR, "binomial": ()}}


def arms_of(case):
    """The arms of one case: its tier's (TIER_ARMS), or ARMS (and PM1_ARMS for binary rewards)."""
    if case["tier"] in TIER_ARMS:
        return TIER_ARMS[case["tier"]][case["family"]]
    return ARMS + (PM1_ARMS if case["family"] == PM1_FAMILY else ())


def estr_rank(case, arm):
    """The rank ESTR is given in a cell: 4 in the estr_r4 arms, the case's rank otherwise."""
    return 4 if arm.startswith("estr_r4") else case["rank"]

# B = 50 replications per design, numbered from 601.
PLAN = {"sphere": (("main", "karm"), 50), "normal": (("main", "karm"), 50)}
PHASE = "evaluation"                  # the seed namespace (seed_base, solver_seeds) of the single phase
FIRST_REPLICATION = 601


def plan(tiers=None, designs=None, replications=None):
    """The (design, tiers, replications) entries of the study; any argument given overrides
    the table for every design (which is how the debug arrays run one small plan)."""
    out = []
    for design, (design_tiers, B) in PLAN.items():
        if designs is not None and design not in designs:
            continue
        out.append((design, tuple(tiers) if tiers else design_tiers, replications or B))
    return out

# Stable case indices: the position of (design, d, r, level) on fixed master lists,
# so adding or removing cases never changes another case's seeds.  Extend the
# master lists only by appending; never remove an entry.
MASTER_DESIGNS = ("sphere", "gaussian", "normal")      # "gaussian" (retired) keeps the seeds of "normal" in place
MASTER_D = (8, 12, 16, 24, 32)                         # d = 8, 12, 24: no study runs them; they keep the seeds in place
MASTER_R = (1, 2, 3, 4)
MASTER_LEVELS = {"gaussian": (0.1, 0.2, 0.5, 1.0), "binomial": (1, 4, 10, 16)}


def case_index(design, d, r, level, family):
    levels = MASTER_LEVELS[family]
    i_g, i_d = MASTER_DESIGNS.index(design), MASTER_D.index(d)
    i_r, i_l = MASTER_R.index(r), levels.index(level)
    return ((i_g * len(MASTER_D) + i_d) * len(MASTER_R) + i_r) * len(levels) + i_l + 1   # 1..240 (environment_seed)


def cases(family="gaussian", tiers=DEFAULT_TIERS, levels=None, designs=DESIGNS):
    """The case table of one family: a list of dicts, ordered by (design, K, rank, d, level, truth).  A tier with its
    own truths (TIER_TRUTH) has one case per truth, with the truth's name as the suffix of its case ids ("_decay2") and
    the seed namespace "approx" (`seed_phase`): its truths share the streams; a tier with its own noise levels
    (TIER_LEVELS) keeps only those of `levels`; a tier with its own horizons (TIER_HORIZONS) has one case per horizon,
    with the key `horizon` and the suffix "_n<horizon>"."""
    if levels is None:
        levels = NB_LEVELS if family == "binomial" else RHO_LEVELS
    out = {}
    for design in designs:
        for tier in tiers:
            pairs, Ks = TIERS[tier]
            tier_levels = [v for v in levels if tier not in TIER_LEVELS or v in TIER_LEVELS[tier][family]]
            for kind in TIER_TRUTH.get(tier, ("exact",)):
                for K in Ks:
                    for d, r in pairs:
                        for level in tier_levels:
                            tag = design[:3]
                            if family == "binomial":
                                case_id = "%s_d%02d_r%d_K%02d_nb%02d" % (tag, d, r, K, level)
                                noise_sd, nb = 0.5 * float(np.sqrt(level)), int(level)
                                amp = link_scale(design, d, r, K, kind=kind)
                            else:
                                case_id = "%s_d%02d_r%d_K%02d_rho%03d" % (tag, d, r, K, round(level * 100))
                                noise_sd, nb, amp = float(level) * signal_sd(design, d), 1, 1.0      # sigma = rho
                            suffix = "" if kind == "exact" else "_" + kind
                            case_id += suffix
                            if case_id in out:
                                continue
                            Theta = truth(d, d, r, kind)
                            case = dict(
                                case_id=case_id, tier=tier, design=design, d1=d, d2=d, rank=r, K=K,
                                family=family, noise_level=level, noise_sd=noise_sd,
                                d_over_r=d / r, nb=nb, amp=amp,
                                report_scale=1.0 / signal_sd(design, d) if family == "gaussian" else 1.0,
                                arm_scale=float(d),          # ||vec(m x')|| relative to unit-norm arms (baselines.py)
                                case_index=case_index(design, d, r, level, family),
                                group="%s_d%d_r%d_K%d%s" % (tag, d, r, K, suffix),
                                truth=kind, seed_phase=PHASE if kind == "exact" else "approx",
                                truth_fro=float(np.linalg.norm(Theta)), truth_op=float(np.linalg.norm(Theta, 2)))
                            for n in TIER_HORIZONS.get(tier, (None,)):          # tier scale: one case per horizon
                                c = case if n is None else dict(case, case_id="%s_n%06d" % (case_id, n), horizon=n)
                                out[c["case_id"]] = c
    return sorted(out.values(), key=lambda c: (c["design"], c["K"], c["rank"], c["d1"], c["noise_level"], c["truth"],
                                               c.get("horizon", 0)))


# ---------------------------------------------------------------- seeds
SEED_BASES = {"development": 123_000_000, "evaluation": 133_000_000,
              "approx": 1_133_000_000}       # the approximately low-rank tier
SOLVER_BASES = {"development": 760_000_000, "evaluation": 800_000_000, "approx": 1_800_000_000}


def seed_base(phase, K):
    block = K_BLOCKS.index(K)
    return SEED_BASES[phase] + block * 3_000_000


def environment_seed(n, case_idx, replication, phase, K):
    """One stream per (horizon, case, replication, phase, K), on fixed arithmetic coordinates
    so that adding designs, cases or replications never moves an existing environment."""
    # case_idx <= 240 keeps case_idx * 2000 + replication below the 500,000 of one horizon step
    assert n % 100 == 0 and n <= 100_000 and case_idx <= 240 and replication <= 1000
    seed = seed_base(phase, K) + (n // 100) * 500_000 + case_idx * 2000 + replication
    assert 0 < seed < 2 ** 31 - 1, "seed overflow: use a smaller horizon or seed base"
    return seed


def solver_seeds(case_idx, replication, phase, K):
    """Five seeds for the baselines' Burer-Monteiro starts, in their own namespace."""
    base = SOLVER_BASES[phase] + K_BLOCKS.index(K) * 10_000_000
    env_index = (case_idx - 1) * 1000 + replication
    return [base + (env_index - 1) * 5 + j for j in range(1, 6)]


# ---------------------------------------------------------------- the environment
def truth(d1, d2, rank, kind="exact"):
    """Theta* = U[:, :r] V[:, :r]' / sqrt(r) (rank r, ||Theta*||_F = 1), fixed per (d1, d2).  kind "decay<a>" (tier
    approx): U diag(s) V' with s_j = j^-a, j = 1, ..., min(d1, d2), rescaled to unit Frobenius norm, the same U and V;
    a = 1/q is the decay of the paper's Schatten-q ball, sum_j s_j^q <= L^q, which gives s_j <= L j^(-1/q):
    "decay2" is the profile of q = 1/2, "decay1.5" that of q = 2/3."""
    rng = np.random.default_rng(240900 + 100 * d1 + d2)
    U, _ = np.linalg.qr(rng.standard_normal((d1, d1)))
    V, _ = np.linalg.qr(rng.standard_normal((d2, d2)))
    if kind == "exact":
        return U[:, :rank] @ V[:, :rank].T / np.sqrt(rank)
    assert kind.startswith("decay")
    k = min(d1, d2)
    s = np.arange(1, k + 1) ** -float(kind[len("decay"):])
    return (U[:, :k] * (s / np.linalg.norm(s))) @ V[:, :k].T


def covariates(rng, shape, design):
    """Arms and contexts with E[m m'] = I_d: `sphere` uniform on the sphere of radius sqrt(d), `normal` N(0, I_d)
    (drawn on the unit scale and multiplied by sqrt(d))."""
    d = shape[-1]
    z = rng.standard_normal(shape)
    if design == "sphere":
        unit = z / np.linalg.norm(z, axis=-1, keepdims=True)
    elif design == "normal":
        unit = z / np.sqrt(d)
    else:
        raise ValueError("unknown design %s" % design)
    return unit * np.sqrt(d)


def signal_sd(design, d):
    """Standard deviation of the linear predictor m' Theta* x (||Theta*||_F = 1).  Both laws
    are rotation invariant with E[m m'] = I_d, so the sd is E||m||^2 / d = 1."""
    return 1.0


def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


_SCALE = {}


def link_scale(design, d, r, K, p_bar=P_BAR, rounds=50_000, seed=917, kind="exact"):
    """The binomial link scale amp: the greedy arm's mean success probability equals p_bar.

    The linear predictor m' Theta* x has standard deviation signal_sd(design, d), and the
    greedy arm is the largest of the K candidates of one round -- which
    share that round's context -- so its operating point is E[sigmoid(amp * max_k eta_k)];
    solve for amp.  Monotone in amp, so bisection; frozen seed and round count, memoized
    per (design, d, r, K, p_bar, kind of truth)."""
    key = (design, d, r, K, round(p_bar, 6)) + (() if kind == "exact" else (kind,))
    if key not in _SCALE:
        g = np.random.default_rng(seed)
        Theta = truth(d, d, r, kind)
        top = np.zeros(rounds)
        for lo_i in range(0, rounds, 5000):                  # in chunks: K arms per context
            hi_i = min(rounds, lo_i + 5000)
            m = covariates(g, (hi_i - lo_i, K, d), design)
            x = covariates(g, (hi_i - lo_i, d), design)
            top[lo_i:hi_i] = np.einsum("gkj,gj->gk", m, x @ Theta.T).max(axis=1)
        lo, hi = 1e-4, 500.0                               # amp is 0.78-1.41 in these units
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(np.mean(sigmoid(mid * top))) < p_bar:
                lo = mid
            else:
                hi = mid
        _SCALE[key] = 0.5 * (lo + hi)
    return _SCALE[key]


class Environment:
    """One (case, replication): contexts, candidate arms, noise, exploration arms, truth."""

    def __init__(self, n, case, seed):
        rng = np.random.default_rng(seed)
        self.n, self.K, self.d1, self.d2 = n, case["K"], case["d1"], case["d2"]
        self.family, self.nb, self.amp = case["family"], case["nb"], case["amp"]
        self.arm_scale = case["arm_scale"]                     # features d times the unit-norm ones,
        self.reward_scale = self.arm_scale if self.family == "gaussian" else 1.0    # rewards d (Gaussian) or 1 times
        self.design, self.seed = case["design"], seed
        self.Theta = truth(self.d1, self.d2, case["rank"], case.get("truth", "exact"))
        self.contexts = covariates(rng, (n, self.d2), self.design)
        self.candidates = covariates(rng, (n, self.K, self.d1), self.design)
        self.noise = case["noise_sd"] * rng.standard_normal(n)
        self.exploration_arms = rng.integers(0, self.K, size=n)
        self.uniforms = rng.uniform(size=(n, self.nb)) if self.family == "binomial" else None

    def means(self, t):
        """True mean reward of the K candidates of round t (probability scale for binomial)."""
        eta = self.candidates[t] @ (self.Theta @ self.contexts[t])
        return eta if self.family == "gaussian" else sigmoid(self.amp * eta)

    def reward(self, t, arm, means):
        """The realized reward of `arm` at round t; `means` is means(t)."""
        if self.family == "gaussian":
            return means[arm] + self.noise[t]
        return float(np.sum(self.uniforms[t] < means[arm]))

    def link_truth(self):
        """The parameter the estimator targets: Theta* (Gaussian) or amp * Theta* (binomial)."""
        return self.Theta if self.family == "gaussian" else self.amp * self.Theta


# ---------------------------------------------------------------- the time grid of the curves
def time_grid(n=HORIZON, m=200):
    """Rounds, shared by every cell, on which cumulative regret is stored: m log-spaced rounds
    (the exploration phase in detail) together with every (n/100)-th round (evenly spaced
    checkpoints for a linear time axis)."""
    g = np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(n), m))).astype(int))
    lin = np.arange(0, n + 1, max(n // 100, 1))[1:]
    return np.unique(np.concatenate([[1], g, lin, [n]]))
