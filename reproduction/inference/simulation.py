"""Design of the inference experiment: the blocks, the data-generating processes and the seeds.  Nothing in this
module looks at the procedure.

Model.  Round t shows K candidate arms m_{t,a} in R^d1 and one context x_t in R^d2, shared by the arms; the design
row is z = vec(m x'), and the reward follows the canonical GLM y ~ exp(y <z, theta*> - psi(<z, theta*>)) with
theta* = vec(Theta*), Theta* of rank r0.  The target is mu* = <A, theta*> for a fixed loading A.  As in the regret
experiment the model has no main effects, so the collecting policy is the regret experiment's Algorithm 1 in the same
parametrization (p = d1 d2, p_T = r0 (d1 + d2 - r0)).  Only the adaptive-tie design of the appendix (`TieDesign`)
uses the stacked row (m, x, vec(m x')), because its target is a main effect.

Units.  E[m m'] = E[x x'] = I (the `lowrank` regime is uniform on the sphere of radius sqrt(d)), so the linear
predictor has standard deviation ||Theta*||_F = 1, the Gaussian noise sd is rho, and the binomial link scale is
solved from the greedy arm's mean success probability P_BAR.

Regimes.  `lowrank`: m and x uniform on the sphere (the regret experiment's design); `normal`: m, x ~ N(0, I).
Every block except I3 runs both.

Collection.  tau = ceil(C_TAU (d1 + d2) log(n d1 d2)) with the regret experiment's C_TAU = 1/20, and the penalty
constant chosen inside each track by 10-fold cross-validation (algorithms.collect with c_lam = "cv").  I3 keeps its
own 12 exploration rounds and c_lam = 1/16.

Blocks, in total rounds n = tau + N; one collection per replicate to the block's horizon, every smaller n is the same
run stopped at n:
  I1     identity link, rho = 0.2, n from 500 to 10000 (14 points), B = 500;
  I2l    logistic rewards (nb = 1), n from 1000 to 60000 (24 points), B = 500;
  I1d32  identity link, rho = 0.2, d1 = d2 = 32 (p = 1024, p_T = 124), n from 1000 to 10000 (8 points), B = 500;
  I3     the adaptive-tie design (d1 = d2 = 1, K = 2); post-exploration rounds N from 50 to 12800, B = 500.

Targets (`targets`; every ||A_Theta||_F = 1): dense, U J V' / sqrt(d1 d2); entry_max and entry_min, the entries
e_i e_j' with the largest and the smallest tangent projection; rank1, a random rank-one loading of median tangent
projection.  I3 has its single main-effect target.

Arms: Steps 2-4 of Algorithm 2 on the same collections with different choices of n0 (N0_CHOICES below); the
displayed choice is the design-stage split with cap 3/4 (PRIMARY).
"""
import numpy as np

SCAFFOLD = dict(d=16, r=2, K=16, rho=0.2)
REGIMES = ("lowrank", "normal")          # append only: the index enters the seeds

# ---------------------------------------------------------------- the collection's constants
C_TAU = 1 / 20                                 # = regret/simulation.C_TAU: one constant for every study
C_LAM = "cv"                                   # the penalty constant chosen inside each track (algorithms.collect)
TIE_TAU, TIE_C_LAM = 12, 1 / 16                # I3: 12 exploration rounds = 4 p_T (p_T = 3), its own penalty


def constants(block):
    """(c_tau, c_lam, source) of one block; c_tau is None for I3 (TIE_TAU rounds)."""
    if block == "I3":
        return None, TIE_C_LAM, "I3's own: tau = %d, c_lam = 1/16" % TIE_TAU
    return C_TAU, C_LAM, "the regret study's c_tau = 1/20 and CV"


TOTALS = (1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000)
BLOCKS = {  # block: family, nb, grid (n = total rounds; I3: N = post-exploration), B, replicates per task, d
    "I1":  dict(family="gaussian", nb=1,  n=(500,) + TOTALS, B=500, chunk=100),
    "I2l": dict(family="binomial", nb=1,  n=TOTALS + (12000, 14000, 16000, 18000, 20000, 25000, 30000, 35000, 40000,
                                                     50000, 60000), B=500, chunk=2),
    "I3":  dict(family="gaussian", nb=1,  N=(50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600,
                                            2400, 3200, 4800, 6400, 12800), B=500, chunk=500),
    "I1d32": dict(family="gaussian", nb=1, d=32, n=(1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000), B=500, chunk=10),
}
DEBUG_BLOCKS = {  # the same runners on a few replicates: exercises the pipeline, claims nothing
    "I1":  dict(family="gaussian", nb=1,  n=(500, 1000), B=2, chunk=2),
    "I2l": dict(family="binomial", nb=1,  n=(1500, 2500), B=1, chunk=1),
    "I3":  dict(family="gaussian", nb=1,  N=(50, 100, 200, 400), B=4, chunk=4),
    "I1d32": dict(family="gaussian", nb=1, d=32, n=(1000, 2000), B=1, chunk=1),
}
# The arms: Steps 2-4 of Algorithm 2 on the same collections (paired comparisons), with different choices of n0, the
# number of each track's rounds that go to the nuisance fit (the rest go to the predictable tangent refit):
#   design_75      the displayed choice (PRIMARY): the design-stage split with cap 3/4
#                  (algorithms.design_split, a function of the design only: the same split in every replicate and both
#                  tracks), the equal split lengthened when half a track is short of the burn-in
#   info_rule      the information rule (algorithms.n0_rule; kappa0 averaged over all coordinates of S_o, c0 = 1/3)
#   share_10 ... share_90   the fixed shares n0 = floor(s n_tr), s = 0.1, 0.3, 0.5, 0.7, 0.9
#   no_refit       n0 = n_tr (s = 1): the nuisance fit uses the whole track and the tangent coordinates are not
#                  re-estimated, so the penalty's shrinkage stays in the plug-in term; outside the theorem (no refit),
#                  it shows what the refit buys
#   design_90      the design-stage split with cap 0.9
# Every choice is judged by the same standard and reported.  Every choice except no_refit meets the condition of
# The theorem on asymptotic normality: n0 sigma(S_o)-measurable with n0 ~ n_tr - n0 ~ n_tr (deterministic constants); the shares
# and design-stage splits are deterministic, the information rule uses S_o only and stays within [n_tr/4, 3 n_tr/4].
# The appendix "unknown r0" adds two arms, rank_gic and rank_aic: design_75's n0, with
# each fold's truncation rank chosen on that fold's pilot data by an information criterion (algorithms.rank_rule; GIC's
# price log(s)/s, Lee, Sun and Liu 2025, or the AIC-type price 1/s); n0 and tau keep the declared rank.
N0_CHOICES = (("design_75", "design 0.75"), ("info_rule", "rule"), ("share_10", "0.1"), ("share_30", "0.3"),
              ("share_50", "0.5"), ("share_70", "0.7"), ("share_90", "0.9"), ("no_refit", "whole track"),
              ("design_90", "design 0.9"))
PRIMARY = "design_75"                    # the displayed choice of n0 (figures, the MIND paper figures and checks)
RANK_ARMS = {"rank_gic": "gic", "rank_aic": "aic"}


def arms(block):
    """The arms run on one block's collections: I3 (d1 = d2 = r0 = 1, nothing to choose) the design-stage split (with
    kappa_D = 1 the equal split), the information rule and no_refit."""
    if block == "I3":
        return (("design_75", "design 0.75"), ("info_rule", "rule"), ("no_refit", "whole track"))
    return N0_CHOICES + (("rank_gic", "design 0.75"), ("rank_aic", "design 0.75"))


def tasks(blocks=BLOCKS):
    """One task per (block, regime, replicate range); I3 has one regime, 'tie'.  Replicates are
    numbered from the block's `first` (1 by default)."""
    out = []
    for block, spec in blocks.items():
        first = spec.get("first", 1)
        for regime in (("tie",) if block == "I3" else REGIMES):
            for lo in range(first, first + spec["B"], spec["chunk"]):
                hi = min(first + spec["B"] - 1, lo + spec["chunk"] - 1)
                out.append(dict(block=block, regime=regime, lo=lo, hi=hi, **spec,
                                key="%s_%s_r%04d-%04d" % (block, regime, lo, hi)))
    return out


SEED_BASE = {"I1": 84_000_000, "I2l": 94_000_000, "I1d32": 204_000_000}   # I3: 74,000,000 + 100,000 rep + N


def seed(block, regime, rep, N=None):
    """The replicate's seed (the runners' convention: one namespace per block; I1d32's base lies above every seed of
    the other blocks, I3's included)."""
    if block == "I3":
        return 74_000_000 + 100_000 * rep + N
    extra = 10 * BLOCKS[block]["nb"] if block == "I2l" else 0
    return SEED_BASE[block] + 1000 * rep + REGIMES.index(regime) + 1 + extra


# ---------------------------------------------------------------- the truth
def truth(d1, d2, rank):
    """Theta* = U[:, :r] V[:, :r]' / sqrt(r) (||Theta*||_F = 1, r equal singular values), the
    same construction and seed as the regret group, and the dense target
    U J V' / sqrt(d1 d2) (||.||_F = 1) whose tangent projection has squared norm r(d1+d2-r)/(d1 d2)."""
    rng = np.random.default_rng(240900 + 100 * d1 + d2)
    U, _ = np.linalg.qr(rng.standard_normal((d1, d1)))
    V, _ = np.linalg.qr(rng.standard_normal((d2, d2)))
    Theta = U[:, :rank] @ V[:, :rank].T / np.sqrt(rank)
    target = U @ np.full((d1, d2), 1.0 / np.sqrt(d1 * d2)) @ V.T
    return Theta, target


TARGET_NAMES = ("dense", "entry_max", "entry_min", "rank1")       # the order of Design.As


def targets(d1, d2, rank):
    """The four loadings, a 4 x (d1 d2) array of column-major vec(A_Theta), every ||A_Theta||_F = 1.

    dense      U J V' / sqrt(d1 d2), J the all-ones matrix (= `truth`'s target);
    entry_max  e_i e_j' with the largest ||P_T(e_i e_j')||^2 = a_i + b_j - a_i b_j, a_i = ||U' e_i||^2,
               b_j = ||V' e_j||^2 (the first in column-major order if tied);
    entry_min  the same with the smallest;
    rank1      m0 x0' / (||m0|| ||x0||): draw k of default_rng(20261019).standard_normal((2001, 2, d)) gives
               m0 = [k, 0], x0 = [k, 1]; the draw whose ||P_T A|| is the median of the 2001.
    U, V are the left and right singular vectors of Theta* (rank r)."""
    assert d1 == d2, "rank1 draws (m0, x0) from one d"
    Theta, dense = truth(d1, d2, rank)
    U, _, Vt = np.linalg.svd(Theta)
    U, V = U[:, :rank], Vt[:rank].T
    a, b = np.sum(U ** 2, axis=1), np.sum(V ** 2, axis=1)
    tan2 = vec(a[:, None] + b[None, :] - a[:, None] * b[None, :])      # ||P_T(e_i e_j')||^2 at index i + j d1
    out = np.zeros((4, d1 * d2))
    out[0] = vec(dense)
    out[1, int(np.argmax(tan2))] = 1.0
    out[2, int(np.argmin(tan2))] = 1.0
    g = np.random.default_rng(20261019).standard_normal((2001, 2, d1))
    cand = [np.outer(m, x) / (np.linalg.norm(m) * np.linalg.norm(x)) for m, x in g]
    tn = np.array([np.linalg.norm(tangent_project(vec(C), U, V, d1, d2)) for C in cand])
    out[3] = vec(cand[int(np.argsort(tn)[1000])])
    return out


def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def vec(M):
    """Column-major vectorization (the convention of the stacked design)."""
    return M.reshape(-1, order="F")


def tangent_project(v, U, V, d1, d2):
    """Projection onto the tangent space T_Theta of vec(Theta) (and, for a stacked vector
    with main effects in front, R^d1 x R^d2 x T_Theta: the main effects are kept)."""
    p0 = v.size - d1 * d2
    out = v.copy()
    M = v[p0:].reshape(d1, d2, order="F")
    out[p0:] = vec(U @ (U.T @ M) + (M @ V) @ V.T - U @ ((U.T @ M) @ V) @ V.T)
    return out


# ---------------------------------------------------------------- the designs
P_BAR = 0.80          # binomial: mean success probability at the greedy arm (= regret/simulation.P_BAR)
_SCALE = {}


def link_scale(kind, d, r, K, p_bar=P_BAR, rounds=50000, seed=917):
    """The binomial link scale amp: the greedy arm's mean success probability equals p_bar.

    The linear predictor m' Theta* x has standard deviation ||Theta*||_F = 1 in every
    regime, and the greedy arm is the largest of the K candidates of one round -- which
    share that round's context -- so its operating point is E[sigmoid(amp * max_k eta_k)];
    solve for amp.  Monotone in amp, so bisection; frozen seed and round count, memoized
    per (kind, d, r, K, p_bar)."""
    key = (kind, d, r, K, round(p_bar, 6))
    if key not in _SCALE:
        g = np.random.default_rng(seed)
        Theta, _ = truth(d, d, r)
        top = np.zeros(rounds)
        for lo_i in range(0, rounds, 5000):                  # in chunks: K arms per context
            hi_i = min(rounds, lo_i + 5000)
            m = covariates(g, (hi_i - lo_i) * K, d, kind).reshape(hi_i - lo_i, K, d)
            x = covariates(g, hi_i - lo_i, d, kind)
            top[lo_i:hi_i] = np.einsum("gkj,gj->gk", m, x @ Theta.T).max(axis=1)
        lo, hi = 1e-4, 50.0                                # amp is 1.04-1.06
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(np.mean(sigmoid(mid * top))) < p_bar:
                lo = mid
            else:
                hi = mid
        _SCALE[key] = 0.5 * (lo + hi)
    return _SCALE[key]


def covariates(rng, n, d, kind):
    """n draws of the regime's covariate law (second moment I): uniform on the sphere of radius sqrt(d) ("lowrank")
    or N(0, I) ("normal")."""
    z = rng.standard_normal((n, d))
    if kind == "lowrank":
        return np.sqrt(d) * z / np.linalg.norm(z, axis=1, keepdims=True)
    if kind == "normal":
        return z
    raise ValueError("unknown covariate law %r" % kind)


class Design:
    """One regime on the scaffold, Gaussian or binomial rewards; interaction-only.

    The truth is theta* = vec(Theta*) with Theta* of rank r and ||Theta*||_F = 1, and the
    target loadings are As = `targets` (four rows, names `target_names`; A = As[0] is the dense target of `truth`);
    the design row is z = vec(m x'), d = d1 d2 coordinates.  The regimes differ only in the covariate law, all with
    E[m m'] = I:

    lowrank:   m, x uniform on the sphere of radius sqrt(d) (the regret cell's design).
    normal:    m, x ~ N(0, I), unbounded                    (Goldenshluger-Zeevi geometry).

    Gaussian rewards: eps ~ N(0, rho^2), rho the noise-to-signal level (signal sd 1).
    Binomial rewards: y ~ Binomial(nb, sigmoid(amp * eta)) with amp from `link_scale`; the
    residual scale is 0.5 sqrt(nb) and the domain C_R is the Frobenius ball of radius
    2 ||amp Theta*||_F.

    kappa_D, the planning value of the design-stage split (algorithms.design_split), is fixed by the design: 1 under
    the identity link (E[z z'] = I), and nb / 4 under the logit link (no intercept, so the mean response under
    uniform exploration is 1/2 by symmetry, and psi'' = nb p (1 - p) = nb / 4 there).
    """

    def __init__(self, kind, family="gaussian", nb=1, d=SCAFFOLD["d"], r=SCAFFOLD["r"],
                 K=SCAFFOLD["K"], rho=SCAFFOLD["rho"]):
        self.kind = self.regime = kind
        self.family, self.nb = family, int(nb)
        self.d1 = self.d2 = d
        self.r0, self.K, self.d = r, K, d * d
        self.sigma_eps = rho                 # signal sd = ||Theta*||_F = 1, so rho = noise / signal
        Theta, target = truth(d, d, r)
        U, _, Vt = np.linalg.svd(Theta)
        self.Theta, self.Us, self.Vs = Theta, U[:, :r], Vt[:r].T
        self.theta = vec(Theta)
        self.As, self.target_names = targets(d, d, r), TARGET_NAMES
        self.A = self.As[0]                                       # = vec(target): the dense target
        self.radius = None
        self.amp = 1.0
        if family == "binomial":                                 # the link scale: target p_bar
            self.amp = link_scale(kind, d, r, K)
            self.theta = self.amp * self.theta
            self.Theta = self.amp * Theta
            self.sigma_eps = 0.5 * np.sqrt(self.nb)
            self.radius = (None, None, 2.0 * np.linalg.norm(self.Theta))   # (R_m, R_x, R_Theta)
        self.mu = float(self.A @ self.theta)
        self.mus = self.As @ self.theta                           # mu* of every target
        self.kappa_D = 1.0 if family == "gaussian" else 0.25 * self.nb

    def draw_arms(self, rng):
        return covariates(rng, self.K, self.d1, self.kind)

    def context(self, rng):
        return covariates(rng, 1, self.d2, self.kind)[0]

    def reward(self, rng, eta):
        if self.family == "gaussian":
            return eta + rng.standard_normal() * self.sigma_eps
        return float(rng.binomial(self.nb, sigmoid(eta)))


class TieDesign:
    """The appendix's adaptive-tie comparison: d1 = d2 = 1, K = 2, theta* = (0, 0, 1/2),
    A = (1, 0, 0), N(0, 1) noise; the joint law of (x, m1, m2) has five atoms (x takes three values).  kappa_D = 1, the
    identity link's planning value, so the design-stage split is the equal split (its burn-in, log n rounds,
    is below half a track at every N of the grid)."""

    def __init__(self):
        self.kind = self.regime = "tie"
        self.family, self.nb = "gaussian", 1
        self.d1 = self.d2 = self.r0 = 1
        self.K, self.d = 2, 3
        self.sigma_eps = 1.0
        self.Us = self.Vs = np.ones((1, 1))
        self.Theta = np.array([[0.5]])
        self.theta = np.array([0.0, 0.0, 0.5])
        self.A = np.array([1.0, 0.0, 0.0])
        self.As, self.target_names = self.A[None], ("main_effect",)
        self.mu = 0.0
        self.mus = np.zeros(1)
        self.radius = None
        self.kappa_D = 1.0
        self._x = None

    def draw_arms(self, rng):
        u = rng.random()
        C = 1.0 if rng.random() < 0.5 else -1.0
        if u < 0.75:
            x, m1, m2 = 0.0, 2.0, 0.0
        elif u < 0.875:
            x, m1, m2 = 1.0, C, C - 1.0
        else:
            x, m1, m2 = -1.0, C + 1.0, C
        self._x = np.array([x])
        return np.array([[m1], [m2]])

    def context(self, rng):
        return self._x                         # drawn jointly with the arms

    def reward(self, rng, eta):
        return eta + rng.standard_normal() * self.sigma_eps


def make_design(block, regime, blocks=BLOCKS):
    if regime == "tie":
        return TieDesign()
    spec = blocks[block]
    return Design(regime, spec["family"], spec["nb"], d=spec.get("d", SCAFFOLD["d"]))
