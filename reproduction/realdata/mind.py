"""The MIND environment of the real-data study, the full-feedback fits that build its truth, and the greedy
comparators without the interaction.

Data.  MIND-small (Wu et al., ACL 2020), prepared by prepare_mind_rounds.py: impressions (a user, a time, the
displayed articles with their logged clicks), GloVe title vectors projected on D principal components, and the users'
reading histories.  Kept users: at least 10 encoded history articles.  An impression is eligible for K when its user is
kept and it displays at least K articles.

Features, in the paper's normalization (every block with second moments about I, as the simulations'
E[z z'] = I).  Article vector m: the prepared title vector (arm-weighted PCA to D components, whitened), so
E[m m'] = I over the displayed articles.  User vector x: the mean of the user's last <= 50 history articles in the same
projection, re-whitened on all kept users' eligible rounds and scaled to length sqrt(D) (setup_mind.py), so
E[x x'] is about I.  Main-effect covariates c of a displayed pair: intercept, same-category share (the share of the
user's history in the article's category), log of the impression's size; the last two standardized over the displayed
pairs of all eligible impressions.  The model "cmx" is the paper's GLM with unpenalized main effects:

    logit P(click) = c' gamma_c + m' gamma_m + x' gamma_x + m' Theta x,

and "c" the same without the m and x main effects (the design ESTR augments).  Scales: "design" (Algorithms 1 and 2,
the truth and the fits) uses c, m and x as above; "unit" (the competitors) divides m and x by sqrt(D), the unit-length
arms and contexts the authors' constants are written for (regret/baselines.py, Units), and keeps c standardized.  This
is the regret simulation's conversion (arms sqrt(d) times the unit sphere there, sqrt(D) here) applied to the arms and
contexts; the covariates c are no part of the authors' unit-sphere assumption and keep their standardized scale.

Rounds.  A stream takes n eligible impressions in random order without replacement and K of each impression's
displayed articles uniformly at random.  Rewards are semi-synthetic: a truth (gamma*, Theta*), the rank-r maximum-
likelihood logistic fit of cmx on all labeled pairs of all kept users (setup_mind.py), gives every displayed candidate
a click probability p_tk.  The click of the chosen candidate is 1{U_t < p_t,chosen}, with one uniform U_t per round
shared by every method (common random numbers), and the regret of a round is max_k p_tk - p_t,chosen, the
paper's expected regret.  No method sees the truth.  The inference study also runs in a down-sampled world
(`downsampled`): the click model of the logs after keeping every click and only a share w of the non-clicks.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

N_COV = 3                                   # intercept, same-category share, log impression size
SETUP_VERSION = "2.30"                      # the setup file's format (setup_mind.py and add_world_mind.py write it,
                                            # the runners check it)


def load(path, D, setup=None):
    """The arrays a stream needs (the article vectors whitened: E[m m'] = I).  With the setup file of setup_mind.py:
    only the kept users' impressions are eligible, the user vectors are the setup's (length sqrt(D)), and so is the
    covariates' standardization."""
    f = np.load(path)
    data = {k: f[k] for k in ("imp_user", "imp_size", "c_imp", "c_news", "c_lab", "samecat", "eligible_K5",
                              "eligible_K10", "news_cat")}
    data["D"] = D
    data["M"] = f["M%d" % D].astype(np.float64)
    data["X"] = f["X%d" % D].astype(np.float64)
    data["first"] = np.searchsorted(data["c_imp"], np.arange(len(data["imp_user"])))
    if setup is not None:
        assert int(setup["D"]) == D, "the setup file is for D = %d" % int(setup["D"])
        keep = setup["keep"].astype(bool)
        for K in (5, 10):
            data["eligible_K%d" % K] = data["eligible_K%d" % K] & keep[data["imp_user"]]
        data["X"] = setup["X"].astype(np.float64)
        data["cov_stats"] = {name: tuple(setup["cov_stats_" + name]) for name in ("samecat", "logK")}
    return data


def covariate_stats(data):
    """Mean and sd of the same-category share and of log K over the displayed pairs of the eligible (K >= 5)
    impressions."""
    pairs = np.flatnonzero(data["eligible_K5"][data["c_imp"]])
    logK = np.log(data["imp_size"].astype(np.float64))[data["c_imp"][pairs]]
    sc = data["samecat"][pairs].astype(np.float64)
    return {"samecat": (float(sc.mean()), float(sc.std())), "logK": (float(logK.mean()), float(logK.std()))}


def covariates(data, pairs, main, scale="design"):
    """The main-effect covariates of the displayed pairs `pairs` (an index array of any shape); the last axis of the
    result holds the covariates: c, then (main "cmx") m and x.  scale "design" (m and x as they are) or "unit" (m and
    x divided by sqrt(D): the competitors' unit-length arms and contexts); c is the same in both."""
    s = data["cov_stats"]
    imp = data["c_imp"][pairs]
    cols = [np.ones(pairs.shape),
            (data["samecat"][pairs] - s["samecat"][0]) / s["samecat"][1],
            (np.log(data["imp_size"][imp].astype(np.float64)) - s["logK"][0]) / s["logK"][1]]
    C = np.stack(cols, axis=-1)
    if main == "cmx":
        b = 1.0 if scale == "design" else np.sqrt(data["D"])
        C = np.concatenate([C, data["M"][data["c_news"][pairs]] / b, data["X"][data["imp_user"][imp]] / b], axis=-1)
    return C


def to_unit(gamma, Theta, main, D):
    """Coefficients of the "design" covariates expressed for the "unit" ones (m and x divided by sqrt(D), so their
    coefficients times sqrt(D), and Theta times D); c's coefficients are unchanged."""
    if main == "c":
        return gamma, D * Theta
    return np.r_[gamma[:N_COV], np.sqrt(D) * gamma[N_COV:]], D * Theta


def group_sizes(main, D):
    return {"none": (), "c": (N_COV,), "cmx": (N_COV, D, D)}[main]


def draw_rounds(data, K, n, seed):
    """n rounds, i.i.d. from the empirical distribution of the real rounds: each round one eligible impression drawn
    uniformly WITH replacement, and K of its displayed pairs drawn uniformly without replacement (as the inference
    study's rounds, run_mind_inference.MindDesign.next_round).  Returns n x K pair indices."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(data["eligible_K%d" % K])
    pairs = np.zeros((n, K), dtype=np.int64)
    for t in range(n):
        i = pool[rng.integers(len(pool))]
        pairs[t] = data["first"][i] + rng.choice(int(data["imp_size"][i]), K, replace=False)
    return pairs


class MindEnv:
    """One stream, with the interface of regret/simulation.Environment (n, K, d1, d2, family, nb, candidates,
    contexts, exploration_arms, means, reward, link_truth, arm_scale) plus the main-effect block (main, main_groups)
    and the Frobenius radius."""

    family, nb = "binomial", 1

    def __init__(self, data, pairs, main, seed, radii, truth, scale="design"):
        """main: "cmx", "c" or "none"; radii: {"Theta": R, "main": (R_1, ...)}, the domain of Algorithm 1's estimator
        (design units); truth: (gamma*, Theta*) of the cmx model (design units); scale: "design" (Algorithm 1) or
        "unit" (the competitors: m and x divided by sqrt(D))."""
        rng = np.random.default_rng(seed)
        D = data["D"]
        self.n, self.K = pairs.shape
        self.d1 = self.d2 = D
        self.main_spec, self.seed = main, seed
        imp = data["c_imp"][pairs[:, 0]]
        m = data["M"][data["c_news"][pairs]]                             # n x K x D, design units
        x = data["X"][data["imp_user"][imp]]                             # n x D
        gamma, Theta = truth
        u = np.einsum("tki,ij,tj->tk", m, Theta, x) + covariates(data, pairs, "cmx") @ gamma
        self.prob = expit(u)                                             # n x K click probabilities
        self.Theta_star = Theta
        b = 1.0 if scale == "design" else np.sqrt(D)
        self.candidates, self.contexts = m / b, x / b
        self.arm_scale = D if scale == "design" else 1.0                # Algorithm 1's solver tests: the features
        #   are D times those of unit-length arms and contexts (regret/algorithms.unit_floors), so its iterates are
        #   those of the unit-length data
        self.main = None if main == "none" else covariates(data, pairs, main, scale)
        self.main_groups = list(zip(group_sizes(main, D), radii["main"])) if main != "none" else []
        self.radius = radii["Theta"]
        self.exploration_arms = rng.integers(0, self.K, size=self.n)
        self.uniforms = rng.uniform(size=self.n)                         # one per round, shared by every method

    def means(self, t):
        return self.prob[t]

    def reward(self, t, arm, means):
        return float(self.uniforms[t] < means[arm])

    def link_truth(self):
        """Theta* (Algorithm 1 records ||Theta_hat - Theta*||_F)."""
        return self.Theta_star

    def stream_constants(self):
        """The policy-free quantities of the stream: best_possible = sum_t max_k p_tk (regret + expected =
        best_possible for every policy), uniform = the regret of uniform play, mean_mean = the average click
        probability."""
        mu = self.prob
        return dict(best_possible=float(mu.max(1).sum()), uniform=float(np.sum(mu.max(1) - mu.mean(1))),
                    mean_mean=float(mu.mean()))


# ---------------------------------------------------------------- the down-sampled world (the inference study)
def downsampled(setup, w):
    """The setup with the click model of the logs after keeping every click and each non-click with probability w
    (negative down-sampling, or case-control sampling of the non-clicks).  Under the logistic model that sampling adds
    log(1 / w) to the click log-odds of every pair and leaves the other coefficients unchanged (Anderson 1972; Prentice
    and Pyke 1979): the truth's intercept rises by log(1 / w), Theta* and the other coefficients are the setup file's.
    With the truth change the constants taken from it, by the rules of setup_mind.py: Algorithm 1's balls (twice the
    truth's norms per block) and the design-stage split's planning value kappa_D = p_D (1 - p_D), p_D the click rate
    of the down-sampled logs, p_D / (p_D + w (1 - p_D)).  No target loads on the intercept, so every mu* is the setup
    file's.  Only the inference's keys are changed (the competitors' declared norms are not).  The rounds are not:
    a run in this world keeps every round, as in the main world."""
    out = dict(setup)
    gamma = setup["truth_gamma"].copy()
    gamma[0] += np.log(1 / w)                                            # the intercept column of the rows is 1
    starts = np.cumsum((0,) + group_sizes("cmx", int(setup["D"])))
    out["truth_gamma"] = gamma
    out["radii_main"] = np.array([2 * np.linalg.norm(gamma[a:b]) for a, b in zip(starts[:-1], starts[1:])])
    p_D = float(setup["p_D"])
    out["p_D"] = p_D / (p_D + w * (1 - p_D))
    out["kappa_D"] = out["p_D"] * (1 - out["p_D"])
    assert np.all(setup["target_A"][:, 0] == 0)                         # no target loads on the intercept
    return out


# ---------------------------------------------------------------- the full-feedback fits
def full_feedback(data, main):
    """All labeled pairs of the eligible (K >= 5) impressions, each with weight 1.  Returns the arrays of the fit and
    the counts n_pairs and n_impressions."""
    imp_ok = data["eligible_K5"]
    rows = np.flatnonzero(imp_ok[data["c_imp"]])
    y = data["c_lab"][rows].astype(np.float64)
    return dict(rows=rows, y=y, w=np.ones(len(rows)),
                M=data["M"][data["c_news"][rows]], X=data["X"][data["imp_user"][data["c_imp"][rows]]],
                C=covariates(data, rows, main) if main != "none" else np.zeros((len(rows), 0)),
                n_pairs=len(rows), n_impressions=int(imp_ok.sum()))


def fit_logistic(ff, rank=None, start=None, ridge=1e-2):
    """The logistic fit of c' gamma + m' Theta x on the full-feedback data `ff` (L-BFGS); ff["y"] may hold
    probabilities (the fit to a truth's click probabilities).  rank None: Theta unrestricted, with a small ridge on
    Theta.  rank 0: no interaction.  rank r: the rank-constrained maximum-likelihood fit, Theta = A B' with A, B of r
    columns (a tiny ridge on A and B), started at the truncated SVD of start = (gamma, Theta) (Lee, Sun and Liu 2025:
    "rank-constrained maximum likelihood estimation ... on the full dataset").  Returns (gamma, Theta, total), total
    = the weighted negative log-likelihood without the ridge."""
    M, X, C, y, w = ff["M"], ff["X"], ff["C"], ff["y"], ff["w"]
    q, D = C.shape[1], M.shape[1]

    def nll(eta):
        return np.sum(w * (np.logaddexp(0, eta) - y * eta))

    if rank is None:
        def loss(par):
            g, T = par[:q], par[q:].reshape(D, D)
            eta = C @ g + np.einsum("ij,ij->i", M @ T, X)
            r = w * (expit(eta) - y)
            return nll(eta) + 0.5 * ridge * np.sum(T ** 2), np.r_[C.T @ r, (M.T @ (r[:, None] * X) + ridge * T).ravel()]
        par = minimize(loss, np.zeros(q + D * D), jac=True, method="L-BFGS-B", options={"maxiter": 3000}).x
        gamma, Theta = par[:q], par[q:].reshape(D, D)
    elif rank == 0:
        def loss0(g):
            eta = C @ g
            return nll(eta), C.T @ (w * (expit(eta) - y))
        gamma = minimize(loss0, start[0], jac=True, method="L-BFGS-B", options={"maxiter": 3000}).x
        Theta = np.zeros((D, D))
    else:
        U, s, Vt = np.linalg.svd(start[1])
        A0, B0 = U[:, :rank] * np.sqrt(s[:rank]), Vt[:rank].T * np.sqrt(s[:rank])
        def loss(par):
            g, A, B = par[:q], par[q:q + D * rank].reshape(D, rank), par[q + D * rank:].reshape(D, rank)
            MA, XB = M @ A, X @ B
            eta = C @ g + np.sum(MA * XB, axis=1)
            r = w * (expit(eta) - y)
            value = nll(eta) + 0.5 * ridge * (np.sum(A ** 2) + np.sum(B ** 2))
            return value, np.r_[C.T @ r, (M.T @ (r[:, None] * XB) + ridge * A).ravel(), (X.T @ (r[:, None] * MA) + ridge * B).ravel()]
        par = minimize(loss, np.r_[start[0], A0.ravel(), B0.ravel()], jac=True, method="L-BFGS-B", options={"maxiter": 3000}).x
        gamma = par[:q]
        Theta = par[q:q + D * rank].reshape(D, rank) @ par[q + D * rank:].reshape(D, rank).T
    return gamma, Theta, float(nll(C @ gamma + np.einsum("ij,ij->i", M @ Theta, X)))


def linear_fit(ff, ridge=1e-6, chunk=20000):
    """The weighted least-squares fit of 2y - 1 on (c, vec(m x')) (the model the baselines fit under the authors'
    binary treatment), in design units; ff["y"] may hold probabilities.  Returns (gamma, Theta)."""
    C, M, X, w = ff["C"], ff["M"], ff["X"], ff["w"]
    q, D = C.shape[1], M.shape[1]
    G, h = np.zeros((q + D * D,) * 2), np.zeros(q + D * D)
    for a in range(0, len(w), chunk):
        sl = slice(a, a + chunk)
        Zs = np.hstack([C[sl], (M[sl][:, :, None] * X[sl][:, None, :]).reshape(-1, D * D)])
        G += Zs.T @ (w[sl, None] * Zs)
        h += Zs.T @ (w[sl] * (2 * ff["y"][sl] - 1))
    beta = np.linalg.solve(G + ridge * np.eye(len(h)), h)
    return beta[:q], beta[q:].reshape(D, D)


def declared(gamma, Theta, main, D):
    """The competitors' declared norms (S_F, S2) of design-unit coefficients, in the units of the data they see
    (scale "unit", `to_unit`): S_F = ||(gamma, Theta)|| (OFUL's) and S2 = max(||Theta||_op, ||gamma_c||); ESTR's
    norms are `declared_estr`."""
    g, T = to_unit(gamma, Theta, main, D)
    return float(np.sqrt(np.sum(g ** 2) + np.sum(T ** 2))), float(max(np.linalg.norm(T, 2), np.linalg.norm(g[:N_COV])))


def estr_matrix(gamma, Theta, D):
    """The cmx truth (gamma = (gamma_c, gamma_m, gamma_x), Theta; design units) as the matrix of ESTR's augmented
    bilinear model, in the units ESTR sees (`to_unit`): arm (m, c), context (x, 1, 0, 0) (regret/baselines.py, Main
    effects), so (m, c)' B (x, 1, 0, 0) = c' gamma_c + m' gamma_m + x' gamma_x + m' Theta x with
        B = [[Theta,       gamma_m, 0, 0],
             [gamma_x',    gamma_c1, 0, 0],
             [0,           gamma_c2, 0, 0],
             [0,           gamma_c3, 0, 0]]
    -- the m main effect in the column that meets the context's constant, the x main effect in the row of the
    intercept c1 = 1.  The representation is unique (the zero-padded context coordinates meet nothing)."""
    g, T = to_unit(gamma, Theta, "cmx", D)
    B = np.zeros((D + N_COV, D + N_COV))
    B[:D, :D] = T
    B[:D, D] = g[N_COV:N_COV + D]
    B[D, :D] = g[N_COV + D:]
    B[D:, D] = g[:N_COV]
    return B


def declared_estr(gamma, Theta, D):
    """ESTR's declared norms (S_F, S2) = the Frobenius and spectral norms of its augmented truth `estr_matrix` (the
    authors take S_F and S2 from the true parameter; ESTR sees c, and represents the m and x main effects through the
    augmentation)."""
    B = estr_matrix(gamma, Theta, D)
    return float(np.linalg.norm(B)), float(np.linalg.norm(B, 2))


# ---------------------------------------------------------------- the greedy comparators without the interaction
def run_greedy(env, tau, blocks):
    """A greedy comparator without the interaction: uniform arms for tau rounds, then the arm with the largest fitted
    logit z' gamma_hat, gamma_hat the logistic MLE of the rounds so far on the first columns of env.main with each
    block of coefficients in its ball (`_fit`), refitted every round from the previous fit.  blocks: ((size, radius),
    ...) in the order of the columns of env.main: ((N_COV, R),) the auxiliary covariates c (intercept, same-category
    share, log impression size); ((N_COV, R_c), (D, R_m)) c and the article vector m."""
    p = sum(size for size, _ in blocks)
    C = env.main[:, :, :p]
    Z, ys, k = np.zeros((env.n, p)), np.zeros(env.n), 0
    gamma = np.zeros(p)

    def choose(t):
        nonlocal gamma
        if t >= tau and k >= 1:
            gamma = _fit(Z[:k], ys[:k], gamma, blocks)
            return int(np.argmax(C[t] @ gamma))
        return int(env.exploration_arms[t])

    def record(t, a, y):
        nonlocal k
        Z[k], ys[k] = C[t, a], y
        k += 1

    return _trace(env, choose, record)


def _slack(gamma, blocks):
    """R^2 - |gamma_b|^2 for each block b (blocks as in run_greedy): all >= 0 inside the balls."""
    out, start = [], 0
    for size, radius in blocks:
        block = gamma[start:start + size]
        out.append(radius ** 2 - block @ block)
        start += size
    return np.array(out)


def _fit(Z, y, gamma, blocks, ridge=1e-6, tol=1e-10, gap=1e-9):
    """The logistic MLE of y on Z with each block of coefficients in its ball (blocks as in run_greedy), from gamma
    (the previous round's fit; a ridge of 1e-6 keeps the Newton systems non-singular).  The problem is convex, so:
    1. Newton's method with backtracking; when it converges inside every ball, its limit is the constrained MLE
       (after the first rounds, nearly always: two or three steps from the previous fit).
    2. Otherwise -- a ball binds, or the rounds so far separate the clicks and the MLE does not exist -- the
       log-barrier method from inside the balls (Boyd and Vandenberghe, 2004, section 11.3): Newton on
       f - mu sum_b log(R_b^2 - |gamma_b|^2) for mu = 1, 0.1, 0.01, ..., until the bound (number of blocks) mu on
       the objective's distance to the constrained minimum is below `gap`."""
    d = len(gamma)

    def f(g):
        eta = Z @ g
        return np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * ridge * (g @ g)

    def gradient_hessian(g):
        q = expit(Z @ g)
        return Z.T @ (q - y) + ridge * g, (Z * (q * (1 - q))[:, None]).T @ Z + ridge * np.eye(d)

    g = np.array(gamma, dtype=float)                       # 1. Newton's method
    fg = f(g)
    for _ in range(20):
        grad, H = gradient_hessian(g)
        step = np.linalg.solve(H, grad)
        if np.linalg.norm(step) < tol * max(1.0, np.linalg.norm(g)):          # converged
            if np.all(_slack(g, blocks) >= 0):
                return g
            break
        t, slope = 1.0, grad @ step
        while True:                                        # Armijo, up to f's rounding (near the minimum the
            cand = g - t * step                            # predicted decrease is below it: the full step)
            fc = f(cand)
            if fc <= fg - 1e-4 * t * slope + 1e-12 * abs(fg) or t < 1e-12:
                break
            t /= 2
        g, fg = cand, fc
    g, start = np.array(gamma, dtype=float), 0             # 2. the log-barrier method, from strictly inside
    for size, radius in blocks:
        norm = np.linalg.norm(g[start:start + size])
        if norm > 0.99 * radius:
            g[start:start + size] *= 0.99 * radius / norm
        start += size

    def barrier(x, mu):
        s = _slack(x, blocks)
        return f(x) - mu * np.sum(np.log(s)) if np.all(s > 0) else np.inf

    mu = 1.0
    while True:
        for _ in range(100):                               # centring: Newton on the barrier problem
            grad, H = gradient_hessian(g)
            s, start = _slack(g, blocks), 0
            for b, (size, _) in enumerate(blocks):
                block = g[start:start + size]
                grad[start:start + size] += 2 * mu * block / s[b]
                H[start:start + size, start:start + size] += mu * (2 * np.eye(size) / s[b]
                                                                  + 4 * np.outer(block, block) / s[b] ** 2)
                start += size
            step = np.linalg.solve(H, grad)
            decrement = grad @ step                        # the squared Newton decrement
            if decrement < 2e-12:
                break
            t, F0 = 1.0, barrier(g, mu)
            while barrier(g - t * step, mu) > F0 - 0.25 * t * decrement + 1e-12 * abs(F0) and t > 1e-12:
                t /= 2
            g = g - t * step
        if len(blocks) * mu < gap:
            return g
        mu /= 10


def _trace(env, choose, record=None):
    n = env.n
    tr = dict(arm=np.zeros(n, int), reward=np.zeros(n), regret=np.zeros(n), expected=np.zeros(n))
    for t in range(n):
        means = env.means(t)
        a = choose(t)
        y = env.reward(t, a, means)
        tr["arm"][t], tr["reward"][t], tr["regret"][t], tr["expected"][t] = a, y, means.max() - means[a], means[a]
        if record is not None:
            record(t, a, y)
    tr["cumulative_regret"] = np.cumsum(tr["regret"])
    return tr
