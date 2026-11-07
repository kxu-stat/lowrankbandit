"""Algorithm 2 of the paper: two-track cross-fitted debiased inference for a linear functional mu* = <A, theta*> under
greedy data collection.

  1. the model                 canonical GLM on z = vec(m x') (column-major vec); a design's leading p0 coordinates
                               are unpenalized main effects (the tie design's (m, x); MIND's covariates)
  2. the tangent geometry      projection, basis, rank truncation
  3. the estimator             FISTA (identity link), proximal Newton (logit link), on the domain C_R
  4. the recipe                tau and lambda_s as in Algorithm 1 (c by 10-fold CV inside each track),
                               lambda_ref (C_ref = 1/15), lambda_cal (C_cal = 1), the split n0
  5. Step 1                    collect(): uniform exploration S_o for tau rounds, then two greedy tracks in alternation
  6. Steps 2-4                 fold_estimates() and two_track_inference() / calibrate_loadings(): per fold j the
                               nuisance fit on (S_o, first n0 rounds of track j), rank-r0 truncation, the predictable
                               refit on the rest of track j, then the predictable calibration over the opposite track;
                               the two folds are averaged
  7. unknown r0 (appendix)     rank_rule(): the truncation rank chosen in each fold by an information criterion
Also: cramer_rao() (the Cramer-Rao standard error of the collected design) and the one-step comparator.

A design object D supplies d1, d2, d (= p0 + d1 d2), r0, K, family, nb, theta (the truth, used only for the rewards,
the scoring and the labeled offline diagnostics), radius (R_1, ..., R_G, R_Theta), optionally main_sizes (the
main-effect groups; default (d1, d2)), and either draw_arms / context / reward or next_round / reward_arm (a design
that supplies each round's stacked rows).  D.anchor_at_main_fit takes the pilot score at the main-effects-only fit.
The offline diagnostics also read sigma_eps (identity link) and the truth's frame Us, Vs; design_split reads kappa_D.
"""
from fractions import Fraction
import numpy as np
from simulation import sigmoid, vec, tangent_project


# ================================================================ 1. the model
def psi1(u, family, nb):
    return u if family == "gaussian" else nb * sigmoid(u)


def psi2(u, family, nb):
    if family == "gaussian":
        return np.ones_like(u, dtype=float) if np.ndim(u) else 1.0
    q = sigmoid(u)
    return nb * q * (1.0 - q)


def main_len(D):
    """p0, the number of main-effect coordinates of the design."""
    return D.d - D.d1 * D.d2


def stack(m, x, p0):
    """The design row of one (arm, context): vec(m x'), preceded by (m, x) when p0 > 0."""
    if p0 == 0:
        return vec(np.outer(m, x))
    return np.concatenate([m, x, vec(np.outer(m, x))])


# ================================================================ 2. the tangent geometry
def tangent_basis(U, V, d1, d2, r0, p0):
    """Orthonormal columns spanning vec({U B' + C V'}), preceded by the p0 main-effect
    coordinates when p0 > 0; p_T columns."""
    Uc = np.linalg.qr(np.eye(d1) - U @ U.T)[0][:, : d1 - r0]
    Vc = np.linalg.qr(np.eye(d2) - V @ V.T)[0][:, : d2 - r0]
    cols = []
    for L, R in ((U, V), (U, Vc), (Uc, V)):
        for i in range(L.shape[1]):
            for j in range(R.shape[1]):
                cols.append(vec(np.outer(L[:, i], R[:, j])))
    Qi = np.column_stack(cols)
    Q = np.zeros((p0 + d1 * d2, p0 + Qi.shape[1]))
    Q[:p0, :p0] = np.eye(p0)
    Q[p0:, p0:] = Qi
    return Q


def truncate(theta, d1, d2, r0):
    """Rank-r0 truncation of the interaction block; returns (theta, U, V) of the truncation."""
    p0 = theta.size - d1 * d2
    M = theta[p0:].reshape(d1, d2, order="F")
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    U, s, V = U[:, :r0], s[:r0], Vt[:r0].T
    out = theta.copy()
    out[p0:] = vec((U * s) @ V.T)
    return out, U, V


# ================================================================ 3. the estimator
def prox(u, step, lam, d1, d2, radius=None, sizes=None):
    """Proximal map of step lam ||Theta||_* on the domain C_R: the interaction block's singular values thresholded and
    scaled into the Frobenius ball, each main-effect block projected onto its ball.  radius = (R_1, ..., R_G, R_Theta)
    or None (no domain); sizes = the main-effect groups (None: (d1, d2))."""
    p = u.size - d1 * d2
    M = u[p:].reshape(d1, d2, order="F")
    Uu, sv, Vt = np.linalg.svd(M, full_matrices=False)
    sv = np.maximum(sv - step * lam, 0.0)
    if radius is not None:
        RT = radius[-1]
        nT = np.linalg.norm(sv)
        if nT > RT: sv *= RT / nT
    inter = vec((Uu * sv) @ Vt)
    if p == 0:
        return inter
    if sizes is None:
        sizes = (d1, p - d1)                                    # (m, x)
    blocks, start = [], 0
    for g, size in enumerate(sizes):
        block = u[start:start + size].copy()
        if radius is not None:
            nrm = np.linalg.norm(block)
            if nrm > radius[g]: block *= radius[g] / nrm
        blocks.append(block)
        start += size
    return np.concatenate(blocks + [inter])


def fista(G, b, lam, d1, d2, x0, L=None, iters=400, tol=1e-6, radius=None, sizes=None):
    """min 1/2 x'Gx - b'x + lam ||Theta(x)||_* on C_R by FISTA with adaptive restart."""
    if L is None:
        L = np.linalg.eigvalsh(G)[-1]
    L = max(L, 1e-12)
    x = x0.copy(); y = x.copy(); tk = 1.0
    for _ in range(iters):
        xn = prox(y - (G @ y - b) / L, 1.0 / L, lam, d1, d2, radius, sizes)
        if np.linalg.norm(L * (y - xn)) <= tol * max(1.0, L * np.linalg.norm(xn)):
            return xn
        if np.dot(y - xn, xn - x) > 0:
            tk = 1.0                                             # adaptive restart
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = xn + ((tk - 1) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def proximal_newton(Z, y, lam, d1, d2, x0, nb, radius=None, H=None, LH=None, L_max=None,
                    iters=30, tol=1e-6, inner_iters=1000, inner_tol=1e-8, sizes=None):
    """min (1/s) sum_i [psi(z_i'x) - y_i z_i'x] + lam ||Theta(x)||_* on C_R (logit link, nb trials) by proximal Newton:
    each step minimizes the quadratic model with the curvature-weighted Gram H by `fista` and is accepted by Armijo
    backtracking.  H (possibly passed in lagged) is rebuilt after a backtracked step, when the residual fails to halve,
    or after eight steps.  Returns (x, steps, hit_cap, exact refreshes)."""
    s, p = len(y), x0.size - d1 * d2
    if L_max is None:
        L_max = 0.25 * nb * np.linalg.eigvalsh(Z.T @ Z / s)[-1]
    L_max = max(L_max, 1e-12)

    def nuc(v):
        return float(np.linalg.svd(v[p:].reshape(d1, d2, order="F"), compute_uv=False).sum())

    def objective(v, u):
        return float(np.mean(nb * np.logaddexp(0.0, u) - y * u)) + lam * nuc(v)

    x = x0.copy(); u = Z @ x
    refresh, n_ref, res_old = H is None, 0, np.inf
    for it in range(1, iters + 1):
        g = Z.T @ (nb * sigmoid(u) - y) / s
        xp = prox(x - g / L_max, 1.0 / L_max, lam, d1, d2, radius, sizes)
        res = np.linalg.norm(L_max * (x - xp)) / max(1.0, L_max * np.linalg.norm(xp))
        if res <= tol:
            return xp, it - 1, False, n_ref
        refresh = refresh or res > 0.5 * res_old or it > 8
        res_old = res
        if refresh:
            w = psi2(u, "binomial", nb)
            H = Z.T @ (Z * w[:, None]) / s
            LH = np.linalg.eigvalsh(H)[-1]
            n_ref += 1
        if LH is None:
            LH = np.linalg.eigvalsh(H)[-1]
        v = fista(H, H @ x - g, lam, d1, d2, x, L=LH, iters=inner_iters, tol=max(inner_tol, 0.1 * res), radius=radius,
                  sizes=sizes)
        dlt = v - x
        Fx, nx = objective(x, u), nuc(x)
        model = float(g @ dlt) + lam * (nuc(v) - nx)
        alpha, xn, un = 1.0, v, Z @ v
        while objective(xn, un) > Fx + 1e-4 * alpha * model and alpha > 1e-3:
            alpha *= 0.5
            xn = x + alpha * dlt
            un = Z @ xn
        x, u = xn, un
        refresh = alpha < 1.0
    g = Z.T @ (nb * sigmoid(u) - y) / s
    xp = prox(x - g / L_max, 1.0 / L_max, lam, d1, d2, radius, sizes)
    return xp, iters, True, n_ref


# ================================================================ 4. the recipe
def p_T(d1, d2, r0, p0):
    return p0 + r0 * (d1 + d2 - r0)


def tau_rule(n, d1, d2, c_tau, q=0):
    """Exploration length tau = ceil(c_tau (d1 + d2 + q) log(n d1 d2)), Algorithm 1's rule (n the collection's horizon,
    q the number of unpenalized main-effect coefficients)."""
    return int(np.ceil(c_tau * (d1 + d2 + q) * np.log(n * d1 * d2)))


def main_only_fit(C, y, family, nb, radii, sizes, tol=1e-10, maxit=200000):
    """The pilot fit with Theta = 0: the main-effect coefficients alone, unpenalized on their
    balls (radii None: no balls), by projected gradient with step 1 / L, L = sup psi'' x top
    eigenvalue of C'C / s.  The same function as regret/algorithms.main_only_fit."""
    s = len(y)
    L = max((1.0 if family == "gaussian" else 0.25 * nb) * np.linalg.eigvalsh(C.T @ C / s)[-1], 1e-12)
    gamma = np.zeros(C.shape[1])
    for _ in range(maxit):
        new = gamma - C.T @ (psi1(C @ gamma, family, nb) - y) / s / L
        if radii is not None:
            start = 0
            for size, R_g in zip(sizes, radii):
                nrm = np.linalg.norm(new[start:start + size])
                if nrm > R_g: new[start:start + size] *= R_g / nrm
                start += size
        if np.linalg.norm(new - gamma) <= tol * max(1.0, np.linalg.norm(new)):
            return new
        gamma = new
    raise RuntimeError("main-effects-only pilot fit did not converge")


def pilot_score(Zs, Ys, d1, d2, family, nb, main_fit=False, radius=None, sizes=None):
    """||G_tau||_op on the exploration sample S_o: the operator norm of the interaction block of the score at Theta = 0
    (main_fit: at the main-effects-only fit), as in Algorithm 1.  It is zero only when S_o does not excite the
    interaction block (the appendix's tie design)."""
    s_n, p = len(Ys), Zs.shape[1] - d1 * d2
    if main_fit and p > 0:
        sz = (d1, p - d1) if sizes is None else sizes
        gamma0 = main_only_fit(Zs[:, :p], Ys, family, nb, None if radius is None else radius[:-1], sz)
        g = Zs.T @ (psi1(Zs[:, :p] @ gamma0, family, nb) - Ys) / s_n
    else:
        g = Zs.T @ (psi1(0.0, family, nb) - Ys) / s_n
    score = np.linalg.norm(g[p:].reshape(d1, d2, order="F"), 2)
    if not (np.isfinite(score) and score >= 0):
        raise RuntimeError("degenerate pilot score")
    return float(score)


def penalty_level(c_lam, score, tau, n, d1, d2, s):
    """The penalty level of eq. lambdaTheta-main at sample size s, as in Algorithm 1:
    lambda_s = c_lam sigma_hat sqrt((d1 + d2) log(n d1 d2) / s), sigma_hat = score sqrt(tau / (d1 + d2))."""
    sigma_hat = score * np.sqrt(tau / (d1 + d2))
    return c_lam * sigma_hat * np.sqrt((d1 + d2) * np.log(n * d1 * d2) / max(s, 1))


# The penalty constant of each track, as in Algorithm 1 (regret/algorithms.py): 10-fold cross-validation on the
# track's rows (S_o and the track so far), first at s = max(2 tau, 200) and then whenever s doubles; CV_START before.
CV_GRID = (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)   # largest first: each fit starts at the previous one
CV_START = 1 / 8
CV_FOLDS = 10
CV_MIN = 200


def cv_checkpoints(tau, n):
    """The sample sizes at which c is chosen: s_1 = max(2 tau, CV_MIN), 2 s_1, 4 s_1, ... (below n)."""
    out, s = [], max(2 * tau, CV_MIN)
    while s < n:
        out.append(s)
        s *= 2
    return out


def cv_constant(Zs, Ys, base, theta, D, seed):
    """The penalty constant by CV_FOLDS-fold cross-validation on one track's rows: each c in CV_GRID is fitted on the
    other folds at lambda = c base(s') and scored on the fold by squared error or deviance; the smallest loss wins."""
    d1, d2, nb, radius = D.d1, D.d2, D.nb, D.radius
    sizes = getattr(D, "main_sizes", None)
    fold = np.random.default_rng(seed).permutation(len(Ys)) % CV_FOLDS
    loss = np.zeros(len(CV_GRID))
    for f in range(CV_FOLDS):
        tr, te = fold != f, fold == f
        Zt, Yt = Zs[tr], Ys[tr]
        s = len(Yt)
        start = theta
        for j, c in enumerate(CV_GRID):
            if D.family == "gaussian":
                start = fista(Zt.T @ Zt / s, Zt.T @ Yt / s, c * base(s), d1, d2, start, radius=radius, sizes=sizes)
            else:
                start = proximal_newton(Zt, Yt, c * base(s), d1, d2, start, nb, radius, sizes=sizes)[0]
            eta = Zs[te] @ start
            if D.family == "gaussian":
                loss[j] += np.sum((Ys[te] - eta) ** 2)
            else:
                loss[j] += np.sum(nb * np.logaddexp(0.0, eta) - Ys[te] * eta)
    return CV_GRID[int(np.argmin(loss))]


def c_in_force(rec, j, s):
    """The penalty constant track j used at sample size s: the declared c_lam, or with CV the choice of the last
    checkpoint <= s (CV_START before the first)."""
    if rec["c_lam"] != "cv":
        return rec["c_lam"]
    c = CV_START
    for s_k, c_k in rec["cv"][j]:
        if s_k <= s:
            c = c_k
    return c


C_REF = 1 / 15                # the refit's level requires C_ref >= 1/15


def lam_ref_sharp(m):
    """The ridge of the predictable tangent refit (eq. predictable-tangent-refit-level-main), divided by kappa:
    lambda_ref / kappa = m / (2 C_ref log m), m = n_tr - n0.  C_ref = 1/15, the largest ridge the theorem allows (the
    refit's noise grows with C_ref).  m = 0 (no refit): nan."""
    if m <= 0:
        return np.nan, np.nan
    return m / (2.0 * C_REF * np.log(max(m, 3))), C_REF


C_CAL = 1.0                   # lambda_cal = n kappa_hat / (2 C_cal log n); the theorem needs C_cal >= 1/2


def kappa_hat(Q, Gw, tau):
    """The information constant kappa of the two recursions, from the exploration sample: the average eigenvalue of the
    curvature-weighted exploration Gram Gw on the tangent coordinates Q (the average, not the smallest: with tau of
    the order of p_T the smallest sample eigenvalue is near zero)."""
    return float(np.trace(Q.T @ (Gw / max(tau, 1)) @ Q) / Q.shape[1])


C0 = 1 / 3                    # the constant of both splits, in units of the information itself


def kappa0_exploration(D, rec):
    """The information per round and coordinate of the exploration sample S_o,

        kappa0 = tr(G_w) / (tau p),   G_w = sum_{t in S_o} psi''(z_t' theta_o) z_t z_t',

    theta_o the exploration pilot (under the identity link G_w is the Gram matrix itself).  The average over all p
    coordinates: with tau rounds and p > tau no direction-specific information can be estimated from S_o (the pilot
    often has rank below r0), while the trace is estimated by the average of tau numbers."""
    w = psi2(rec["Z_ex"] @ rec["pilot"], D.family, D.nb)            # the curvature at the pilot (1: identity link)
    return float(np.sum(w * np.sum(rec["Z_ex"] ** 2, axis=1)) / (rec["tau"] * D.d))


def n0_rule(D, rec, N):
    """The information rule for the split of each track: the nuisance fit gets the rounds it needs to locate the
    tangent space and the rest goes to the refit,

        n0 = min{ max( ceil(C0 p_T log(n d1 d2) / kappa0), ceil(n_tr / 4), 10 ), floor(3 n_tr / 4) },

    n = tau + N, n_tr = N / 2, kappa0 = kappa0_exploration (S_o only, so n0 is fixed before either track is used).
    The floor and the cap keep n0 ~ n_tr - n0 ~ n_tr.  Returns (n0, kappa0)."""
    d1, d2, r0, p0 = D.d1, D.d2, D.r0, main_len(D)
    n_tr = N // 2
    kappa0 = kappa0_exploration(D, rec)
    need = int(np.ceil(C0 * p_T(d1, d2, r0, p0) * np.log((rec["tau"] + N) * d1 * d2) / kappa0))
    n0 = min(max(need, int(np.ceil(n_tr / 4)), 10), int(np.floor(3 * n_tr / 4)))
    return n0, kappa0


def design_split(D, rec, N, cap=Fraction(3, 4)):
    """A split fixed by the design before any data are seen (the displayed choice of n0):

        n0 = min{ max( ceil(C0 p_T log(n d1 d2) / kappa_D), floor(n_tr / 2) ), floor(cap n_tr) },

    kappa_D = D.kappa_D the planning value of the information per round (nb p_D (1 - p_D), or 1 under the identity
    link, in the normalization E[z z'] = I).  The equal split, lengthened when half a track is short of the burn-in, at
    most cap n_tr (cap a Fraction, so the floor is exact)."""
    d1, d2, r0, p0 = D.d1, D.d2, D.r0, main_len(D)
    n_tr = N // 2
    need = int(np.ceil(C0 * p_T(d1, d2, r0, p0) * np.log((rec["tau"] + N) * d1 * d2) / D.kappa_D))
    return min(max(need, n_tr // 2), int(cap * n_tr))


def pilot_rank(D, rec, tol=1e-10):
    """The numerical rank of the exploration pilot's interaction block (singular values above tol times the largest);
    a record of the run, not used by any rule."""
    p0 = main_len(D)
    s = np.linalg.svd(rec["pilot"][p0:].reshape(D.d1, D.d2, order="F"), compute_uv=False)
    return 0 if s[0] == 0 else int(np.sum(s > tol * s[0]))

# ================================================================ 5. Step 1: collection
def collect(D, tau, N_max, rng, c_lam):
    """Step 1: uniform exploration for tau rounds (S_o), then two greedy tracks in alternation for N_max / 2 rounds each;
    each track refits its penalized estimator every round on (S_o, its own past) at lambda_s = penalty_level(s) and
    pulls a_t = argmax_a z_{t,a}' theta_hat_{t-1}.  c_lam: a number or "cv" (each track chooses its constant by
    `cv_constant` at `cv_checkpoints`; the fold seeds are drawn from rng before the exploration).  Records the rows,
    the responses, the pilot, the CV choices, every round's expected regret and the arm played in every round (arm_ex,
    arm: the index among the round's K candidates, so a collection can be replayed from its seed)."""
    cv = isinstance(c_lam, str) and c_lam == "cv"
    cv_seed = int(rng.integers(2 ** 31)) if cv else None
    d1, d2, K, d, p0 = D.d1, D.d2, D.K, D.d, main_len(D)
    family, nb, radius = D.family, D.nb, D.radius
    sizes = getattr(D, "main_sizes", None)            # None: the groups (m, x)
    rows_design = hasattr(D, "next_round")            # the design supplies each round's stacked rows
    G, b = np.zeros((d, d)), np.zeros(d)
    Zex, Yex = np.zeros((tau, d)), np.zeros(tau)
    reg_ex = np.zeros(tau)
    arm_ex = np.zeros(tau, dtype=np.int64)
    for t in range(tau):
        if rows_design:
            Zc = D.next_round(rng)
            a = rng.integers(K)
            z = Zc[a]
            yt = D.reward_arm(rng, a)
        else:
            arms = D.draw_arms(rng)
            x = D.context(rng)
            a = rng.integers(K)
            Zc = np.array([stack(m, x, p0) for m in arms])
            z = Zc[a]
            yt = D.reward(rng, z @ D.theta)
        mean = mean_reward(Zc @ D.theta, family)
        reg_ex[t] = mean.max() - mean[a]
        G += np.outer(z, z); b += yt * z
        Zex[t], Yex[t], arm_ex[t] = z, yt, a
    n_tr = N_max // 2
    solver = dict(iters=0, calls=0, caps=0, refresh=0)
    Lmx = [None, None]                        # lambda_max of the per-track Gram, refreshed on a schedule
    score = pilot_score(Zex, Yex, d1, d2, family, nb, getattr(D, "anchor_at_main_fit", False), radius, sizes)
    n_h = tau + N_max                         # the collection's horizon in total rounds
    base = lambda s: penalty_level(1.0, score, tau, n_h, d1, d2, s)       # lambda_s at c = 1
    c = [CV_START, CV_START] if cv else [c_lam, c_lam]                     # each track's constant in force
    checkpoints, chosen = (cv_checkpoints(tau, tau + n_tr) if cv else []), [[], []]
    if family == "gaussian":
        pilot = fista(G / tau, b / tau, c[0] * base(tau), d1, d2, np.zeros(d), radius=radius, sizes=sizes)
    else:
        pilot, it, cap, _ = proximal_newton(Zex, Yex, c[0] * base(tau), d1, d2, np.zeros(d), nb, radius, sizes=sizes)
        solver["iters"] += it; solver["calls"] += 1; solver["caps"] += int(cap)
        Zall = [np.zeros((tau + n_tr, d)), np.zeros((tau + n_tr, d))]
        Yall = [np.zeros(tau + n_tr), np.zeros(tau + n_tr)]
        for j in (0, 1):
            Zall[j][:tau], Yall[j][:tau] = Zex, Yex
        Hw, LH, s_ref = [None, None], [None, None], [0, 0]
    Z = [np.zeros((n_tr, d)), np.zeros((n_tr, d))]
    Y = [np.zeros(n_tr), np.zeros(n_tr)]
    reg = [np.zeros(n_tr), np.zeros(n_tr)]
    arm = [np.zeros(n_tr, dtype=np.int64), np.zeros(n_tr, dtype=np.int64)]
    Gj, bj = [G.copy(), G.copy()], [b.copy(), b.copy()]
    th = [pilot.copy(), pilot.copy()]
    for t in range(n_tr):
        for j in (0, 1):
            s = tau + t
            if t > 0:
                if s in checkpoints:                                 # choose c again on the track's s rows
                    c[j] = cv_constant(np.vstack([Zex, Z[j][:t]]), np.concatenate([Yex, Y[j][:t]]), base, th[j], D,
                                       [cv_seed, j, s])
                    chosen[j].append((s, c[j]))
                lam = c[j] * base(s)
                if t <= 200 or t % 10 == 0 or Lmx[j] is None:     # G/s moves by O(1/s): every
                    Lmx[j] = np.linalg.eigvalsh(Gj[j] / s)[-1]    # tenth round is enough
                if family == "gaussian":
                    th[j] = fista(Gj[j] / s, bj[j] / s, lam, d1, d2, th[j], L=Lmx[j], radius=radius, sizes=sizes)
                else:
                    if s >= 1.25 * s_ref[j]:                     # exact refresh of the weighted Gram
                        w = psi2(Zall[j][:s] @ th[j], "binomial", nb)
                        Hw[j] = Zall[j][:s].T @ (Zall[j][:s] * w[:, None]); s_ref[j] = s
                        LH[j] = np.linalg.eigvalsh(Hw[j] / s)[-1]
                    th[j], it, cap, nref = proximal_newton(Zall[j][:s], Yall[j][:s], lam, d1, d2, th[j], nb,
                                                           radius, H=Hw[j] / s, LH=LH[j], L_max=0.25 * nb * Lmx[j],
                                                           sizes=sizes)
                    solver["iters"] += it; solver["calls"] += 1; solver["caps"] += int(cap); solver["refresh"] += nref
            if rows_design:
                Zc = D.next_round(rng)
            else:
                arms = D.draw_arms(rng)
                x = D.context(rng)
                Zc = np.array([stack(m, x, p0) for m in arms])
            a = int(np.argmax(Zc @ th[j]))                           # the greedy choice
            mean = mean_reward(Zc @ D.theta, family)
            reg[j][t] = mean.max() - mean[a]
            z = Zc[a]
            yt = D.reward_arm(rng, a) if rows_design else D.reward(rng, z @ D.theta)
            Z[j][t], Y[j][t], arm[j][t] = z, yt, a
            Gj[j] += np.outer(z, z); bj[j] += yt * z
            if family != "gaussian":
                Zall[j][s], Yall[j][s] = z, yt
                if Hw[j] is not None:
                    Hw[j] += psi2(z @ th[j], "binomial", nb) * np.outer(z, z)
    return dict(Z=Z, Y=Y, G_ex=G, b_ex=b, Z_ex=Zex, Y_ex=Yex, tau=tau, pilot=pilot, solver=solver,
                c_lam=c_lam if cv else float(c_lam), cv=chosen if cv else None, score=float(score), horizon=int(n_h),
                regret_ex=reg_ex, regret=reg, arm_ex=arm_ex, arm=arm)


def mean_reward(u, family):
    """The true mean reward of linear predictors u: u (identity link), the success probability (logit link)."""
    return u if family == "gaussian" else sigmoid(u)


def collection_regret(rec, N):
    """The cumulative expected regret of the first tau + 2 (N // 2) rounds of a collection (the rounds the inference on
    N post-exploration rounds uses): exploration, track 0, track 1, and their sum."""
    n_tr = N // 2
    ex, t0, t1 = (float(np.sum(v)) for v in (rec["regret_ex"], rec["regret"][0][:n_tr], rec["regret"][1][:n_tr]))
    return dict(regret=ex + t0 + t1, regret_ex=ex, regret_t0=t0, regret_t1=t1, regret_rounds=int(rec["tau"] + 2 * n_tr))


def cramer_rao(D, rec, N, As):
    """The Cramer-Rao standard error of each target (rows of As) for the design the two tracks collected.
    With Q* an orthonormal basis of the TRUE tangent space (main effects kept), a = Q*' A and

        I_N = sum_t psi''(z_t' theta*) Q*' z_t z_t' Q*   over the first n_tr = N // 2 rounds of both tracks,

    V_CR^2 = b a' I_N^{-1} a, b = sigma^2 (identity link) or 1 (logit link): the standard error of the efficient
    estimator if this design had been fixed in advance and the tangent space were known.  kappa_A = ||a||^2 /
    (a' (I_N / 2 n_tr)^{-1} a), the information along the target, so V_CR^2 = b ||P_T A||^2 / (2 n_tr kappa_A).
    Offline: uses theta*.  Returns (V_CR, kappa_A), one entry per target (nan if I_N is singular)."""
    d1, d2, r0, p0 = D.d1, D.d2, D.r0, main_len(D)
    n_tr = N // 2
    Q = tangent_basis(D.Us, D.Vs, d1, d2, r0, p0)
    ZQ = np.vstack([rec["Z"][0][:n_tr], rec["Z"][1][:n_tr]]) @ Q
    w = psi2(np.vstack([rec["Z"][0][:n_tr], rec["Z"][1][:n_tr]]) @ D.theta, D.family, D.nb)
    I = ZQ.T @ (ZQ * w[:, None])
    a = np.atleast_2d(As) @ Q                                         # L x p_T
    try:
        q = np.einsum("lp,pl->l", a, np.linalg.solve(I, a.T))         # a' I_N^{-1} a
    except np.linalg.LinAlgError:
        return np.full(len(a), np.nan), np.full(len(a), np.nan)
    b = D.sigma_eps ** 2 if D.family == "gaussian" else 1.0
    return np.sqrt(b * q), np.sum(a ** 2, axis=1) / (q * len(ZQ))


# ================================================================ 6. Steps 2-4: inference
def fold_estimates(D, rec, N, n0, rank_rule=None):
    """Step 2 of Algorithm 2 for both folds j = 0, 1, on the first N rounds of the collection (n_tr = N / 2 per
    track).  Fold j: the nuisance fit theta_breve on S_o and the first n0 rounds of track j, at the penalty level
    its track used at that sample size; its rank-r0 truncation and tangent basis Q; kappa_hat, the average tangent
    eigenvalue of the curvature-weighted exploration Gram; and the predictable ridge-type refit on rounds
    n0 + 1, ..., n_tr of track j at lambda_ref = m kappa_hat / (2 C_ref log m), truncated again.  This gives
    theta_hat^(j) and its tangent frame (U, V).  With rank_rule, fold j's truncation rank is chosen on fold j's
    pilot data at its nuisance fit.  Step 2 does not involve the loading A, so one call serves every target."""
    d1, d2, r0, p0 = D.d1, D.d2, D.r0, main_len(D)
    family, nb, radius = D.family, D.nb, D.radius
    sizes = getattr(D, "main_sizes", None)
    n_tr = N // 2
    tau, score, n_h = rec["tau"], rec["score"], rec["horizon"]
    c_nuis = [c_in_force(rec, j, tau + n0) for j in (0, 1)]          # each fold's nuisance-fit constant
    ranks = [r0, r0]                                                 # the truncation rank of each fold
    th_hat, frames, kap = [None, None], [None, None], [1.0, 1.0]
    err_pre = np.zeros(2)
    refit_info = {}
    nuis_it = nuis_cap = 0
    for j in (0, 1):
        Z, Y = rec["Z"][j][:n_tr], rec["Y"][j][:n_tr]
        s = tau + n0
        lam = penalty_level(c_nuis[j], score, tau, n_h, d1, d2, s)  # the track's level at this fit's size
        if family == "gaussian":
            G = rec["G_ex"] + Z[:n0].T @ Z[:n0]
            b = rec["b_ex"] + Z[:n0].T @ Y[:n0]
            brev = fista(G / s, b / s, lam, d1, d2, rec["pilot"], iters=600, tol=1e-8, radius=radius, sizes=sizes)
        else:
            Zs, Ys = np.vstack([rec["Z_ex"], Z[:n0]]), np.concatenate([rec["Y_ex"], Y[:n0]])
            brev, it, cap, _ = proximal_newton(Zs, Ys, lam, d1, d2, rec["pilot"], nb, radius, tol=1e-8, sizes=sizes)
            nuis_it += it; nuis_cap += int(cap)
        if rank_rule is not None:                                    # fold j's pilot data only
            ranks[j] = rank_rule(np.vstack([rec["Z_ex"], Z[:n0]]), np.concatenate([rec["Y_ex"], Y[:n0]]), brev)
        brev, U, V = truncate(brev, d1, d2, ranks[j])
        err_pre[j] = float(np.linalg.norm(brev - D.theta))              # offline: before the refit
        Q = tangent_basis(U, V, d1, d2, ranks[j], p0)
        m_ref = n_tr - n0
        sigma_hat = float(np.sqrt(np.mean((Y[:n0] - psi1(Z[:n0] @ brev, family, nb)) ** 2)))
        # kappa_hat from fold j's own data only (its lambda_cal acts on the other track)
        if family == "gaussian":
            Gw = rec["G_ex"]
        else:
            cw = psi2(rec["Z_ex"] @ brev, family, nb)
            Gw = rec["Z_ex"].T @ (rec["Z_ex"] * cw[:, None])
        kap[j] = kappa_hat(Q, Gw, tau)
        lam_ref, C_ref = lam_ref_sharp(m_ref)
        lam_ref *= kap[j]                                            # nan when there is no refit
        R, acc = np.eye(Q.shape[1]), np.zeros(Q.shape[1])
        for t in range(n0, n_tr):                                    # the predictable refit recursion
            q = Q.T @ Z[t]
            u = Z[t] @ brev
            c = psi2(u, family, nb)
            w = (R.T @ q) / (lam_ref + c * (q @ q))
            R = R - c * np.outer(q, w)
            acc += w * (Y[t] - psi1(u, family, nb))
        th_hat[j], Uh, Vh = truncate(brev + Q @ acc, d1, d2, ranks[j])
        frames[j] = (Uh, Vh)
        if j == 0:
            refit_info = dict(lam_ref=float(lam_ref), C_ref=float(C_ref), p_T=int(Q.shape[1]), sigma_hat=sigma_hat)
    return dict(th_hat=th_hat, frames=frames, kap=kap, ranks=ranks, err_pre=err_pre, refit_info=refit_info,
                nuis_it=nuis_it, nuis_cap=nuis_cap, c_nuis=c_nuis, lam=lam, n0=n0, N=N)


def two_track_inference(D, rec, N, A=None, n0=None, rank_rule=None, F=None):
    """Algorithm 2, Steps 2-4, on the first N rounds of the collection (N/2 per track).

    Step 2 (per fold j): fold_estimates above; F passes its output when the caller already has it (it does not
      depend on the target, so one Step 2 serves every target).
    Step 3 (per fold j, over the OPPOSITE track j'): the predictable calibration recursion at
      lambda_cal = n kappa_hat / (2 C_cal log n) gives the weights omega_t;
      mu_hat^(j) = <A, theta_hat^(j)> + sum_t omega_t (y_t - psi'(z_t' theta_hat^(j))),
      V_j^2 = sum_t omega_t^2 (y_t - psi'(.))^2 (realized quadratic variation).
    Step 4: mu_hat = (mu^(0) + mu^(1))/2, V = sqrt(V_0^2 + V_1^2)/2, T = (mu_hat - mu*)/V, the predictable
      cross-fitted fold variance of the theorem; the contamination-corrected V_corr is a diagnostic only.

    n0: each track's rounds for the nuisance fit; None takes the information rule `n0_rule`; n0 = N // 2 is the
    no-refit arm.  rank_rule: None uses the known r0; otherwise a function (Zs, Ys, theta) -> rank applied in each
    fold to its pilot data (the appendix's unknown-r0 version; see `rank_rule` below).
    Also reported: the one-step empirical-Hessian comparator on the same data, the cross-track contamination estimate,
    and offline diagnostics that use theta* (corr_eps, corr_nuis, theta_err, the remainder decomposition)."""
    d1, d2, r0, p0 = D.d1, D.d2, D.r0, main_len(D)
    family, nb = D.family, D.nb
    if A is None:
        A = D.A
    mu_star = float(A @ D.theta)
    PTA = tangent_project(A, D.Us, D.Vs, d1, d2)                    # true tangent projection (scoring)
    n_tr = N // 2
    kappa0 = np.nan                                                  # reported when the rule sets n0
    if n0 is None:
        n0, kappa0 = n0_rule(D, rec, N)
    tau, score = rec["tau"], rec["score"]
    if F is None:
        F = fold_estimates(D, rec, N, n0, rank_rule)                 # ---- Step 2, both folds
    th_hat, frames, kap, ranks = F["th_hat"], F["frames"], F["kap"], F["ranks"]
    err_pre, refit_info, c_nuis, lam = F["err_pre"], F["refit_info"], F["c_nuis"], F["lam"]
    nuis_it, nuis_cap = F["nuis_it"], F["nuis_cap"]
    mus, Vs, plug, corr, ess, rn = (np.zeros(2) for _ in range(6))
    Dhat2, ce, cd, ols_mu, ols_V, psi2_bar = (np.zeros(2) for _ in range(6))
    r_cal, r_norA, r_norz, r_tay, vo2, vp2 = (np.zeros(2) for _ in range(6))      # ablation record
    err_tan, err_nor, lam_cal_used = (np.zeros(2) for _ in range(3))
    diag = {}
    # ---- Step 3
    for j in (0, 1):
        jp = 1 - j
        Z, Y = rec["Z"][jp][:n_tr], rec["Y"][jp][:n_tr]
        Uh, Vh = frames[j]
        rho = tangent_project(A, Uh, Vh, d1, d2)
        lam_cal = n_tr / (2.0 * C_CAL * np.log(max(n_tr, 3))) * kap[j]
        uh = Z @ th_hat[j]
        cc = psi2(uh, family, nb) if family != "gaussian" else np.ones(n_tr)
        om = np.zeros(n_tr)
        for t in range(n_tr):                                        # the calibration recursion
            Pz = tangent_project(Z[t], Uh, Vh, d1, d2)
            w = (Z[t] @ rho) / (lam_cal + cc[t] * (Pz @ Pz))
            rho = rho - cc[t] * Pz * w
            om[t] = w
        res = Y - psi1(uh, family, nb)
        psi2_bar[j] = float(np.mean(cc))
        plug[j], corr[j] = float(A @ th_hat[j]), float(res @ om)
        mus[j] = plug[j] + corr[j]
        Vs[j] = np.sqrt(np.sum(res ** 2 * om ** 2))
        rn[j] = float(np.linalg.norm(rho))
        ess[j] = float((om @ om) ** 2 / np.sum(om ** 4))
        # cross-track estimate of the nuisance contamination of the residuals
        dq = cc * (Z @ (th_hat[0] - th_hat[1]))
        Dhat2[j] = 0.5 * float(np.sum(om ** 2 * dq ** 2))
        # offline diagnostics (use theta*), kept for ablation.  Exactly,
        #   mu_hat^(j) - mu* = plug error + ce + cd,   ce = sum om eps (the martingale),
        #   plug error + cd  = r_cal + r_norA + r_norz + r_tay,
        # the theorem's master decomposition: calibration residual (P Delta)' rho_n, the
        # loading's and the designs' normal components, and the Taylor remainder (0 under
        # the identity link).  vo2 is the quadratic variation with the true noise, vp2 the
        # predictable one, so V can be compared with both.
        u_star = Z @ D.theta
        eps_true = Y - psi1(u_star, family, nb)
        ce[j] = float(om @ eps_true)
        cd[j] = float(om @ (psi1(u_star, family, nb) - psi1(uh, family, nb)))
        Delta = th_hat[j] - D.theta
        PDelta = tangent_project(Delta, Uh, Vh, d1, d2)
        NDelta = Delta - PDelta
        r_cal[j] = float(PDelta @ rho)
        r_norA[j] = float(A @ NDelta)
        r_norz[j] = -float(om @ (cc * (Z @ NDelta)))
        r_tay[j] = -float(om @ (psi1(uh, family, nb) - psi1(u_star, family, nb) - cc * (Z @ Delta)))
        var_t = np.full(n_tr, D.sigma_eps ** 2) if family == "gaussian" else psi2(u_star, family, nb)
        vo2[j] = float(np.sum(om ** 2 * eps_true ** 2))
        vp2[j] = float(np.sum(om ** 2 * var_t))
        err_tan[j], err_nor[j] = float(np.linalg.norm(PDelta)), float(np.linalg.norm(NDelta))
        lam_cal_used[j] = lam_cal
        # the comparator: cross-fitted one-step with the inverse empirical Hessian of the track; nan when that
        # Hessian is singular (fewer rounds than coordinates, or the tie design's integer rows at small N)
        ols_mu[j] = ols_V[j] = np.nan
        if n_tr >= Z.shape[1]:
            try:
                sc = Z @ np.linalg.solve(Z.T @ (Z * cc[:, None]) / n_tr, A)
                ols_mu[j] = float(A @ th_hat[j] + np.mean(sc * res))
                ols_V[j] = float(np.sqrt(np.sum((sc * res) ** 2)) / n_tr)
            except np.linalg.LinAlgError:
                pass
        if j == 0:                                                   # weight and excitation diagnostics
            Q_hat = tangent_basis(Uh, Vh, d1, d2, ranks[0], p0)
            ex = np.linalg.eigvalsh(Q_hat.T @ (rec["G_ex"] / max(tau, 1)) @ Q_hat)
            eg = np.linalg.eigvalsh(Q_hat.T @ (Z.T @ Z / n_tr) @ Q_hat)
            w2 = om ** 2
            cw = np.cumsum(w2) / np.sum(w2)
            PhA = tangent_project(A, Uh, Vh, d1, d2)
            diag = dict(maxw=float(np.max(w2) / np.sum(w2)),
                        res2_w=float(np.sum(res ** 2 * w2) / np.sum(w2) / D.sigma_eps ** 2),
                        res2_u=float(np.mean(res ** 2) / D.sigma_eps ** 2),
                        PT_A=float(np.linalg.norm(PTA)),
                        proj_gap=float(np.linalg.norm(PhA - PTA) / max(np.linalg.norm(PTA), 1e-12)),
                        off_T=float(np.linalg.norm(A - PTA) / max(np.linalg.norm(A), 1e-12)),
                        om_mass50=float(np.sum(np.sort(w2)[::-1][:50]) / np.sum(w2)),
                        om_argmax=int(np.argmax(w2)),
                        om_t50=float(np.searchsorted(cw, .5) / n_tr), om_t90=float(np.searchsorted(cw, .9) / n_tr),
                        kap_ex_min=float(ex[0]), kap_ex_avg=float(ex.mean()),
                        kap_gr_min=float(eg[0]), kap_gr_avg=float(eg.mean()),
                        kap_gr_cond=float(eg[-1] / max(eg[0], 1e-14)))
    # ---- Step 4
    mu = 0.5 * (mus[0] + mus[1])
    V = 0.5 * np.sqrt(Vs[0] ** 2 + Vs[1] ** 2)
    mu_ols = 0.5 * (ols_mu[0] + ols_mu[1])
    V_ols = 0.5 * np.sqrt(ols_V[0] ** 2 + ols_V[1] ** 2)
    Vc = 0.5 * np.sqrt(np.sum(np.maximum(Vs ** 2 - Dhat2, 1e-24)))
    sv = rec["solver"]
    return dict(N=N, n0=n0, n0_frac=n0 / n_tr, kappa0=float(kappa0), c0=C0, tau=tau, mu=mu, V=V, mu_star=mu_star,
                T=(mu - mu_star) / V if V > 0 else 0.0,
                V_corr=float(Vc), infl_hat=float(V / Vc) if Vc > 0 else np.nan,
                T_corr=(mu - mu_star) / Vc if Vc > 0 else 0.0,
                mu_ols=float(mu_ols), V_ols=float(V_ols),
                T_ols=float((mu_ols - mu_star) / V_ols) if V_ols > 0 else np.nan,
                mu_plug=float(plug.mean()), mu_corr=float(corr.mean()), plug_err=float(plug.mean() - mu_star),
                corr_eps=float(ce.mean()), corr_nuis=float(cd.mean()),
                r_cal=float(r_cal.mean()), r_norA=float(r_norA.mean()), r_norz=float(r_norz.mean()),
                r_tay=float(r_tay.mean()),
                V_orc=float(0.5 * np.sqrt(vo2.sum())), V_pqv=float(0.5 * np.sqrt(vp2.sum())),
                corr_eps_0=float(ce[0]), corr_eps_1=float(ce[1]),         # per fold: the martingales and
                V_pqv_0=float(np.sqrt(vp2[0])), V_pqv_1=float(np.sqrt(vp2[1])),   # their predictable s.e.
                mu_0=float(mus[0]), mu_1=float(mus[1]), V_0=float(Vs[0]), V_1=float(Vs[1]),
                err_pre=float(err_pre.mean()), err_post=float(0.5 * (np.linalg.norm(th_hat[0] - D.theta)
                                                                    + np.linalg.norm(th_hat[1] - D.theta))),
                err_tan=float(err_tan.mean()), err_nor=float(err_nor.mean()),
                lam_cal=float(lam_cal_used.mean()), C_cal=float(C_CAL),
                ess=float(ess.mean()), resnorm=float(rn.mean()),
                kap_hat=float(0.5 * (kap[0] + kap[1])), kap_hat_0=float(kap[0]), kap_hat_1=float(kap[1]),
                theta_err=float(np.linalg.norm(th_hat[0] - D.theta)),
                theta_relerr=float(np.linalg.norm(th_hat[0] - D.theta) / max(np.linalg.norm(D.theta), 1e-12)),
                psi2_bar=float(psi2_bar.mean()), nuis_iters=int(nuis_it), nuis_caps=int(nuis_cap),
                c_lam=rec["c_lam"] if rec["c_lam"] == "cv" else float(rec["c_lam"]), c_nuis_0=float(c_nuis[0]),
                c_nuis_1=float(c_nuis[1]), score=float(score), lam_nuis=float(lam),
                lam_over_score=float(lam / score) if score > 0 else np.nan,
                col_iters=int(sv["iters"]), col_calls=int(sv["calls"]), col_caps=int(sv["caps"]),
                r0=int(r0), rank_0=int(ranks[0]), rank_1=int(ranks[1]),
                **refit_info, **diag)


def tangent_project_rows(Zr, U, V, d1, d2):
    """tangent_project applied to every row of Zr (rows are design vectors; main-effect coordinates kept)."""
    p0 = Zr.shape[1] - d1 * d2
    out = Zr.copy()
    M = Zr[:, p0:].reshape(len(Zr), d2, d1).transpose(0, 2, 1)       # each row as a d1 x d2 matrix (column-major vec)
    UUt, VVt = U @ U.T, V @ V.T
    P = UUt @ M + M @ VVt - UUt @ M @ VVt
    out[:, p0:] = P.transpose(0, 2, 1).reshape(len(Zr), d1 * d2)
    return out


def calibrate_loadings(D, rec, N, As, F, comparator=False):
    """Steps 3-4 of Algorithm 2 for several loadings at once (the rows of As), given Step 2's fold estimates
    F = fold_estimates(D, rec, N, n0).  For fold j the calibration recursion runs over the opposite track,

        omega_t = <z_t, rho_{t-1}> / (lambda_cal + psi''_t ||P z_t||^2),   rho_t = rho_{t-1} - psi''_t omega_t P z_t,

    from rho_0 = P A, P the projection on fold j's tangent space.  This is the recursion of two_track_inference; it
    depends on the loading only through rho_0, so the loadings run together, one column of rho each.  Then
    mu^(j) = <A, theta_hat^(j)> + sum_t omega_t (y_t - psi'(z_t' theta_hat^(j))), V_j^2 = sum_t omega_t^2 (y_t -
    psi'(.))^2, mu_hat = (mu^(0) + mu^(1)) / 2 and V = sqrt(V_0^2 + V_1^2) / 2.  Returns one dict per loading: the
    estimate, its standard error and T, the oracle standard error V_pqv and the remainder in its four parts
    (r_cal, r_norA, r_norz, r_tay; computed with theta*, as in two_track_inference).  comparator: also the one-step
    comparator of two_track_inference for every loading (mu_ols, V_ols, T_ols; nan when its Hessian is singular)."""
    d1, d2, family, nb = D.d1, D.d2, D.family, D.nb
    As = np.atleast_2d(As)
    L, n_tr = len(As), N // 2
    mus, Vs, rc, rnA, rnz, rty, vp2, en = (np.zeros((2, L)) for _ in range(8))
    ols_mu, ols_V = np.full((2, L), np.nan), np.full((2, L), np.nan)
    for j in (0, 1):
        Z, Y = rec["Z"][1 - j][:n_tr], rec["Y"][1 - j][:n_tr]        # the opposite track
        Uh, Vh = F["frames"][j]                                      # fold j's tangent frame
        th = F["th_hat"][j]
        PZ = tangent_project_rows(Z, Uh, Vh, d1, d2)
        qq = np.einsum("ij,ij->i", PZ, PZ)
        rho = np.ascontiguousarray(tangent_project_rows(As, Uh, Vh, d1, d2).T)     # d x L
        lam_cal = n_tr / (2.0 * C_CAL * np.log(max(n_tr, 3))) * F["kap"][j]
        uh = Z @ th
        cc = psi2(uh, family, nb) if family != "gaussian" else np.ones(n_tr)
        om = np.zeros((n_tr, L))
        for t in range(n_tr):                                        # the calibration recursion
            w = (Z[t] @ rho) / (lam_cal + cc[t] * qq[t])
            rho -= cc[t] * np.outer(PZ[t], w)
            om[t] = w
        res = Y - psi1(uh, family, nb)
        mus[j] = As @ th + res @ om
        Vs[j] = np.sqrt((res ** 2) @ (om ** 2))
        # mu^(j) - mu* - sum_t omega_t eps_t = r_cal + r_norA + r_norz + r_tay (eps_t the true noise)
        u_star = Z @ D.theta
        Delta = th - D.theta
        PDelta = tangent_project(Delta, Uh, Vh, d1, d2)
        NDelta = Delta - PDelta
        rc[j] = PDelta @ rho
        rnA[j] = As @ NDelta
        rnz[j] = -(cc * (Z @ NDelta)) @ om
        rty[j] = -(psi1(uh, family, nb) - psi1(u_star, family, nb) - cc * (Z @ Delta)) @ om
        var_t = np.full(n_tr, D.sigma_eps ** 2) if family == "gaussian" else psi2(u_star, family, nb)
        vp2[j] = var_t @ (om ** 2)
        en[j] = np.linalg.norm(NDelta)
        if comparator and n_tr >= Z.shape[1]:                        # the one-step comparator, as in two_track_inference
            try:
                sc = Z @ np.linalg.solve(Z.T @ (Z * cc[:, None]) / n_tr, As.T)        # n_tr x L
                ols_mu[j] = As @ th + np.mean(sc * res[:, None], axis=0)
                ols_V[j] = np.sqrt(np.sum((sc * res[:, None]) ** 2, axis=0)) / n_tr
            except np.linalg.LinAlgError:
                pass
    mu, V, mu_star = 0.5 * (mus[0] + mus[1]), 0.5 * np.sqrt(Vs[0] ** 2 + Vs[1] ** 2), As @ D.theta
    mu_o, V_o = 0.5 * (ols_mu[0] + ols_mu[1]), 0.5 * np.sqrt(ols_V[0] ** 2 + ols_V[1] ** 2)
    out = [dict(n0=int(F["n0"]), n0_frac=F["n0"] / n_tr, mu=float(mu[l]), V=float(V[l]), mu_star=float(mu_star[l]),
                T=float((mu[l] - mu_star[l]) / V[l]) if V[l] > 0 else 0.0, V_pqv=float(0.5 * np.sqrt(vp2[:, l].sum())),
                r_cal=float(rc[:, l].mean()), r_norA=float(rnA[:, l].mean()), r_norz=float(rnz[:, l].mean()),
                r_tay=float(rty[:, l].mean()), err_nor=float(en[:, l].mean()), kap_hat=float(np.mean(F["kap"])))
           for l in range(L)]
    if comparator:
        for l, o in enumerate(out):
            o.update(mu_ols=float(mu_o[l]), V_ols=float(V_o[l]),
                     T_ols=float((mu_o[l] - mu_star[l]) / V_o[l]) if V_o[l] > 0 else np.nan)
    return out

# ================================================================ 7. unknown r0 (appendix)
# The rank is chosen on each fold's pilot data at its nuisance fit, by an information criterion, before the truncation.
RANK_PRICE = {"gic": lambda s: np.log(s) / s,         # Lee, Sun and Liu (JASA 2025), eq. 4: rank-consistent
              "aic": lambda s: 1.0 / s}               # an AIC-type price: lighter, may over-select


def neg_loglik(Zs, Ys, theta, family, nb):
    """The average negative log-likelihood L_s(theta) on s observations (identity link: the profile
    0.5 log(RSS / s), the noise variance unknown; logit link: the binomial(nb) likelihood)."""
    u = Zs @ theta
    if family == "gaussian":
        return 0.5 * float(np.log(np.mean((Ys - u) ** 2)))
    return float(np.mean(nb * np.logaddexp(0.0, u) - Ys * u))


def rank_rule(D, kind):
    """The rank rule of the appendix: a function (Zs, Ys, theta) -> the rank minimizing
        L_s(theta_r) + price(s) r (d1 + d2 - r),   r = 0, 1, ..., min(d1, d2),
    theta_r the rank-r truncation of the nuisance fit theta (main effects, if any, kept), s = len(Ys),
    price(s) = RANK_PRICE[kind](s); the chosen rank is at least 1.  r (d1 + d2 - r) is the number of
    free parameters of a rank-r d1 x d2 matrix."""
    price = RANK_PRICE[kind]

    def rule(Zs, Ys, theta):
        s = len(Ys)
        crit = [neg_loglik(Zs, Ys, truncate(theta, D.d1, D.d2, r)[0], D.family, D.nb) + price(s) * r * (D.d1 + D.d2 - r)
                for r in range(min(D.d1, D.d2) + 1)]
        return max(int(np.argmin(crit)), 1)
    return rule
