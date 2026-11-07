"""Algorithm 1 of the paper (the greedy nuclear-norm GLM bandit) and its estimator.

Sections:
  1. the exponential family     psi, psi', psi''
  2. the data record            rows (m_t, x_t, y_t) and sufficient statistics
  3. the penalized estimator    (eq. prefix-estimator): FISTA (Gaussian), proximal Newton (binomial)
  4. the penalty                lambda_s = c sigma_hat sqrt((d1 + d2) log(n d1 d2) / s)  (eq. lambdaTheta-main),
                                c fixed or chosen inside the run by 10-fold cross-validation
  5. the exploration length     tau = ceil(c_tau (d1 + d2 + q) log(n d1 d2))
  6. Algorithm 1                run_algorithm1: pure exploration, then greedy exploitation

Notation.  Theta is d1 x d2.  Arm m (R^d1) and context x (R^d2) give z = vec(m x') (row-major), so
<z, vec(Theta)> = m' Theta x.  Rewards follow the canonical GLM y ~ exp{y <z, Theta> - psi(<z, Theta>)}:
Gaussian (psi(u) = u^2 / 2) or binomial with nb trials (psi(u) = nb log(1 + e^u)).  The estimator on s
observations is

    min_{||Theta||_F <= radius}  (1/s) sum_i [psi(<z_i, Theta>) - y_i <z_i, Theta>] + lambda ||Theta||_*.

Optional main effects: an environment may give covariates c (env.main, n x K x q) whose coefficients gamma are
unpenalized and constrained to balls (env.main_groups = [(size, radius), ...]); the index is then
c' gamma + m' Theta x.  The paper's vector main effects are c = (m, x); the MIND analysis adds known covariates.
"""
import numpy as np
from scipy.linalg import eigh


# ================================================================ 1. the family
def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def psi(u, family, nb):
    """Cumulant of the family."""
    return 0.5 * u * u if family == "gaussian" else nb * np.logaddexp(0.0, u)


def psi1(u, family, nb):
    """Mean function psi'."""
    return u if family == "gaussian" else nb * sigmoid(u)


def psi2(u, family, nb):
    """Variance function psi''."""
    if family == "gaussian":
        return np.ones_like(u)
    q = sigmoid(u)
    return nb * q * (1.0 - q)


def curvature_bound(family, nb):
    """sup psi'': the Lipschitz constant of the gradient is this times lambda_max(Gram)."""
    return 1.0 if family == "gaussian" else nb / 4.0


def lambda_max(A):
    """The largest eigenvalue of the symmetric matrix A (LAPACK dsyevr for the top eigenvalue only: numpy's
    eigvalsh value to rounding, at about half the cost)."""
    k = A.shape[0] - 1
    return float(eigh(A, eigvals_only=True, subset_by_index=[k, k])[0])


# ================================================================ 2. the data record
class Record:
    """The observations (m_t, x_t, y_t) of the chosen arms and their sufficient statistics.

    G = sum z z', cross = sum y z and y2 = sum y^2 are all the identity link needs, so a Gaussian round costs O(p^2).
    H = sum w z z' is the curvature-weighted Gram that preconditions the logit link's Newton steps; its weights are
    taken when each row is added and it is rebuilt exactly once the prefix has grown by a quarter (H_n: the prefix
    length at the last rebuild).  The logit link's gradient and objective read the rows.
    """

    def __init__(self, d1, d2, capacity, main_dim=0):
        self.d1, self.d2, self.p, self.n = d1, d2, d1 * d2, 0
        self.q = main_dim                     # number of main-effect covariates per row (0: none)
        k = self.q + self.p
        self.M = np.zeros((capacity, d1))
        self.X = np.zeros((capacity, d2))
        self.C = np.zeros((capacity, self.q))
        self.y = np.zeros(capacity)
        self.G = np.zeros((k, k))
        self.cross = np.zeros(k)
        self.y2 = 0.0
        self.H = np.zeros((k, k))
        self.H_n = 0

    def add(self, m, x, y, weight=None, c=None):
        """Append a row.  weight: psi'' at the current estimate (None under the identity link: no H).  c: the row's
        main-effect covariates (the design row is then (c, vec(m x')))."""
        i = self.n
        self.M[i], self.X[i], self.y[i] = m, x, y
        z = np.outer(m, x).ravel()
        if self.q > 0:
            self.C[i] = c
            z = np.concatenate([c, z])
        zz = np.outer(z, z)
        self.G += zz
        self.cross += y * z
        self.y2 += y * y
        if weight is not None:
            self.H += weight * zz
        self.n = i + 1

    def rows(self):
        return self.M[:self.n], self.X[:self.n], self.y[:self.n]

    def main_rows(self):
        return self.C[:self.n]

    def refresh_H(self, theta, family, nb, gamma=None):
        """Exact curvature-weighted Gram at (gamma, theta)."""
        M, X, _ = self.rows()
        w = psi2(index(self, theta, gamma), family, nb)
        Z = np.einsum("ni,nj->nij", M, X).reshape(self.n, self.p)
        if self.q > 0:
            Z = np.hstack([self.main_rows(), Z])
        self.H = (Z * w[:, None]).T @ Z
        self.H_n = self.n

    def curvature_gram(self, theta, family, nb, exact=False, gamma=None):
        """H / n, rebuilt exactly when asked or when the prefix has grown by a quarter; else the running sum."""
        if exact or self.H_n == 0 or self.n >= 1.25 * self.H_n:
            self.refresh_H(theta, family, nb, gamma)
        return self.H / self.n


def record_from_rows(M, X, y, C=None):
    rec = Record(M.shape[1], X.shape[1], len(y), 0 if C is None else C.shape[1])
    for i in range(len(y)):
        rec.add(M[i], X[i], y[i], c=None if C is None else C[i])
    return rec


def linear_predictor(theta, M, X):
    return np.einsum("ij,ij->i", M @ theta, X)


def index(record, theta, gamma=None):
    """The linear index of the recorded rows: m' Theta x, plus c' gamma with main effects."""
    M, X, _ = record.rows()
    eta = linear_predictor(theta, M, X)
    if record.q > 0:
        eta = eta + record.main_rows() @ gamma
    return eta


def stack(gamma, theta):
    """The parameter as one vector (gamma, vec(Theta))."""
    if gamma is None or gamma.size == 0:
        return theta.ravel()
    return np.concatenate([gamma, theta.ravel()])


def pair_norm(dgamma, dtheta):
    """Euclidean norm of (gamma, vec(Theta))."""
    if dgamma is None or dgamma.size == 0:
        return np.linalg.norm(dtheta)
    return np.sqrt(np.sum(dgamma * dgamma) + np.sum(dtheta * dtheta))


def project_main(gamma, groups):
    """Projection of the main-effect coefficients onto their balls, groups = [(size, radius), ...]."""
    out, start = np.array(gamma, dtype=float), 0
    for size, radius in groups:
        norm = np.linalg.norm(out[start:start + size])
        if norm > radius:
            out[start:start + size] *= radius / norm
        start += size
    return out


def gradient(record, theta, family, nb, gamma=None):
    """Gradient of the mean negative log-likelihood: (gamma part, Theta part)."""
    s, q = record.n, record.q
    if family == "gaussian":
        full = record.G @ stack(gamma, theta) - record.cross
        return full[:q] / s, full[q:].reshape(record.d1, record.d2) / s
    M, X, y = record.rows()
    resid = psi1(index(record, theta, gamma), family, nb) - y
    return record.main_rows().T @ resid / s, M.T @ (X * resid[:, None]) / s


def nuclear_norm(theta):
    return np.linalg.svd(theta, compute_uv=False).sum()


def objective(record, theta, family, nb, lam, gamma=None):
    """The penalized mean negative log-likelihood at (gamma, theta)."""
    s = record.n
    if family == "gaussian":
        c = stack(gamma, theta)
        loss = 0.5 * (c @ (record.G @ c) - 2.0 * (c @ record.cross) + record.y2) / s
    else:
        _, _, y = record.rows()
        eta = index(record, theta, gamma)
        loss = np.mean(psi(eta, family, nb) - y * eta)
    return loss + lam * nuclear_norm(theta)


# ================================================================ 3a. FISTA for the quadratic model
def prox(theta, threshold, radius):
    """Proximal map of threshold ||.||_* on the Frobenius ball: soft-threshold the singular values, then rescale."""
    return prox_sv(theta, threshold, radius)[0]


def prox_sv(theta, threshold, radius):
    """`prox` and the singular values of its result (their sum is its nuclear norm)."""
    u, s, vt = np.linalg.svd(theta, full_matrices=False)
    s = np.maximum(s - threshold, 0.0)
    norm = np.sqrt(np.sum(s * s))
    if norm > radius:
        s = s * (radius / norm)
    return (u * s) @ vt, s


def fista(G, b, y2, lam, radius, start, maxit, tol, L=None):
    """min 0.5 (c'Gc - 2 c'b + y2) + lam ||Theta||_* over ||Theta||_F <= radius, c = vec(Theta).

    Monotone FISTA with adaptive restart, step 1 / L, L = lambda_max(G) (passed in when the caller has it).  Stops
    when the scaled proximal-gradient residual is <= tol and the relative objective change <= sqrt(tol).
    Returns (theta, iterations, converged)."""
    theta, it, converged, _ = fista_pair(G, b, y2, lam, radius, start, maxit, tol, L)
    return theta, it, converged


def fista_pair(G, b, y2, lam, radius, start, maxit, tol, L=None, start_main=None, groups=(), scale=(1.0, 1.0)):
    """`fista` with the optional main-effect block: the variable is (gamma, Theta), G and b are over
    (gamma, vec(Theta)), and gamma is projected onto its balls instead of penalized.  scale: see `unit_floors`.
    Returns (theta, iterations, converged, gamma)."""
    floor_g, floor_f, _ = unit_floors(scale)
    d1, d2 = start.shape
    q = 0 if start_main is None else start_main.size
    theta, sv = prox_sv(start, 0.0, radius)
    gamma = project_main(start_main, groups) if q else np.zeros(0)
    if L is None:
        L = lambda_max(G)
    L = max(L, 1e-8)

    def evaluate(g_, A, sv_A):
        """The objective at (g_, A), A a prox output with singular values sv_A, and the gradient of its smooth
        part (each point's product with G is computed once)."""
        c = stack(g_, A)
        Gc = G @ c
        return 0.5 * (c @ Gc - 2.0 * (c @ b) + y2) + lam * np.sum(sv_A), Gc - b

    def prox_step(g_, A, grad):
        """The proximal-gradient step from (g_, A), grad the smooth part's gradient there."""
        A_new, sv_new = prox_sv(A - grad[q:].reshape(d1, d2) / L, lam / L, radius)
        return project_main(g_ - grad[:q] / L, groups), A_new, sv_new

    previous, grad = evaluate(gamma, theta, sv)
    ext_main, extrapolated, ext_grad, momentum = gamma, theta, grad, 1.0
    converged, it = False, 0
    for it in range(1, maxit + 1):
        cand_main, candidate, cand_sv = prox_step(ext_main, extrapolated, ext_grad)
        value, cand_grad = evaluate(cand_main, candidate, cand_sv)
        restart = value > previous
        if restart:                                     # monotone: fall back to a plain step
            cand_main, candidate, cand_sv = prox_step(gamma, theta, grad)
            value, cand_grad = evaluate(cand_main, candidate, cand_sv)
        grad_norm = np.linalg.norm(cand_grad)
        step_main, step, _ = prox_step(cand_main, candidate, cand_grad)
        residual = L * pair_norm(cand_main - step_main, candidate - step) / max(floor_g, grad_norm, lam)
        relative = abs(value - previous) / max(floor_f, abs(previous))
        old_main, gamma, old_theta, theta, previous, grad = gamma, cand_main, theta, candidate, value, cand_grad
        if residual <= tol and relative <= np.sqrt(tol):
            converged = True
            break
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        direction = np.sum((extrapolated - theta) * (theta - old_theta))
        if q:
            direction += np.sum((ext_main - gamma) * (gamma - old_main))
        if restart or direction > 0:
            next_momentum = 1.0                         # adaptive restart of the momentum
        extrapolated = theta + (momentum - 1.0) / next_momentum * (theta - old_theta)
        ext_main = gamma + (momentum - 1.0) / next_momentum * (gamma - old_main)
        if next_momentum == 1.0:
            extrapolated, ext_main, ext_grad = theta, gamma, grad
        else:
            ext_grad = G @ stack(ext_main, extrapolated) - b
        momentum = next_momentum
    return theta, it, converged, gamma


def stationarity(record, theta, family, nb, lam, radius, L, gamma=None, groups=(), scale=(1.0, 1.0)):
    """Scaled proximal-gradient residual of the objective at (gamma, theta); 0 at a minimizer."""
    floor_g = unit_floors(scale)[0]
    g_main, g = gradient(record, theta, family, nb, gamma)
    step = prox(theta - g / L, lam / L, radius)
    if record.q == 0:
        return L * np.linalg.norm(theta - step) / max(floor_g, np.linalg.norm(g), lam)
    step_main = project_main(gamma - g_main / L, groups)
    return L * pair_norm(gamma - step_main, theta - step) / max(floor_g, pair_norm(g_main, g), lam)


def unit_floors(scale):
    """The floors of the stopping tests in the data's units.  The tests compare a residual with
    max(1, ||gradient||, lambda) and an objective change with max(1, |objective|), the floor 1 being in the units of
    unit-norm arms.  For features s times and rewards a times those of unit-norm data (scale = (s, a); the regret
    simulation has s = d, and a = d or 1), the parameter is a / s times, the gradient and lambda a s times and the
    objective a^2 times the unit-scale ones, so the floors become a s and a^2 and every iterate is the unit-scale one
    in the data's units.  Returns (gradient floor, objective floor, parameter factor a / s)."""
    s, a = scale
    return a * s, a * a, a / s


# ================================================================ 3b. proximal Newton (binomial)
def proximal_newton(record, lam, radius, start, family, nb, maxit, tol, L, newton_max=40,
                    start_main=None, groups=(), scale=(1.0, 1.0)):
    """Proximal Newton for the nuclear-norm penalized GLM likelihood.

    Each step solves min 0.5 v'Hv - (H vec(Theta) - g)'v + lam ||V||_* on the ball (`fista_pair` with
    (G, b) = (H, H vec(Theta) - g)), followed by Armijo backtracking on the objective.  H, the curvature-weighted Gram,
    is rebuilt exactly after a backtracked step, when the residual failed to halve, or after eight steps.  L is the
    Lipschitz constant of the gradient.  Returns (theta, refreshes, inner iterations, gamma)."""
    floor_g = unit_floors(scale)[0]
    q = record.q
    M, X, y = record.rows()
    C = record.main_rows()
    theta = prox(start, 0.0, radius)
    gamma = project_main(start_main, groups) if q else np.zeros(0)

    def objective(g_, A):
        eta = linear_predictor(A, M, X)
        if q:
            eta = eta + C @ g_
        return np.mean(psi(eta, family, nb) - y * eta) + lam * nuclear_norm(A)

    H = record.curvature_gram(theta, family, nb, gamma=gamma)
    LH = lambda_max(H)
    refresh, refreshes, inner, converged, res_old = False, 0, 0, False, np.inf
    for it in range(1, newton_max + 1):
        g_main, g = gradient(record, theta, family, nb, gamma)
        step = prox(theta - g / L, lam / L, radius)
        if q == 0:
            res = L * np.linalg.norm(theta - step) / max(floor_g, np.linalg.norm(g), lam)
        else:
            step_main = project_main(gamma - g_main / L, groups)
            res = L * pair_norm(gamma - step_main, theta - step) / max(floor_g, pair_norm(g_main, g), lam)
        if res <= tol:
            converged = True
            break
        refresh = refresh or res > 0.5 * res_old or it > 8
        res_old = res
        if refresh:
            H = record.curvature_gram(theta, family, nb, exact=True, gamma=gamma)
            LH = lambda_max(H)
            refreshes += 1
        b = H @ stack(gamma, theta) - stack(g_main, g)
        V, k, _, V_main = fista_pair(H, b, 0.0, lam, radius, theta, maxit,
                                     max(min(0.1 * res, tol), 1e-10), LH, gamma, groups, scale)
        inner += k
        delta, delta_main = V - theta, V_main - gamma
        model = np.sum(g * delta) + lam * (nuclear_norm(V) - nuclear_norm(theta))
        if q:
            model += np.sum(g_main * delta_main)
        f0, alpha, candidate, cand_main = objective(gamma, theta), 1.0, V, V_main
        while objective(cand_main, candidate) > f0 + 1e-4 * alpha * model and alpha > 1e-3:
            alpha /= 2.0
            candidate, cand_main = theta + alpha * delta, gamma + alpha * delta_main
        theta, gamma = candidate, cand_main
        refresh = alpha < 1.0
    if not converged:
        converged = stationarity(record, theta, family, nb, lam, radius, L, gamma, groups, scale) <= tol
    if not converged:
        raise RuntimeError("proximal Newton did not converge")
    return theta, refreshes, inner, gamma


# ================================================================ 3c. the estimator
def fit_nuclear(record, lam, radius, start, family, nb, tol=1e-7, maxit=20000, start_main=None, groups=(),
                scale=(1.0, 1.0)):
    """The penalized estimator on a prefix (eq. prefix-estimator): one FISTA solve (Gaussian) or proximal Newton
    (binomial), started at the previous estimate.  The minimizer is unique once the Gram matrix is non-singular;
    before that the solver's minimizer from the previous estimate is a fixed function of the ordered record, which is
    the measurable choice the appendix allows.  A fit that ends a relative tolerance outside a ball is put back on it.
    Returns a dict: Theta, Gamma (empty without main effects), lam, iterations, stationarity (and refreshes)."""
    slack = 10 * tol * unit_floors(scale)[2]
    s, q = record.n, record.q
    gamma_start = np.zeros(q) if start_main is None else start_main
    gram_max = lambda_max(record.G / s)
    L = max(curvature_bound(family, nb) * gram_max, 1e-8)
    if family == "gaussian":
        theta, iters, ok, gamma = fista_pair(record.G / s, record.cross / s, record.y2 / s, lam, radius,
                                             start, maxit, tol, gram_max, gamma_start, groups, scale)
        if not ok:
            raise RuntimeError("FISTA did not converge")
        info = dict(iterations=iters)
    else:
        theta, refreshes, iters, gamma = proximal_newton(record, lam, radius, start, family, nb,
                                                         maxit, tol, L, 40, gamma_start, groups, scale)
        info = dict(iterations=iters, refreshes=refreshes)
    info["stationarity"] = stationarity(record, theta, family, nb, lam, radius, L, gamma, groups, scale)
    unit = pair_norm(gamma, theta) if q > 0 else 0.0
    norm = np.linalg.norm(theta)
    if norm > radius + slack:
        if norm > radius + 10 * tol * max(radius, unit):
            raise RuntimeError("fit violates the Frobenius bound")
        theta = theta * (radius / norm)
    gamma, start_g = np.array(gamma, dtype=float), 0
    for size, R_g in groups:
        block = np.linalg.norm(gamma[start_g:start_g + size])
        if block > R_g + slack:
            if block > R_g + 10 * tol * max(R_g, unit):
                raise RuntimeError("fit violates a main-effect bound")
            gamma[start_g:start_g + size] *= R_g / block
        start_g += size
    return dict(Theta=theta, Gamma=gamma, lam=lam, **info)


# ================================================================ 4. the penalty
def main_only_fit(C, y, family, nb, groups, tol=1e-10, maxit=200000):
    """The fit with Theta = 0: the main-effect coefficients alone, on their balls, by projected gradient."""
    s = len(y)
    L = max(curvature_bound(family, nb) * lambda_max(C.T @ C / s), 1e-12)
    gamma = np.zeros(C.shape[1])
    for _ in range(maxit):
        grad = C.T @ (psi1(C @ gamma, family, nb) - y) / s
        new = project_main(gamma - grad / L, groups)
        if np.linalg.norm(new - gamma) <= tol * max(1.0, np.linalg.norm(new)):
            return new
        gamma = new
    raise RuntimeError("main-effects-only pilot fit did not converge")


def pilot_score(M, X, y, family, nb, C=None, groups=()):
    """||G_tau||_op, the operator norm of the interaction block of the score on the exploration sample at the
    main-effects-only fit: G_tau = (1/tau) sum_i {psi'(c_i' gamma_hat_0) - y_i} m_i x_i' (gamma_hat_0 = 0 without main
    effects).  Above this level the penalized fit's Theta is exactly zero."""
    s = len(y)
    if C is None or C.shape[1] == 0:
        mean0 = psi1(0.0, family, nb)
    else:
        mean0 = psi1(C @ main_only_fit(C, y, family, nb, groups), family, nb)
    score = np.linalg.norm(M.T @ (X * (mean0 - y)[:, None]) / s, 2)
    if not (np.isfinite(score) and score > 0):
        raise RuntimeError("degenerate pilot score")
    return float(score)


def penalty_levels(n, d1, d2, tau, score, c_lam):
    """The penalty levels of eq. lambdaTheta-main:

        lambda_s = c_lam sigma_hat sqrt((d1 + d2) log(n d1 d2) / s),   sigma_hat = score sqrt(tau / (d1 + d2)),

    with score = ||G_tau||_op from the exploration sample, so lambda_s is a fixed function of s after exploration.
    Returns (sigma_hat, levels), levels[s - 1] = lambda_s."""
    sigma_hat = score * np.sqrt(tau / (d1 + d2))
    s = np.arange(1, n + 1)
    return float(sigma_hat), c_lam * sigma_hat * np.sqrt((d1 + d2) * np.log(n * d1 * d2) / s)


# The penalty constant c chosen inside the run: 10-fold cross-validation on all observations so far (Qian, Ing and
# Liu, 2024; cv.glmnet's default, as in Chambaz, Zheng and van der Laan, 2017), first at s = max(2 tau, 200) and
# then whenever s doubles; c = CV_START before.  The smallest CV loss wins, ties going to the larger c.
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


def record_of_rows(M, X, y, C=None):
    """A Record holding the rows (M, X, y[, C])."""
    n, d1, d2 = len(y), M.shape[1], X.shape[1]
    q = 0 if C is None else C.shape[1]
    rec = Record(d1, d2, n, q)
    rec.M[:], rec.X[:], rec.y[:] = M, X, y
    Z = np.einsum("ni,nj->nij", M, X).reshape(n, d1 * d2)
    if q:
        rec.C[:] = C
        Z = np.hstack([C, Z])
    rec.G, rec.cross, rec.y2, rec.n = Z.T @ Z, Z.T @ y, float(y @ y), n
    return rec


def cv_constant(record, base, theta, gamma, family, nb, fit_args, seed):
    """The penalty constant by CV_FOLDS-fold cross-validation on the record's rows: for every fold and every c in
    CV_GRID the estimator is fitted on the other folds at lambda = c base[s' - 1] (s' rows) and scored on the fold
    by squared error (Gaussian) or deviance (binomial).  Returns (the c with the smallest loss, the losses)."""
    radius, tol, maxit, groups, scale = fit_args
    M, X, y = record.rows()
    C = record.main_rows() if record.q else None
    fold = np.random.default_rng(seed).permutation(record.n) % CV_FOLDS
    loss = np.zeros(len(CV_GRID))
    for f in range(CV_FOLDS):
        train, test = fold != f, fold == f
        rec = record_of_rows(M[train], X[train], y[train], None if C is None else C[train])
        start, start_main = theta, gamma
        for j, c in enumerate(CV_GRID):
            fit = fit_nuclear(rec, c * base[rec.n - 1], radius, start, family, nb, tol, maxit, start_main, groups, scale)
            start, start_main = fit["Theta"], fit["Gamma"]
            eta = linear_predictor(start, M[test], X[test])
            if C is not None:
                eta = eta + C[test] @ start_main
            if family == "gaussian":
                loss[j] += np.sum((y[test] - eta) ** 2)
            else:
                loss[j] += np.sum(psi(eta, family, nb) - y[test] * eta)
    return CV_GRID[int(np.argmin(loss))], loss


# ================================================================ 5. the exploration length
TAU_MIN = 5              # the smallest exploration sample


def tau_rule(n, d1, d2, c_tau, q=0):
    """The exploration length tau = ceil(c_tau (d1 + d2 + q) log(n d1 d2)): the rate of the paper's sufficient
    length (eq. rank-leading-calibration-main) with the rank and the remaining constants absorbed into c_tau.  q is the
    number of unpenalized main-effect coefficients, which the pilot fits on the exploration sample alone."""
    return int(np.ceil(c_tau * (d1 + d2 + q) * np.log(n * d1 * d2)))


def tau_feasible(tau, n, q=0):
    """tau is usable when the pilot can be fitted (at least TAU_MIN rows and more than q) and rounds remain."""
    return max(TAU_MIN, q + 1) <= tau < n


# ================================================================ 6. Algorithm 1
def run_algorithm1(env, tau, c_lam, tol=1e-7, maxit=20000):
    """Algorithm 1 on one environment with exploration length tau and penalty constant c_lam (a number or "cv").

    I.  Exploration (rounds 1..tau): pull the pre-drawn uniform arm; compute the pilot score and the penalty levels;
        fit the pilot at lambda_tau.
    II. Exploitation (rounds tau+1..n): pull the maximizer of m' Theta_hat x (+ c' gamma_hat) over the K candidates
        and refit at lambda_s.  With c_lam = "cv" the constant is chosen at the checkpoints by `cv_constant`, the fold
        split seeded by (env.seed, s).
    The environment may give main effects (main, main_groups), its Frobenius radius (default 2 amp) and its units
    (arm_scale, reward_scale; see `unit_floors`).  Returns the per-round trace (arm, reward, regret, exploration,
    theta_error, fitted_rank, lam, stationarity, cumulative_regret), the pilot score, sigma_hat, the penalty levels,
    the final estimate and, with "cv", the chosen constants [(s, c), ...]."""
    n, d1, d2 = env.n, env.d1, env.d2
    family, nb = env.family, env.nb
    radius = env.radius if hasattr(env, "radius") else 2.0 * env.amp
    main = getattr(env, "main", None)
    q = 0 if main is None else main.shape[2]
    groups = tuple(env.main_groups) if q else ()
    scale = (getattr(env, "arm_scale", 1.0), getattr(env, "reward_scale", 1.0))
    rank_tol = 1e-7 * unit_floors(scale)[2]
    Theta_link = env.link_truth()
    trace = dict(arm=np.zeros(n, int), reward=np.zeros(n), regret=np.zeros(n),
                 exploration=np.arange(n) < tau, theta_error=np.full(n, np.nan),
                 fitted_rank=np.full(n, -1), lam=np.full(n, np.nan), stationarity=np.full(n, np.nan))
    record = Record(d1, d2, n, q)
    theta, gamma = np.zeros((d1, d2)), np.zeros(q)

    def observe(t, a, means):
        y = env.reward(t, a, means)
        trace["arm"][t], trace["reward"][t], trace["regret"][t] = a, y, means.max() - means[a]
        w = None
        if family != "gaussian":                       # curvature at the current estimate
            u = env.candidates[t, a] @ theta @ env.contexts[t]
            if q:
                u = u + main[t, a] @ gamma
            w = psi2(u, family, nb)
        record.add(env.candidates[t, a], env.contexts[t], y, w, main[t, a] if q else None)

    def refit(lam):
        nonlocal theta, gamma
        fit = fit_nuclear(record, lam, radius, theta, family, nb, tol, maxit, gamma, groups, scale)
        theta, gamma = fit["Theta"], fit["Gamma"]
        t = record.n - 1
        trace["lam"][t], trace["stationarity"][t] = lam, fit["stationarity"]
        if Theta_link is not None:
            trace["theta_error"][t] = np.linalg.norm(theta - Theta_link)
        trace["fitted_rank"][t] = int(np.sum(np.linalg.svd(theta, compute_uv=False) > rank_tol))

    # ---- I. pure exploration
    for t in range(tau):
        observe(t, env.exploration_arms[t], env.means(t))
    M, X, y = record.rows()
    score = pilot_score(M, X, y, family, nb, record.main_rows() if q else None, groups)
    cv = c_lam == "cv"
    if cv:                                                      # lambda_s = c base[s - 1], c chosen by CV
        sigma_hat, base = penalty_levels(n, d1, d2, tau, score, 1.0)
        c, checkpoints, chosen = CV_START, cv_checkpoints(tau, n), []
        levels = c * base
    else:
        sigma_hat, levels = penalty_levels(n, d1, d2, tau, score, c_lam)
    refit(levels[tau - 1])                                      # the pilot estimator

    # ---- II. exploitation with the prefix estimator
    for t in range(tau, n):
        means = env.means(t)
        scores = env.candidates[t] @ (theta @ env.contexts[t])
        if q:
            scores = scores + main[t] @ gamma
        observe(t, int(np.argmax(scores)), means)
        s = record.n
        if cv and s in checkpoints:                             # choose c again on the s observations
            c, _ = cv_constant(record, base, theta, gamma, family, nb, (radius, tol, maxit, groups, scale),
                               [getattr(env, "seed", 0), s])
            chosen.append((s, c))
            levels[s - 1:] = c * base[s - 1:]
        refit(levels[s - 1])
    trace["cumulative_regret"] = np.cumsum(trace["regret"])
    return dict(tau=tau, c_lam=c_lam, trace=trace, score=score, sigma_hat=sigma_hat, levels=levels,
                Theta=theta, Gamma=gamma, cv=chosen if cv else None)
