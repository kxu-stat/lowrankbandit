"""The tuning rules of Algorithm 1: the rates of the theory, with constants.

  exploration length   tau = ceil(c_tau (d1 + d2 + p0) log(n d1 d2))
  penalty level        lambda_s = c sigma_hat sqrt((d1 + d2) log(n d1 d2) / s),   sigma_hat = score sqrt(tau / (d1 + d2)),
                       score = ||interaction block of the score on S_o||_op (at the main-effects-only fit)
  penalty constant     c fixed, or chosen by 10-fold cross-validation on the rows so far, first at
                       s = max(2 tau, 200) and again whenever s doubles (1/8 before the first choice)
"""
import numpy as np
from .model import mean_function, softplus
from .estimator import fista, proximal_newton, top_eigenvalue, norm2

C_TAU = 1 / 20
CV_GRID = (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)   # largest first: each fit starts at the previous one
CV_START = 1 / 8
CV_FOLDS = 10
CV_MIN = 200


def tau_rule(n, d1, d2, c_tau=C_TAU, p0=0):
    """Exploration length tau = ceil(c_tau (d1 + d2 + p0) log(n d1 d2)), n the horizon (all rounds)."""
    return int(np.ceil(c_tau * (d1 + d2 + p0) * np.log(n * d1 * d2)))


def main_only_fit(C, y, family, trials, radii=None, sizes=None, tol=1e-10, maxit=200000):
    """The fit with Theta = 0: the main-effect coefficients alone, on their balls, by projected gradient."""
    s = len(y)
    L = max((1.0 if family == "gaussian" else 0.25 * trials) * top_eigenvalue(C.T @ C / s), 1e-12)
    gamma = np.zeros(C.shape[1])
    for _ in range(maxit):
        new = gamma - C.T @ (mean_function(C @ gamma, family, trials) - y) / s / L
        if radii is not None:
            start = 0
            for size, R in zip(sizes, radii):
                norm = norm2(new[start:start + size])
                if R is not None and norm > R:
                    new[start:start + size] *= R / norm
                start += size
        if norm2(new - gamma) <= tol * max(1.0, norm2(new)):
            return new
        gamma = new
    raise RuntimeError("the main-effects-only fit on the exploration rows did not converge: under the binomial family "
                       "the rows are probably separable; use a longer exploration (tau or c_tau)")


def pilot_score(Z, y, d1, d2, family, trials, radius=None, sizes=None):
    """||G_tau||_op: the operator norm of the interaction block of the score on the exploration rows, at Theta = 0
    and the main-effects-only fit (at theta = 0 when there are no main effects)."""
    s, p0 = len(y), Z.shape[1] - d1 * d2
    if p0 > 0:
        gamma = main_only_fit(Z[:, :p0], y, family, trials, None if radius is None else radius[:-1], sizes)
        g = Z.T @ (mean_function(Z[:, :p0] @ gamma, family, trials) - y) / s
    else:
        g = Z.T @ (mean_function(0.0, family, trials) - y) / s
    score = np.linalg.norm(g[p0:].reshape(d1, d2, order="F"), 2)
    if not np.isfinite(score):
        raise RuntimeError("the pilot score is not finite")
    return float(score)


def penalty_level(c, score, tau, n, d1, d2, s):
    """lambda_s = c sigma_hat sqrt((d1 + d2) log(n d1 d2) / s), sigma_hat = score sqrt(tau / (d1 + d2))."""
    sigma_hat = score * np.sqrt(tau / (d1 + d2))
    return c * sigma_hat * np.sqrt((d1 + d2) * np.log(n * d1 * d2) / max(s, 1))


def cv_checkpoints(tau, n_end):
    """The sample sizes at which c is chosen: s_1 = max(2 tau, 200), 2 s_1, 4 s_1, ... below n_end."""
    out, s = [], max(2 * tau, CV_MIN)
    while s < n_end:
        out.append(s)
        s *= 2
    return out


def cv_constant(Z, y, base, start, d1, d2, family, trials, radius=None, sizes=None):
    """The penalty constant by 10-fold cross-validation on the rows (Z, y): row i is in fold i mod 10.  Each c of
    CV_GRID (largest first) is fitted on the other folds at lambda = c base(s') and scored on the fold by squared error
    (Gaussian) or deviance (binomial); the smallest total loss wins.  base(s) is the penalty level at c = 1."""
    fold = np.arange(len(y)) % CV_FOLDS
    loss = np.zeros(len(CV_GRID))
    for f in range(CV_FOLDS):
        train, test = fold != f, fold == f
        Zt, yt = Z[train], y[train]
        s = len(yt)
        G, b = Zt.T @ Zt / s, Zt.T @ yt / s                    # the same for every c: computed once per fold
        L = top_eigenvalue(G)
        x = start
        for k, c in enumerate(CV_GRID):
            if family == "gaussian":
                x = fista(G, b, c * base(s), d1, d2, x, L=L, radius=radius, sizes=sizes)
            else:
                x = proximal_newton(Zt, yt, c * base(s), d1, d2, x, trials, radius, L_max=0.25 * trials * L,
                                    sizes=sizes)
            u = Z[test] @ x
            if family == "gaussian":
                loss[k] += np.sum((y[test] - u) ** 2)
            else:
                loss[k] += np.sum(trials * softplus(u) - y[test] * u)
    return CV_GRID[int(np.argmin(loss))]
