"""The nuclear-norm-penalized GLM estimator of Algorithm 1:

    theta_hat in argmin_{theta in C_R}  (1/s) sum_i [psi(z_i' theta) - y_i z_i' theta] + lam ||Theta||_*,

Theta the interaction block of theta; the main effects are not penalized.  C_R is an optional domain:
each main-effect group in a Euclidean ball and Theta in a Frobenius ball, radius = (R_1, ..., R_G, R_Theta); with
radius None there is no domain.  Gaussian family: FISTA with adaptive restart.  Binomial family: proximal Newton
(each step a FISTA solve of the quadratic model, Armijo backtracking).
"""
import math
import numpy as np
from scipy.linalg import lapack
from .model import sigmoid, softplus, variance_function, vec


def top_eigenvalue(G):
    """The largest eigenvalue of the symmetric matrix G."""
    return float(np.linalg.eigvalsh(G)[-1])


def norm2(v):
    """The Euclidean norm of a vector: np.linalg.norm(v) without its overhead, which counts in the solvers' loops."""
    return math.sqrt(v.dot(v))


def svd(M):
    """The thin SVD M = U diag(s) Vt by LAPACK's dgesdd: what np.linalg.svd computes, without its overhead."""
    U, s, Vt, info = lapack.dgesdd(M, full_matrices=0)
    if info != 0:
        raise np.linalg.LinAlgError("the SVD did not converge")
    return U, s, Vt


def nuclear_norm(theta, d1, d2):
    p0 = theta.size - d1 * d2
    return float(svd(theta[p0:].reshape(d1, d2, order="F"))[1].sum())


def prox(u, step, lam, d1, d2, radius=None, sizes=None):
    """Proximal map of step * lam ||Theta||_* on C_R: the singular values of the interaction block are soft-thresholded
    at step * lam and scaled into the Frobenius ball; each main-effect group is projected onto its ball."""
    p0 = u.size - d1 * d2
    Uu, sv, Vt = svd(u[p0:].reshape(d1, d2, order="F"))
    sv = np.maximum(sv - step * lam, 0.0)
    if radius is not None and radius[-1] is not None:
        norm = norm2(sv)
        if norm > radius[-1]:
            sv = sv * (radius[-1] / norm)
    inter = vec((Uu * sv) @ Vt)
    if p0 == 0:
        return inter
    if sizes is None:
        sizes = (p0,)                                             # one main-effect group
    blocks, start = [], 0
    for g, size in enumerate(sizes):
        block = u[start:start + size].copy()
        if radius is not None and radius[g] is not None:
            norm = norm2(block)
            if norm > radius[g]:
                block = block * (radius[g] / norm)
        blocks.append(block)
        start += size
    return np.concatenate(blocks + [inter])


def fista(G, b, lam, d1, d2, x0, L=None, iters=400, tol=1e-6, radius=None, sizes=None):
    """min 1/2 x'Gx - b'x + lam ||Theta(x)||_* on C_R by FISTA with adaptive restart, started at x0.
    L: the largest eigenvalue of G (computed when None)."""
    if L is None:
        L = top_eigenvalue(G)
    L = max(L, 1e-12)
    x = x0.copy()
    y = x.copy()
    tk = 1.0
    for _ in range(iters):
        xn = prox(y - (G @ y - b) / L, 1.0 / L, lam, d1, d2, radius, sizes)
        if norm2(L * (y - xn)) <= tol * max(1.0, L * norm2(xn)):
            return xn
        if np.dot(y - xn, xn - x) > 0:
            tk = 1.0                                              # adaptive restart
        tn = (1.0 + math.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
        y = xn + ((tk - 1.0) / tn) * (xn - x)
        x, tk = xn, tn
    return x


def proximal_newton(Z, y, lam, d1, d2, x0, trials, radius=None, H=None, LH=None, L_max=None,
                    iters=30, tol=1e-6, inner_iters=1000, inner_tol=1e-8, sizes=None):
    """min (1/s) sum_i [trials log(1 + e^{z_i'x}) - y_i z_i'x] + lam ||Theta(x)||_* on C_R by proximal Newton.
    H, the curvature-weighted Gram matrix, may be passed in (possibly computed at an earlier estimate); it is rebuilt
    after a backtracked step, when the residual fails to halve, and after eight steps.  Returns x."""
    s = len(y)
    if L_max is None:
        L_max = 0.25 * trials * top_eigenvalue(Z.T @ Z / s)
    L_max = max(L_max, 1e-12)

    def objective(u, nuc):                              # u = Z x, nuc = ||Theta(x)||_*
        return float(np.mean(trials * softplus(u) - y * u)) + lam * nuc

    x = x0.copy()
    u = Z @ x
    nuc = nuclear_norm(x, d1, d2)
    f = objective(u, nuc)                               # u, nuc and f go with x: each computed once per point
    refresh, res_old = H is None, np.inf
    for it in range(1, iters + 1):
        g = Z.T @ (trials * sigmoid(u) - y) / s
        xp = prox(x - g / L_max, 1.0 / L_max, lam, d1, d2, radius, sizes)
        res = norm2(L_max * (x - xp)) / max(1.0, L_max * norm2(xp))
        if res <= tol:
            return xp
        refresh = refresh or res > 0.5 * res_old or it > 8
        res_old = res
        if refresh:
            w = variance_function(u, "binomial", trials)
            H = Z.T @ (Z * w[:, None]) / s
            LH = top_eigenvalue(H)
        if LH is None:
            LH = top_eigenvalue(H)
        v = fista(H, H @ x - g, lam, d1, d2, x, L=LH, iters=inner_iters, tol=max(inner_tol, 0.1 * res),
                  radius=radius, sizes=sizes)
        step = v - x
        alpha, xn, un, nucn = 1.0, v, Z @ v, nuclear_norm(v, d1, d2)
        model = float(g @ step) + lam * (nucn - nuc)
        fn = objective(un, nucn)
        while fn > f + 1e-4 * alpha * model and alpha > 1e-3:
            alpha *= 0.5
            xn = x + alpha * step
            un, nucn = Z @ xn, nuclear_norm(xn, d1, d2)
            fn = objective(un, nucn)
        x, u, nuc, f = xn, un, nucn, fn
        refresh = alpha < 1.0
    g = Z.T @ (trials * sigmoid(u) - y) / s
    return prox(x - g / L_max, 1.0 / L_max, lam, d1, d2, radius, sizes)


def fit_nuclear(Z, y, lam, d1, d2, family, trials=1, start=None, radius=None, sizes=None, tol=1e-6, iters=400):
    """The nuclear-norm-penalized GLM estimator on given rows.

    Minimizes the average negative log-likelihood of (Z, y) plus lam ||Theta||_* (the main effects unpenalized), by
    FISTA under the Gaussian family and by proximal Newton under the binomial family.

    Parameters
    ----------
    Z : array of shape (s, p)
        The design rows z = (c, m, x, vec(m x')), main effects first.
    y : array of shape (s,)
        The responses.
    lam : float
        The penalty level.
    d1, d2 : int
        The dimensions of the interaction Theta (its d1 d2 coordinates are the last of z).
    family : {"gaussian", "binomial"}
        The reward model.
    trials : int
        The number of trials of the binomial family.
    start : array of shape (p,), optional
        The starting point (zero by default).
    radius : tuple of float, optional
        Radii of the domain C_R for the main-effect groups (`sizes`) and Theta.
    sizes : tuple of int, optional
        The sizes of the main-effect groups.
    tol, iters : float, int
        The stopping tolerance and the iteration cap of the solver.

    Returns
    -------
    array of shape (p,)
        The estimate: main effects first, then vec(Theta).
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    s, p = Z.shape
    x0 = np.zeros(p) if start is None else np.asarray(start, dtype=float)
    if family == "gaussian":
        return fista(Z.T @ Z / s, Z.T @ y / s, lam, d1, d2, x0, iters=iters, tol=tol, radius=radius, sizes=sizes)
    return proximal_newton(Z, y, lam, d1, d2, x0, trials, radius, tol=tol, sizes=sizes)
