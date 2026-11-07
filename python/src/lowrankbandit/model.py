"""The model: the exponential family, the design rows, and the geometry of rank-r matrices.

Round t shows K candidate arms with features m_{t,a} (length d1), one context x_t (length d2) and, optionally,
further covariates c_{t,a} (length q).  The design row of arm a is

    z_{t,a} = (c_{t,a}, m_{t,a}, x_t, vec(m_{t,a} x_t'))        (main effects first, then the interaction),

vec stacking columns, so <z_{t,a}, theta> = c' gamma + m' theta_m + x' theta_x + m' Theta x.  The leading p0
coordinates are the unpenalized main effects (p0 = q + d1 + d2 with main effects, q without; the policies leave the
main effects out unless asked).  The reward follows the
canonical GLM with linear predictor u = <z, theta>: Gaussian (psi(u) = u^2 / 2) or binomial with `trials` trials
(psi(u) = trials log(1 + e^u); trials = 1 is logistic regression).
"""
import numpy as np


# ------------------------------------------------------------------ the family
def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def softplus(u):
    """log(1 + e^u), computed without overflow."""
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))


def mean_function(u, family, trials):
    """psi'(u), the mean reward at linear predictor u."""
    return u if family == "gaussian" else trials * sigmoid(u)


def variance_function(u, family, trials):
    """psi''(u), the variance of the reward at linear predictor u (1 under the Gaussian family)."""
    if family == "gaussian":
        return np.ones_like(u, dtype=float) if np.ndim(u) else 1.0
    q = sigmoid(u)
    return trials * q * (1.0 - q)


def check_family(family, trials):
    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be 'gaussian' or 'binomial'")
    if family == "binomial" and (int(trials) != trials or trials < 1):
        raise ValueError("trials must be a positive integer")


# ------------------------------------------------------------------ design rows
def vec(M):
    """Stack the columns of a matrix into one vector (column-major, as R's `as.vector`)."""
    return np.asarray(M).reshape(-1, order="F")


def design_rows(arms, context, covariates=None, main_effects=False):
    """The K x p matrix of design rows z_{t,a} = (c_{t,a}, m_{t,a}, x_t, vec(m_{t,a} x_t')).

    arms: K x d1, context: length d2, covariates: K x q or None."""
    arms = np.atleast_2d(np.asarray(arms, dtype=float))
    context = np.asarray(context, dtype=float).ravel()
    K = arms.shape[0]
    blocks = []
    if covariates is not None:
        blocks.append(np.asarray(covariates, dtype=float).reshape(K, -1))
    if main_effects:
        blocks.append(arms)
        blocks.append(np.tile(context, (K, 1)))
    blocks.append(np.array([np.kron(context, m) for m in arms]))      # vec(m x') = kron(x, m)
    return np.hstack(blocks)


def main_groups(d1, d2, n_covariates, main_effects):
    """The sizes of the main-effect groups (each group may have its own radius): (q, d1, d2) without the empty ones."""
    sizes = []
    if n_covariates > 0:
        sizes.append(n_covariates)
    if main_effects:
        sizes += [d1, d2]
    return tuple(sizes)


def interaction_loading(A_Theta, p0):
    """The loading of a target that involves the interaction only.

    Parameters
    ----------
    A_Theta : array of shape (d1, d2)
        The target is <A_Theta, Theta> = sum_ij (A_Theta)_ij Theta_ij.
    p0 : int
        The number of main-effect coordinates (`data["p0"]`).

    Returns
    -------
    array of shape (p0 + d1 d2,)
        The loading (0, vec(A_Theta)).
    """
    return np.concatenate([np.zeros(p0), vec(A_Theta)])


# ------------------------------------------------------------------ rank-r geometry
def interaction_block(theta, d1, d2):
    """The d1 x d2 interaction matrix Theta inside the stacked parameter."""
    p0 = theta.size - d1 * d2
    return theta[p0:].reshape(d1, d2, order="F")


def truncate(theta, d1, d2, r):
    """Rank-r truncation of the interaction block (main effects kept).  Returns (theta_r, U, V)."""
    p0 = theta.size - d1 * d2
    U, s, Vt = np.linalg.svd(interaction_block(theta, d1, d2), full_matrices=False)
    U, s, V = U[:, :r], s[:r], Vt[:r].T
    out = theta.copy()
    out[p0:] = vec((U * s) @ V.T)
    return out, U, V


def tangent_project(v, U, V, d1, d2):
    """Projection onto R^{p0} x T(U, V): the main effects are kept and the interaction block M becomes
    U U'M + M V V' - U U'M V V'."""
    p0 = v.size - d1 * d2
    out = np.array(v, dtype=float)
    M = v[p0:].reshape(d1, d2, order="F")
    out[p0:] = vec(U @ (U.T @ M) + (M @ V) @ V.T - U @ ((U.T @ M) @ V) @ V.T)
    return out


def tangent_project_rows(Z, U, V, d1, d2):
    """tangent_project applied to every row of Z."""
    return np.array([tangent_project(z, U, V, d1, d2) for z in Z])


def tangent_basis(U, V, d1, d2, p0):
    """Orthonormal columns spanning R^{p0} x T(U, V); p_T = p0 + r(d1 + d2 - r) columns."""
    r = U.shape[1]
    Uc = np.linalg.svd(U, full_matrices=True)[0][:, r:]          # an orthonormal basis of the complement of U
    Vc = np.linalg.svd(V, full_matrices=True)[0][:, r:]
    cols = []
    for L, R in ((U, V), (U, Vc), (Uc, V)):
        for i in range(L.shape[1]):
            for j in range(R.shape[1]):
                cols.append(vec(np.outer(L[:, i], R[:, j])))
    Q = np.zeros((p0 + d1 * d2, p0 + len(cols)))
    Q[:p0, :p0] = np.eye(p0)
    Q[p0:, p0:] = np.column_stack(cols)
    return Q


def tangent_dimension(d1, d2, r, p0):
    return p0 + r * (d1 + d2 - r)
