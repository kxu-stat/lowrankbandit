"""Algorithm 2: two-track cross-fitted inference for a linear functional mu* = <A, theta*>.

The data are the records of TwoTrackPolicy: the shared exploration S_o and the two tracks of n_tr rounds each.
For fold j = 1, 2 (track j' = the other track):

  Step 2  fit on S_o and the first n0 rounds of track j, truncate the interaction to rank r0: theta_breve, its
          tangent space and orthonormal basis Q.  Refit on the remaining n_ref = n_tr - n0 rounds of track j by the
          predictable recursion with ridge lambda_ref = n_ref kappa / (2 C_ref log n_ref), and truncate again:
          theta_hat^(j) and its projector P_j.
  Step 3  over all of track j', the calibration recursion with
          lambda_cal = n_tr kappa / (2 C_cal log n_tr) gives the weights omega_t, and
              mu^(j) = <A, theta_hat^(j)> + sum_t {y_t - psi'(<z_t, theta_hat^(j)>)} omega_t,
              (V^(j))^2 = sum_t {y_t - psi'(<z_t, theta_hat^(j)>)}^2 omega_t^2.
  Step 4  mu_hat = (mu^(1) + mu^(2)) / 2, V_N = sqrt((V^(1))^2 + (V^(2))^2) / 2, and the interval
          mu_hat -+ Phi^{-1}(1 - delta/2) V_N at the level 1 - delta.

kappa, the information constant of both recursions, is estimated from S_o: the average eigenvalue of the
curvature-weighted exploration Gram matrix on the tangent coordinates.
"""
import numpy as np
from scipy.stats import norm
from .model import (mean_function, variance_function, softplus, truncate, tangent_basis, tangent_project,
                    tangent_project_rows, tangent_dimension)
from .estimator import fista, proximal_newton
from .tuning import CV_START, penalty_level

C_REF = 1 / 15          # the refit's constant: the theorem needs C_ref >= 1/15
C_CAL = 1.0             # the calibration's constant: the theorem needs C_cal >= 2/15
C_SPLIT = 1 / 3         # the constant of the two rules for n0
N0_RULES = ("equal", "information", "design")


# ------------------------------------------------------------------ the split n0
def split_design(data, kappa_D, cap=0.75):
    """The design-stage split: the equal split n_tr / 2, lengthened when half a track is shorter than the burn-in
    C_SPLIT p_T log(n d1 d2) / kappa_D, and at most cap n_tr.  kappa_D is a planning value of the information per
    round and coordinate (for example 1 for the Gaussian family with E[z z'] = I)."""
    n_tr = data["n_tr"]
    need = int(np.ceil(C_SPLIT * data["p_T"] * np.log(data["horizon"] * data["d1"] * data["d2"]) / kappa_D))
    return min(max(need, n_tr // 2), int(np.floor(cap * n_tr)))


def split_information(data):
    """The information rule: n0 = ceil(C_SPLIT p_T log(n d1 d2) / kappa0) within [n_tr / 4, 3 n_tr / 4], kappa0 the
    information per round and coordinate of S_o, tr(G_w) / (tau p), G_w the curvature-weighted exploration Gram matrix
    at the pilot estimate."""
    n_tr = data["n_tr"]
    Z = data["Z_ex"]
    w = variance_function(Z @ data["pilot"], data["family"], data["trials"])
    kappa0 = float(np.sum(w * np.sum(Z ** 2, axis=1)) / (data["tau"] * Z.shape[1]))
    need = int(np.ceil(C_SPLIT * data["p_T"] * np.log(data["horizon"] * data["d1"] * data["d2"]) / kappa0))
    return min(max(need, int(np.ceil(n_tr / 4)), 10), int(np.floor(3 * n_tr / 4)))


def choose_n0(data, n0, kappa_D=None):
    """n0 from a rule name ("equal", "information", "design") or an integer."""
    if isinstance(n0, str):
        if n0 == "equal":
            return data["n_tr"] // 2
        if n0 == "information":
            return split_information(data)
        if n0 == "design":
            if kappa_D is None:
                raise ValueError("n0 = 'design' needs kappa_D, a planning value of the information per round and "
                                 "coordinate")
            return split_design(data, kappa_D)
        raise ValueError("n0 must be an integer or one of %s" % (N0_RULES,))
    n0 = int(n0)
    if not 2 <= n0 <= data["n_tr"] - 2:
        raise ValueError("n0 must lie in [2, n_tr - 2] = [2, %d]" % (data["n_tr"] - 2))
    return n0


# ------------------------------------------------------------------ the rank
def gic_rank(Z, y, theta, d1, d2, family, trials):
    """The rank minimizing L_s(theta_r) + (log s / s) r (d1 + d2 - r), theta_r the rank-r truncation of theta and
    L_s the average negative log-likelihood (Gaussian: 0.5 log of the mean squared residual); at least 1."""
    s = len(y)
    crit = []
    for r in range(min(d1, d2) + 1):
        u = Z @ truncate(theta, d1, d2, r)[0]
        if family == "gaussian":
            loss = 0.5 * float(np.log(np.mean((y - u) ** 2)))
        else:
            loss = float(np.mean(trials * softplus(u) - y * u))
        crit.append(loss + np.log(s) / s * r * (d1 + d2 - r))
    return max(int(np.argmin(crit)), 1)


# ------------------------------------------------------------------ Step 2
def c_in_force(data, j, s):
    """The penalty constant track j used at sample size s."""
    if data["c_lam"] != "cv":
        return data["c_lam"]
    c = CV_START
    for s_k, c_k in data["cv"][j]:
        if s_k <= s:
            c = c_k
    return c


def fold_fit(data, j, n0, rank, C_ref=C_REF):
    """Step 2 for fold j: the preliminary fit on S_o and the first n0 rounds of track j, its rank truncation, and the
    predictable refit on the rest of track j.  Returns theta_hat^(j), its frame (U, V), kappa, lambda_ref and the
    rank used."""
    d1, d2, p0 = data["d1"], data["d2"], data["p0"]
    family, trials, radius, sizes = data["family"], data["trials"], data["radius"], data["sizes"]
    tau, n_tr = data["tau"], data["n_tr"]
    Z, y = data["Z"][j], data["y"][j]
    s = tau + n0
    lam = penalty_level(c_in_force(data, j, s), data["score"], tau, data["horizon"], d1, d2, s)
    Zs, ys = np.vstack([data["Z_ex"], Z[:n0]]), np.concatenate([data["y_ex"], y[:n0]])
    if family == "gaussian":
        breve = fista(Zs.T @ Zs / s, Zs.T @ ys / s, lam, d1, d2, data["pilot"], iters=600, tol=1e-8,
                      radius=radius, sizes=sizes)
    else:
        breve = proximal_newton(Zs, ys, lam, d1, d2, data["pilot"], trials, radius, tol=1e-8, sizes=sizes)
    r = gic_rank(Zs, ys, breve, d1, d2, family, trials) if rank == "gic" else int(rank)
    breve, U, V = truncate(breve, d1, d2, r)
    Q = tangent_basis(U, V, d1, d2, p0)
    # kappa: the average eigenvalue of the curvature-weighted exploration Gram on the tangent coordinates
    w_ex = variance_function(data["Z_ex"] @ breve, family, trials) * np.ones(tau)
    G_w = data["Z_ex"].T @ (data["Z_ex"] * w_ex[:, None])
    kappa = float(np.trace(Q.T @ (G_w / tau) @ Q) / Q.shape[1])
    # the predictable refit on rounds n0 + 1, ..., n_tr
    n_ref = n_tr - n0
    lam_ref = n_ref / (2.0 * C_ref * np.log(max(n_ref, 3))) * kappa
    R, acc = np.eye(Q.shape[1]), np.zeros(Q.shape[1])
    for t in range(n0, n_tr):
        q = Q.T @ Z[t]
        u = Z[t] @ breve
        c = variance_function(u, family, trials)
        w = (R.T @ q) / (lam_ref + c * (q @ q))
        R = R - c * np.outer(q, w)
        acc += w * (y[t] - mean_function(u, family, trials))
    theta_hat, U, V = truncate(breve + Q @ acc, d1, d2, r)
    return dict(theta=theta_hat, U=U, V=V, kappa=kappa, lam_ref=float(lam_ref), rank=r)


# ------------------------------------------------------------------ Step 3
def fold_calibrate(data, j, fold, As, C_cal=C_CAL):
    """Step 3 for fold j over the other track: the calibration recursion for every loading (rows of As) at once.
    Returns mu^(j) and V^(j) for every loading, and lambda_cal."""
    d1, d2, family, trials, n_tr = data["d1"], data["d2"], data["family"], data["trials"], data["n_tr"]
    Z, y = data["Z"][1 - j], data["y"][1 - j]
    U, V, theta = fold["U"], fold["V"], fold["theta"]
    PZ = tangent_project_rows(Z, U, V, d1, d2)
    qq = np.sum(PZ * PZ, axis=1)
    rho = tangent_project_rows(As, U, V, d1, d2).T                       # p x L: the unmatched loadings
    lam_cal = n_tr / (2.0 * C_cal * np.log(max(n_tr, 3))) * fold["kappa"]
    u = Z @ theta
    c = variance_function(u, family, trials) * np.ones(n_tr)
    omega = np.zeros((n_tr, len(As)))
    for t in range(n_tr):                                                # weights fixed before y_t is used
        w = (Z[t] @ rho) / (lam_cal + c[t] * qq[t])
        rho = rho - c[t] * np.outer(PZ[t], w)
        omega[t] = w
    res = y - mean_function(u, family, trials)
    mu = As @ theta + res @ omega
    V2 = (res ** 2) @ (omega ** 2)
    return mu, np.sqrt(V2), float(lam_cal)


# ------------------------------------------------------------------ Algorithm 2
def two_track_inference(data, A, rank, n0="equal", kappa_D=None, level=0.95, C_ref=1 / 15, C_cal=1.0):
    """Steps 2-4 of Algorithm 2: confidence intervals for mu* = <A, theta*>.

    Parameters
    ----------
    data : dict
        The records of a `TwoTrackPolicy` after its last round (`policy.data()`).
    A : array of shape (p,) or (L, p)
        The loading of the target: main effects first, then vec(A_Theta) (see `interaction_loading`).  A matrix
        gives L targets at once; they share Step 2.
    rank : int or "gic"
        The rank r0 of the interaction, or "gic" to choose it in each fold by an information criterion.
    n0 : {"equal", "information", "design"} or int
        The rounds of each track used for the preliminary fit; the remaining rounds go to the refit.  "equal":
        n_tr / 2.  "information": from the information per round estimated on the exploration.  "design": the equal
        split, lengthened when half a track is short for the planning value `kappa_D`.
    kappa_D : float, optional
        The planning value of the information per round and coordinate, for `n0="design"`.
    level : float
        The confidence level.
    C_ref, C_cal : float
        The constants of the refit and calibration ridges (the theory needs C_ref >= 1/15 and C_cal >= 2/15).

    Returns
    -------
    dict
        `estimate`, `se`, `lower`, `upper`: numbers for one loading, arrays for several.  Also `n0`, and for each
        fold the rank, `kappa`, `lambda_ref`, `lambda_cal`, the fold estimates `fold_estimate` and standard errors
        `fold_se`, and the refitted parameter `theta`.

    Notes
    -----
    The intervals are valid for data collected by `TwoTrackPolicy` only: each track's choices depend on the
    exploration and its own rounds, which is what the calibration of Step 3 uses.
    """
    if len(data["Z"]) != 2:
        raise ValueError("Algorithm 2 needs the records of TwoTrackPolicy (two tracks)")
    A = np.asarray(A, dtype=float)
    single = A.ndim == 1
    As = np.atleast_2d(A)
    p = data["Z_ex"].shape[1]
    if As.shape[1] != p:
        raise ValueError("a loading has length %d, expected p = %d" % (As.shape[1], p))
    data = dict(data)
    if rank == "gic":
        if n0 in ("information", "design"):
            raise ValueError("with rank = 'gic', n0 must be 'equal' or an integer (the other rules need the rank)")
        data["p_T"] = None
    else:
        data["p_T"] = tangent_dimension(data["d1"], data["d2"], int(rank), data["p0"])
    n0 = choose_n0(data, n0, kappa_D)
    folds = [fold_fit(data, j, n0, rank, C_ref) for j in (0, 1)]
    mus, ses, lam_cal = np.zeros((2, len(As))), np.zeros((2, len(As))), [0.0, 0.0]
    for j in (0, 1):
        mus[j], ses[j], lam_cal[j] = fold_calibrate(data, j, folds[j], As, C_cal)
    estimate = 0.5 * (mus[0] + mus[1])
    se = 0.5 * np.sqrt(ses[0] ** 2 + ses[1] ** 2)
    z = norm.ppf(0.5 + level / 2.0)
    out = dict(estimate=estimate, se=se, lower=estimate - z * se, upper=estimate + z * se, level=level, n0=n0,
               rank=[f["rank"] for f in folds], kappa=[f["kappa"] for f in folds],
               lambda_ref=[f["lam_ref"] for f in folds], lambda_cal=lam_cal,
               fold_estimate=mus, fold_se=ses, theta=[f["theta"] for f in folds])
    if single:
        for key in ("estimate", "se", "lower", "upper"):
            out[key] = float(out[key][0])
        out["fold_estimate"], out["fold_se"] = mus[:, 0], ses[:, 0]
    return out
