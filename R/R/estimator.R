# The nuclear-norm-penalized GLM estimator of Algorithm 1:
#
#     theta_hat in argmin_{theta in C_R}  (1/s) sum_i [psi(z_i' theta) - y_i z_i' theta] + lam ||Theta||_*,
#
# Theta the interaction block of theta; the main effects are not penalized.  C_R is an optional domain:
# each main-effect group in a Euclidean ball and Theta in a Frobenius ball, radius = c(R_1, ..., R_G, R_Theta); with
# radius NULL there is no domain.  Gaussian family: FISTA with adaptive restart.  Binomial family: proximal Newton
# (each step a FISTA solve of the quadratic model, Armijo backtracking).
#
# The solvers are written in C++ (src/estimator.cpp, the same steps as the Python package's estimator.py), as is
# top_eigenvalue(G), the largest eigenvalue of a symmetric matrix; fista() and proximal_newton() below pass them their
# arguments.

#' @useDynLib lowrankbandit, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

norm2 <- function(x) sqrt(sum(x * x))

# The domain C_R goes to C++ as the radii (none: numeric(0)) and the sizes of the main-effect groups (none: integer(0);
# NULL: one group).
group_sizes <- function(p0, sizes) if (p0 == 0) integer(0) else if (is.null(sizes)) p0 else as.integer(sizes)

# min 1/2 x'Gx - b'x + lam ||Theta(x)||_* on C_R by FISTA with adaptive restart, started at x0.
# L: the largest eigenvalue of G (computed when NULL).
fista <- function(G, b, lam, d1, d2, x0, L = NULL, iters = 400, tol = 1e-6, radius = NULL, sizes = NULL) {
  if (is.null(L)) L <- top_eigenvalue(G)
  fista_cpp(G, b, lam, d1, d2, x0, L, iters, tol, as.numeric(radius), group_sizes(length(x0) - d1 * d2, sizes))
}

# min (1/s) sum_i [trials log(1 + e^{z_i'x}) - y_i z_i'x] + lam ||Theta(x)||_* on C_R by proximal Newton.
# H, the curvature-weighted Gram matrix, may be passed in (possibly computed at an earlier estimate); it is rebuilt
# after a backtracked step, when the residual fails to halve, and after eight steps.
proximal_newton <- function(Z, y, lam, d1, d2, x0, trials, radius = NULL, H = NULL, LH = NULL, L_max = NULL,
                            iters = 30, tol = 1e-6, inner_iters = 1000, inner_tol = 1e-8, sizes = NULL) {
  if (is.null(L_max)) L_max <- 0.25 * trials * top_eigenvalue(crossprod(Z) / length(y))
  proximal_newton_cpp(Z, y, lam, d1, d2, x0, trials, as.numeric(radius), group_sizes(length(x0) - d1 * d2, sizes),
                      if (is.null(H)) matrix(0, 0, 0) else H, if (is.null(LH)) NaN else LH, L_max, iters, tol,
                      inner_iters, inner_tol)
}

#' The nuclear-norm-penalized GLM estimator
#'
#' Minimizes (1/s) sum_i [psi(z_i' theta) - y_i z_i' theta] + lam ||Theta||_* over the domain C_R, Theta the
#' interaction block of theta (the last d1 d2 coordinates, column-major); the leading main-effect coordinates are not
#' penalized.
#'
#' @param Z The s x p matrix of design rows z = (c, m, x, vec(m x')), main effects first.
#' @param y The s responses.
#' @param lam The penalty level.
#' @param d1,d2 The dimensions of the interaction Theta (its d1 d2 coordinates are the last of z).
#' @param family The reward model, \code{"gaussian"} or \code{"binomial"}.
#' @param trials The number of trials of the binomial family.
#' @param start The starting point (zero by default).
#' @param radius Radii c(R_1, ..., R_G, R_Theta) of the domain C_R for the main-effect groups and Theta (NA: no
#'   ball); NULL for no domain.
#' @param sizes The sizes of the main-effect groups.
#' @param tol,iters The stopping tolerance and the iteration cap of the solver.
#' @return The estimate: main effects first, then vec(Theta).
#' @export
fit_nuclear <- function(Z, y, lam, d1, d2, family, trials = 1, start = NULL, radius = NULL, sizes = NULL,
                        tol = 1e-6, iters = 400) {
  Z <- as.matrix(Z)
  s <- nrow(Z)
  x0 <- if (is.null(start)) rep(0, ncol(Z)) else start
  if (family == "gaussian")
    return(fista(crossprod(Z) / s, as.vector(crossprod(Z, y)) / s, lam, d1, d2, x0, iters = iters, tol = tol,
                 radius = radius, sizes = sizes))
  proximal_newton(Z, y, lam, d1, d2, x0, trials, radius, tol = tol, sizes = sizes)
}
