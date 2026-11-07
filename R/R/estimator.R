# The nuclear-norm-penalized GLM estimator of Algorithm 1:
#
#     theta_hat in argmin_{theta in C_R}  (1/s) sum_i [psi(z_i' theta) - y_i z_i' theta] + lam ||Theta||_*,
#
# Theta the interaction block of theta; the main effects are not penalized.  C_R is an optional domain:
# each main-effect group in a Euclidean ball and Theta in a Frobenius ball, radius = c(R_1, ..., R_G, R_Theta); with
# radius NULL there is no domain.  Gaussian family: FISTA with adaptive restart.  Binomial family: proximal Newton
# (each step a FISTA solve of the quadratic model, Armijo backtracking).

norm2 <- function(x) sqrt(sum(x * x))

top_eigenvalue <- function(G) eigen(G, symmetric = TRUE, only.values = TRUE)$values[1]

nuclear_norm <- function(theta, d1, d2) sum(svd(interaction_block(theta, d1, d2), nu = 0, nv = 0)$d)

# Proximal map of step * lam ||Theta||_* on C_R: the singular values of the interaction block are soft-thresholded at
# step * lam and scaled into the Frobenius ball; each main-effect group is projected onto its ball.
prox <- function(u, step, lam, d1, d2, radius = NULL, sizes = NULL) {
  p0 <- length(u) - d1 * d2
  s <- svd(interaction_block(u, d1, d2))
  sv <- pmax(s$d - step * lam, 0)
  RT <- if (is.null(radius)) NA else radius[length(radius)]
  if (!is.na(RT) && norm2(sv) > RT) sv <- sv * (RT / norm2(sv))
  inter <- vec(s$u %*% (sv * t(s$v)))
  if (p0 == 0) return(inter)
  if (is.null(sizes)) sizes <- p0                                   # one main-effect group
  blocks <- numeric(0)
  start <- 0
  for (g in seq_along(sizes)) {
    block <- u[start + seq_len(sizes[g])]
    R <- if (is.null(radius)) NA else radius[g]
    if (!is.na(R) && norm2(block) > R) block <- block * (R / norm2(block))
    blocks <- c(blocks, block)
    start <- start + sizes[g]
  }
  c(blocks, inter)
}

# min 1/2 x'Gx - b'x + lam ||Theta(x)||_* on C_R by FISTA with adaptive restart, started at x0.
# L: the largest eigenvalue of G (computed when NULL).
fista <- function(G, b, lam, d1, d2, x0, L = NULL, iters = 400, tol = 1e-6, radius = NULL, sizes = NULL) {
  if (is.null(L)) L <- top_eigenvalue(G)
  L <- max(L, 1e-12)
  x <- x0
  y <- x
  tk <- 1
  for (i in seq_len(iters)) {
    xn <- prox(y - as.vector(G %*% y - b) / L, 1 / L, lam, d1, d2, radius, sizes)
    if (norm2(L * (y - xn)) <= tol * max(1, L * norm2(xn))) return(xn)
    if (sum((y - xn) * (xn - x)) > 0) tk <- 1                         # adaptive restart
    tn <- (1 + sqrt(1 + 4 * tk * tk)) / 2
    y <- xn + ((tk - 1) / tn) * (xn - x)
    x <- xn
    tk <- tn
  }
  x
}

# min (1/s) sum_i [trials log(1 + e^{z_i'x}) - y_i z_i'x] + lam ||Theta(x)||_* on C_R by proximal Newton.
# H, the curvature-weighted Gram matrix, may be passed in (possibly computed at an earlier estimate); it is rebuilt
# after a backtracked step, when the residual fails to halve, and after eight steps.
proximal_newton <- function(Z, y, lam, d1, d2, x0, trials, radius = NULL, H = NULL, LH = NULL, L_max = NULL,
                            iters = 30, tol = 1e-6, inner_iters = 1000, inner_tol = 1e-8, sizes = NULL) {
  s <- length(y)
  if (is.null(L_max)) L_max <- 0.25 * trials * top_eigenvalue(crossprod(Z) / s)
  L_max <- max(L_max, 1e-12)
  objective <- function(v, u) mean(trials * softplus(u) - y * u) + lam * nuclear_norm(v, d1, d2)
  x <- x0
  u <- as.vector(Z %*% x)
  refresh <- is.null(H)
  res_old <- Inf
  for (it in seq_len(iters)) {
    g <- as.vector(crossprod(Z, trials * sigmoid(u) - y)) / s
    xp <- prox(x - g / L_max, 1 / L_max, lam, d1, d2, radius, sizes)
    res <- norm2(L_max * (x - xp)) / max(1, L_max * norm2(xp))
    if (res <= tol) return(xp)
    refresh <- refresh || res > 0.5 * res_old || it > 8
    res_old <- res
    if (refresh) {
      w <- variance_function(u, "binomial", trials)
      H <- crossprod(Z, Z * w) / s
      LH <- top_eigenvalue(H)
    }
    if (is.null(LH)) LH <- top_eigenvalue(H)
    v <- fista(H, as.vector(H %*% x) - g, lam, d1, d2, x, L = LH, iters = inner_iters, tol = max(inner_tol, 0.1 * res),
               radius = radius, sizes = sizes)
    step <- v - x
    fx <- objective(x, u)
    model <- sum(g * step) + lam * (nuclear_norm(v, d1, d2) - nuclear_norm(x, d1, d2))
    alpha <- 1
    xn <- v
    un <- as.vector(Z %*% v)
    while (objective(xn, un) > fx + 1e-4 * alpha * model && alpha > 1e-3) {
      alpha <- alpha * 0.5
      xn <- x + alpha * step
      un <- as.vector(Z %*% xn)
    }
    x <- xn
    u <- un
    refresh <- alpha < 1
  }
  g <- as.vector(crossprod(Z, trials * sigmoid(u) - y)) / s
  prox(x - g / L_max, 1 / L_max, lam, d1, d2, radius, sizes)
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
