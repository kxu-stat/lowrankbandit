# The model: the exponential family, the design rows, and the geometry of rank-r matrices.
#
# Round t shows K candidate arms with features m_{t,a} (length d1), one context x_t (length d2) and, optionally,
# further covariates c_{t,a} (length q).  The design row of arm a is
#
#     z_{t,a} = (c_{t,a}, m_{t,a}, x_t, vec(m_{t,a} x_t'))     (main effects first, then the interaction),
#
# vec stacking columns, so <z_{t,a}, theta> = c' gamma + m' theta_m + x' theta_x + m' Theta x.  The leading p0
# coordinates are the unpenalized main effects (p0 = q + d1 + d2 with main effects, q without; the policies leave the
# main effects out unless asked).  The reward follows the canonical GLM with linear predictor u = <z, theta>:
# Gaussian (psi(u) = u^2 / 2) or binomial with `trials` trials (psi(u) = trials log(1 + e^u)).

# ------------------------------------------------------------------ the family
sigmoid <- function(u) 0.5 * (1 + tanh(0.5 * u))

softplus <- function(u) pmax(u, 0) + log1p(exp(-abs(u)))       # log(1 + e^u) without overflow

mean_function <- function(u, family, trials) {
  if (family == "gaussian") u else trials * sigmoid(u)
}

variance_function <- function(u, family, trials) {
  if (family == "gaussian") return(rep(1, length(u)))
  q <- sigmoid(u)
  trials * q * (1 - q)
}

check_family <- function(family, trials) {
  if (!family %in% c("gaussian", "binomial")) stop("family must be 'gaussian' or 'binomial'")
  if (family == "binomial" && (trials != round(trials) || trials < 1)) stop("trials must be a positive integer")
}

# ------------------------------------------------------------------ design rows
#' Column-major vectorization
#'
#' @param M A matrix.
#' @return The columns of \code{M} stacked into one vector.
#' @export
vec <- function(M) as.vector(M)

# The K x p matrix of design rows z_{t,a} = (c_{t,a}, m_{t,a}, x_t, vec(m_{t,a} x_t')) of one round; arms K x d1,
# context of length d2, covariates K x q or NULL.
design_rows <- function(arms, context, covariates = NULL, main_effects = FALSE) {
  arms <- as.matrix(arms)
  context <- as.vector(context)
  K <- nrow(arms)
  blocks <- list()
  if (!is.null(covariates)) blocks <- c(blocks, list(matrix(covariates, nrow = K)))
  if (main_effects) blocks <- c(blocks, list(arms, matrix(context, K, length(context), byrow = TRUE)))
  inter <- kronecker(matrix(context, nrow = 1), arms)         # row a: vec(m_a x') = kronecker(x, m_a)
  do.call(cbind, c(blocks, list(inter)))
}

main_groups <- function(d1, d2, n_covariates, main_effects) {
  sizes <- integer(0)
  if (n_covariates > 0) sizes <- c(sizes, n_covariates)
  if (main_effects) sizes <- c(sizes, d1, d2)
  sizes
}

#' Loading of a target that involves the interaction only
#'
#' The loading A = (0, vec(A_Theta)) of the target <A_Theta, Theta>.
#'
#' @param A_Theta The d1 x d2 matrix of the target <A_Theta, Theta> = sum_ij (A_Theta)_ij Theta_ij.
#' @param p0 The number of main-effect coordinates (\code{data$p0}).
#' @return The loading (0, vec(A_Theta)), of length p0 + d1 d2.
#' @export
interaction_loading <- function(A_Theta, p0) c(rep(0, p0), vec(A_Theta))

# ------------------------------------------------------------------ rank-r geometry
interaction_block <- function(theta, d1, d2) {
  p0 <- length(theta) - d1 * d2
  matrix(theta[(p0 + 1):length(theta)], d1, d2)
}

# Rank-r truncation of the interaction block (main effects kept): list(theta, U, V).
truncate_rank <- function(theta, d1, d2, r) {
  p0 <- length(theta) - d1 * d2
  s <- svd(interaction_block(theta, d1, d2))
  U <- s$u[, seq_len(r), drop = FALSE]
  V <- s$v[, seq_len(r), drop = FALSE]
  out <- theta
  out[(p0 + 1):length(theta)] <- vec(U %*% (s$d[seq_len(r)] * t(V)))
  list(theta = out, U = U, V = V)
}

# Projection onto R^{p0} x T(U, V): the main effects are kept and the interaction block M becomes
# U U'M + M V V' - U U'M V V'.
tangent_project <- function(v, U, V, d1, d2) {
  p0 <- length(v) - d1 * d2
  M <- matrix(v[(p0 + 1):length(v)], d1, d2)
  UtM <- crossprod(U, M)
  out <- v
  out[(p0 + 1):length(v)] <- vec(U %*% UtM + (M %*% V) %*% t(V) - U %*% ((UtM %*% V) %*% t(V)))
  out
}

tangent_project_rows <- function(Z, U, V, d1, d2) {
  t(apply(Z, 1, tangent_project, U = U, V = V, d1 = d1, d2 = d2))
}

# Orthonormal columns spanning R^{p0} x T(U, V); p_T = p0 + r (d1 + d2 - r) columns.
tangent_basis <- function(U, V, d1, d2, p0) {
  r <- ncol(U)
  Uc <- svd(U, nu = d1)$u[, seq_len(d1 - r) + r, drop = FALSE]     # an orthonormal basis of the complement of U
  Vc <- svd(V, nu = d2)$u[, seq_len(d2 - r) + r, drop = FALSE]
  cols <- list()
  for (pair in list(list(U, V), list(U, Vc), list(Uc, V))) {
    L <- pair[[1]]
    R <- pair[[2]]
    for (i in seq_len(ncol(L))) for (j in seq_len(ncol(R))) cols[[length(cols) + 1]] <- vec(outer(L[, i], R[, j]))
  }
  Qi <- do.call(cbind, cols)
  Q <- matrix(0, p0 + d1 * d2, p0 + ncol(Qi))
  if (p0 > 0) Q[1:p0, 1:p0] <- diag(p0)
  Q[(p0 + 1):(p0 + d1 * d2), (p0 + 1):(p0 + ncol(Qi))] <- Qi
  Q
}

tangent_dimension <- function(d1, d2, r, p0) p0 + r * (d1 + d2 - r)
