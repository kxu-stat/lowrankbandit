# Algorithm 2: two-track cross-fitted inference for a linear functional mu* = <A, theta*>.
#
# The data are the records of two_track_policy: the shared exploration S_o and the two tracks of n_tr rounds each.
# For fold j = 1, 2 (track j' = the other track):
#
#   Step 2  fit on S_o and the first n0 rounds of track j, truncate the interaction to rank r0: theta_breve, its
#           tangent space and orthonormal basis Q.  Refit on the remaining n_ref = n_tr - n0 rounds of track j by the
#           predictable recursion with ridge lambda_ref = n_ref kappa / (2 C_ref log n_ref), and truncate again:
#           theta_hat^(j) and its projector P_j.
#   Step 3  over all of track j', the calibration recursion with
#           lambda_cal = n_tr kappa / (2 C_cal log n_tr) gives the weights omega_t, and
#               mu^(j) = <A, theta_hat^(j)> + sum_t {y_t - psi'(<z_t, theta_hat^(j)>)} omega_t,
#               (V^(j))^2 = sum_t {y_t - psi'(<z_t, theta_hat^(j)>)}^2 omega_t^2.
#   Step 4  mu_hat = (mu^(1) + mu^(2)) / 2, V_N = sqrt((V^(1))^2 + (V^(2))^2) / 2, and the interval
#           mu_hat -+ Phi^{-1}(1 - delta/2) V_N at the level 1 - delta.
#
# kappa, the information constant of both recursions, is estimated from S_o: the average eigenvalue of the
# curvature-weighted exploration Gram matrix on the tangent coordinates.

C_REF <- 1 / 15        # the refit's constant: the theorem needs C_ref >= 1/15
C_CAL <- 1             # the calibration's constant: the theorem needs C_cal >= 2/15
C_SPLIT <- 1 / 3       # the constant of the two rules for n0
N0_RULES <- c("equal", "information", "design")

# ------------------------------------------------------------------ the split n0
# The design-stage split: the equal split n_tr / 2, lengthened when half a track is shorter than the burn-in
# C_SPLIT p_T log(n d1 d2) / kappa_D, and at most cap n_tr.  kappa_D is a planning value of the information per round
# and coordinate (for example 1 for the Gaussian family with E[z z'] = I).
split_design <- function(data, kappa_D, cap = 0.75) {
  n_tr <- data$n_tr
  need <- ceiling(C_SPLIT * data$p_T * log(data$horizon * data$d1 * data$d2) / kappa_D)
  as.integer(min(max(need, n_tr %/% 2), floor(cap * n_tr)))
}

# The information rule: n0 = ceil(C_SPLIT p_T log(n d1 d2) / kappa0) within [n_tr / 4, 3 n_tr / 4], kappa0 the
# information per round and coordinate of S_o, tr(G_w) / (tau p), G_w the curvature-weighted exploration Gram matrix
# at the pilot estimate.
split_information <- function(data) {
  n_tr <- data$n_tr
  Z <- data$Z_ex
  w <- variance_function(as.vector(Z %*% data$pilot), data$family, data$trials)
  kappa0 <- sum(w * rowSums(Z^2)) / (data$tau * ncol(Z))
  need <- ceiling(C_SPLIT * data$p_T * log(data$horizon * data$d1 * data$d2) / kappa0)
  as.integer(min(max(need, ceiling(n_tr / 4), 10), floor(3 * n_tr / 4)))
}

choose_n0 <- function(data, n0, kappa_D = NULL) {
  if (is.character(n0)) {
    if (n0 == "equal") return(data$n_tr %/% 2)
    if (n0 == "information") return(split_information(data))
    if (n0 == "design") {
      if (is.null(kappa_D)) stop("n0 = 'design' needs kappa_D, a planning value of the information per round and ",
                                 "coordinate")
      return(split_design(data, kappa_D))
    }
    stop("n0 must be an integer or one of ", paste(N0_RULES, collapse = ", "))
  }
  n0 <- as.integer(n0)
  if (n0 < 2 || n0 > data$n_tr - 2) stop("n0 must lie in [2, n_tr - 2] = [2, ", data$n_tr - 2, "]")
  n0
}

# ------------------------------------------------------------------ the rank
# The rank minimizing L_s(theta_r) + (log s / s) r (d1 + d2 - r), theta_r the rank-r truncation of theta and L_s the
# average negative log-likelihood (Gaussian: 0.5 log of the mean squared residual); at least 1.
gic_rank <- function(Z, y, theta, d1, d2, family, trials) {
  s <- length(y)
  crit <- sapply(0:min(d1, d2), function(r) {
    u <- as.vector(Z %*% truncate_rank(theta, d1, d2, r)$theta)
    loss <- if (family == "gaussian") 0.5 * log(mean((y - u)^2)) else mean(trials * softplus(u) - y * u)
    loss + log(s) / s * r * (d1 + d2 - r)
  })
  max(which.min(crit) - 1, 1)
}

# ------------------------------------------------------------------ Step 2
c_in_force <- function(data, j, s) {                     # the penalty constant track j used at sample size s
  if (!(is.character(data$c_lam) && data$c_lam == "cv")) return(data$c_lam)
  c <- CV_START
  cvj <- data$cv[[j]]
  for (i in seq_len(nrow(cvj))) if (cvj[i, 1] <= s) c <- cvj[i, 2]
  c
}

# Step 2 for fold j: the preliminary fit on S_o and the first n0 rounds of track j, its rank truncation, and the
# predictable refit on the rest of track j.
fold_fit <- function(data, j, n0, rank, C_ref = C_REF) {
  d1 <- data$d1; d2 <- data$d2; p0 <- data$p0
  family <- data$family; trials <- data$trials; radius <- data$radius; sizes <- data$sizes
  tau <- data$tau; n_tr <- data$n_tr
  Z <- data$Z[[j]]; y <- data$y[[j]]
  s <- tau + n0
  lam <- penalty_level(c_in_force(data, j, s), data$score, tau, data$horizon, d1, d2, s)
  Zs <- rbind(data$Z_ex, Z[seq_len(n0), , drop = FALSE])
  ys <- c(data$y_ex, y[seq_len(n0)])
  if (family == "gaussian") {
    breve <- fista(crossprod(Zs) / s, as.vector(crossprod(Zs, ys)) / s, lam, d1, d2, data$pilot, iters = 600,
                   tol = 1e-8, radius = radius, sizes = sizes)
  } else {
    breve <- proximal_newton(Zs, ys, lam, d1, d2, data$pilot, trials, radius, tol = 1e-8, sizes = sizes)
  }
  r <- if (identical(rank, "gic")) gic_rank(Zs, ys, breve, d1, d2, family, trials) else as.integer(rank)
  tr <- truncate_rank(breve, d1, d2, r)
  breve <- tr$theta
  Q <- tangent_basis(tr$U, tr$V, d1, d2, p0)
  # kappa: the average eigenvalue of the curvature-weighted exploration Gram on the tangent coordinates
  w_ex <- variance_function(as.vector(data$Z_ex %*% breve), family, trials)
  G_w <- crossprod(data$Z_ex, data$Z_ex * w_ex)
  kappa <- sum(diag(crossprod(Q, (G_w / tau) %*% Q))) / ncol(Q)
  # the predictable refit on rounds n0 + 1, ..., n_tr
  n_ref <- n_tr - n0
  lam_ref <- n_ref / (2 * C_ref * log(max(n_ref, 3))) * kappa
  R <- diag(ncol(Q))
  acc <- rep(0, ncol(Q))
  for (t in (n0 + 1):n_tr) {
    q <- as.vector(crossprod(Q, Z[t, ]))
    u <- sum(Z[t, ] * breve)
    c <- variance_function(u, family, trials)
    w <- as.vector(crossprod(R, q)) / (lam_ref + c * sum(q * q))
    R <- R - c * outer(q, w)
    acc <- acc + w * (y[t] - mean_function(u, family, trials))
  }
  out <- truncate_rank(breve + as.vector(Q %*% acc), d1, d2, r)
  list(theta = out$theta, U = out$U, V = out$V, kappa = kappa, lam_ref = lam_ref, rank = r)
}

# ------------------------------------------------------------------ Step 3
# Step 3 for fold j over the other track: the calibration recursion for every loading (rows of As) at once.
fold_calibrate <- function(data, j, fold, As, C_cal = C_CAL) {
  d1 <- data$d1; d2 <- data$d2; family <- data$family; trials <- data$trials; n_tr <- data$n_tr
  Z <- data$Z[[3 - j]]; y <- data$y[[3 - j]]
  PZ <- tangent_project_rows(Z, fold$U, fold$V, d1, d2)
  qq <- rowSums(PZ * PZ)
  rho <- t(tangent_project_rows(As, fold$U, fold$V, d1, d2))      # p x L: the unmatched loadings
  lam_cal <- n_tr / (2 * C_cal * log(max(n_tr, 3))) * fold$kappa
  u <- as.vector(Z %*% fold$theta)
  c <- variance_function(u, family, trials)
  omega <- matrix(0, n_tr, nrow(As))
  for (t in seq_len(n_tr)) {                                       # weights fixed before y_t is used
    w <- as.vector(crossprod(Z[t, ], rho)) / (lam_cal + c[t] * qq[t])
    rho <- rho - c[t] * outer(PZ[t, ], w)
    omega[t, ] <- w
  }
  res <- y - mean_function(u, family, trials)
  mu <- as.vector(As %*% fold$theta) + as.vector(crossprod(res, omega))
  V2 <- as.vector(crossprod(res^2, omega^2))
  list(mu = mu, se = sqrt(V2), lam_cal = lam_cal)
}

# ------------------------------------------------------------------ Algorithm 2
#' Algorithm 2: confidence intervals after greedy data collection
#'
#' Steps 2-4 of Algorithm 2: estimates, standard errors and confidence intervals for linear functionals
#' mu* = <A, theta*> of the parameter, from the records of \code{\link{two_track_policy}}.
#'
#' The intervals are valid for data collected by \code{two_track_policy} only: each track's choices depend on the
#' exploration and its own rounds, which is what the calibration of Step 3 uses.
#'
#' @param data The records of a two-track policy after its last round, \code{policy$data()}.
#' @param A The loading of the target: a vector of length p, main effects first, then vec(A_Theta) (see
#'   \code{\link{interaction_loading}}); or a matrix whose rows are several loadings, which share Step 2.
#' @param rank The rank r0 of the interaction, or \code{"gic"} to choose it in each fold by an information criterion.
#' @param n0 The rounds of each track used for the preliminary fit; the remaining rounds go to the refit.
#'   \code{"equal"}: n_tr / 2. \code{"information"}: from the information per round estimated on the exploration.
#'   \code{"design"}: the equal split, lengthened when half a track is short for the planning value \code{kappa_D}.
#'   Or a number of rounds.
#' @param kappa_D The planning value of the information per round and coordinate, for \code{n0 = "design"}.
#' @param level The confidence level.
#' @param C_ref,C_cal The constants of the refit and calibration ridges (the theory needs C_ref >= 1/15 and
#'   C_cal >= 2/15).
#' @return A list with \code{estimate}, \code{se}, \code{lower} and \code{upper} (one entry per loading), and
#'   \code{n0}; for each fold the rank, \code{kappa}, \code{lambda_ref}, \code{lambda_cal}, the fold estimates
#'   \code{fold_estimate} and standard errors \code{fold_se}, and the refitted parameter \code{theta}.
#' @examples
#' set.seed(1)
#' d <- 4
#' Theta <- low_rank_truth(d, d, rank = 1)
#' s <- simulate_stream(600, d, d, K = 5, Theta = Theta, noise_sd = 0.3)
#' policy <- two_track_policy(d, d, horizon = 600)
#' regret <- run(policy, s)
#' A <- interaction_loading(diag(d)[, 1] %o% diag(d)[1, ], p0 = 0)     # the entry Theta[1, 1]
#' res <- two_track_inference(policy$data(), A, rank = 1)
#' c(estimate = res$estimate, lower = res$lower, upper = res$upper, truth = Theta[1, 1])
#' @export
two_track_inference <- function(data, A, rank, n0 = "equal", kappa_D = NULL, level = 0.95, C_ref = 1 / 15,
                                C_cal = 1) {
  if (length(data$Z) != 2) stop("Algorithm 2 needs the records of two_track_policy (two tracks)")
  single <- is.null(dim(A))
  As <- if (single) matrix(A, nrow = 1) else as.matrix(A)
  p <- ncol(data$Z_ex)
  if (ncol(As) != p) stop("a loading has length ", ncol(As), ", expected p = ", p)
  if (identical(rank, "gic")) {
    if (identical(n0, "information") || identical(n0, "design"))
      stop("with rank = 'gic', n0 must be 'equal' or an integer (the other rules need the rank)")
    data$p_T <- NA
  } else {
    data$p_T <- tangent_dimension(data$d1, data$d2, as.integer(rank), data$p0)
  }
  n0 <- choose_n0(data, n0, kappa_D)
  folds <- lapply(1:2, function(j) fold_fit(data, j, n0, rank, C_ref))
  cal <- lapply(1:2, function(j) fold_calibrate(data, j, folds[[j]], As, C_cal))
  mus <- rbind(cal[[1]]$mu, cal[[2]]$mu)
  ses <- rbind(cal[[1]]$se, cal[[2]]$se)
  estimate <- 0.5 * (mus[1, ] + mus[2, ])
  se <- 0.5 * sqrt(ses[1, ]^2 + ses[2, ]^2)
  z <- stats::qnorm(0.5 + level / 2)
  list(estimate = estimate, se = se, lower = estimate - z * se, upper = estimate + z * se, level = level, n0 = n0,
       rank = sapply(folds, function(f) f$rank), kappa = sapply(folds, function(f) f$kappa),
       lambda_ref = sapply(folds, function(f) f$lam_ref), lambda_cal = sapply(cal, function(x) x$lam_cal),
       fold_estimate = if (single) mus[, 1] else mus, fold_se = if (single) ses[, 1] else ses,
       theta = lapply(folds, function(f) f$theta))
}
