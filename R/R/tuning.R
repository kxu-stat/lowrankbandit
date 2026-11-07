# The tuning rules of Algorithm 1: the rates of the theory, with constants.
#
#   exploration length   tau = ceil(c_tau (d1 + d2 + p0) log(n d1 d2))
#   penalty level        lambda_s = c sigma_hat sqrt((d1 + d2) log(n d1 d2) / s),   sigma_hat = score sqrt(tau / (d1 + d2)),
#                        score = ||interaction block of the score on S_o||_op (at the main-effects-only fit)
#   penalty constant     c fixed, or chosen by 10-fold cross-validation on the rows so far, first at
#                        s = max(2 tau, 200) and again whenever s doubles (1/8 before the first choice)

C_TAU <- 1 / 20
CV_GRID <- c(1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)   # largest first: each fit starts at the previous one
CV_START <- 1 / 8
CV_FOLDS <- 10
CV_MIN <- 200

# Exploration length tau = ceil(c_tau (d1 + d2 + p0) log(n d1 d2)), n the horizon (all rounds).
tau_rule <- function(n, d1, d2, c_tau = C_TAU, p0 = 0) as.integer(ceiling(c_tau * (d1 + d2 + p0) * log(n * d1 * d2)))

# The fit with Theta = 0: the main-effect coefficients alone, on their balls, by projected gradient.
main_only_fit <- function(C, y, family, trials, radii = NULL, sizes = NULL, tol = 1e-10, maxit = 200000) {
  s <- length(y)
  L <- max((if (family == "gaussian") 1 else 0.25 * trials) * top_eigenvalue(crossprod(C) / s), 1e-12)
  gamma <- rep(0, ncol(C))
  for (i in seq_len(maxit)) {
    new <- gamma - as.vector(crossprod(C, mean_function(as.vector(C %*% gamma), family, trials) - y)) / s / L
    if (!is.null(radii)) {
      start <- 0
      for (g in seq_along(sizes)) {
        idx <- start + seq_len(sizes[g])
        if (!is.na(radii[g]) && norm2(new[idx]) > radii[g]) new[idx] <- new[idx] * (radii[g] / norm2(new[idx]))
        start <- start + sizes[g]
      }
    }
    if (norm2(new - gamma) <= tol * max(1, norm2(new))) return(new)
    gamma <- new
  }
  stop("the main-effects-only fit on the exploration rows did not converge: under the binomial family ",
       "the rows are probably separable; use a longer exploration (tau or c_tau)")
}

# ||G_tau||_op: the operator norm of the interaction block of the score on the exploration rows, at Theta = 0 and the
# main-effects-only fit (at theta = 0 when there are no main effects).
pilot_score <- function(Z, y, d1, d2, family, trials, radius = NULL, sizes = NULL) {
  s <- length(y)
  p0 <- ncol(Z) - d1 * d2
  if (p0 > 0) {
    C <- Z[, 1:p0, drop = FALSE]
    gamma <- main_only_fit(C, y, family, trials, if (is.null(radius)) NULL else radius[-length(radius)], sizes)
    g <- as.vector(crossprod(Z, mean_function(as.vector(C %*% gamma), family, trials) - y)) / s
  } else {
    g <- as.vector(crossprod(Z, mean_function(0, family, trials) - y)) / s
  }
  score <- max(svd(matrix(g[(p0 + 1):length(g)], d1, d2), nu = 0, nv = 0)$d)
  if (!is.finite(score)) stop("the pilot score is not finite")
  score
}

# lambda_s = c sigma_hat sqrt((d1 + d2) log(n d1 d2) / s), sigma_hat = score sqrt(tau / (d1 + d2)).
penalty_level <- function(c, score, tau, n, d1, d2, s) {
  sigma_hat <- score * sqrt(tau / (d1 + d2))
  c * sigma_hat * sqrt((d1 + d2) * log(n * d1 * d2) / max(s, 1))
}

# The sample sizes at which c is chosen: s_1 = max(2 tau, 200), 2 s_1, 4 s_1, ... below n_end.
cv_checkpoints <- function(tau, n_end) {
  out <- integer(0)
  s <- max(2 * tau, CV_MIN)
  while (s < n_end) {
    out <- c(out, s)
    s <- 2 * s
  }
  out
}

# The penalty constant by 10-fold cross-validation on the rows (Z, y): row i (counted from 0) is in fold i mod 10.  Each
# c of CV_GRID (largest first) is fitted on the other folds at lambda = c base(s') and scored on the fold by squared
# error (Gaussian) or deviance (binomial); the smallest total loss wins.  base(s) is the penalty level at c = 1.
cv_constant <- function(Z, y, base, start, d1, d2, family, trials, radius = NULL, sizes = NULL) {
  fold <- (seq_along(y) - 1) %% CV_FOLDS
  loss <- rep(0, length(CV_GRID))
  for (f in 0:(CV_FOLDS - 1)) {
    train <- fold != f
    test <- fold == f
    Zt <- Z[train, , drop = FALSE]
    yt <- y[train]
    s <- length(yt)
    x <- start
    for (k in seq_along(CV_GRID)) {
      c <- CV_GRID[k]
      if (family == "gaussian") {
        x <- fista(crossprod(Zt) / s, as.vector(crossprod(Zt, yt)) / s, c * base(s), d1, d2, x, radius = radius,
                   sizes = sizes)
      } else {
        x <- proximal_newton(Zt, yt, c * base(s), d1, d2, x, trials, radius, sizes = sizes)
      }
      u <- as.vector(Z[test, , drop = FALSE] %*% x)
      if (family == "gaussian") {
        loss[k] <- loss[k] + sum((y[test] - u)^2)
      } else {
        loss[k] <- loss[k] + sum(trials * softplus(u) - y[test] * u)
      }
    }
  }
  CV_GRID[which.min(loss)]
}
