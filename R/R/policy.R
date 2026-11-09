# Data collection: Algorithm 1 (greedy_policy) and Step 1 of Algorithm 2 (two_track_policy).
#
# Both policies explore for tau rounds (the arm drawn uniformly), fit the pilot estimate on the exploration rows S_o,
# and then choose greedily: the arm with the largest fitted linear predictor <z_{t,a}, theta_hat>.  greedy_policy has
# one greedy track.  two_track_policy has two tracks that take the rounds in turn; each track refits on S_o and its own
# rounds only, so the two records are independent given S_o (Step 1 of Algorithm 2).
#
# Use, one round at a time:
#
#     a <- policy$choose(arms, context)      # arms: K x d1 features of the candidates, context: length d2
#     policy$observe(y)                      # the reward of the chosen arm
#
# Before a track chooses in its round k > 0 it refits its estimator on S_o and its first k rounds at
# lambda_{tau + k}, starting from its previous estimate; the penalty constant is fixed (c_lam a number) or chosen by
# cross-validation (c_lam = "cv").  Arms are numbered 1, ..., K.

BINOMIAL_RADIUS <- 10   # the default domain C_R under the binomial family: every main-effect group and Theta in a ball
                        # of radius 10 (log-odds units), so that the fits stay finite when the exploration rows are
                        # separable; pass radius to change it

collector <- function(n_tracks, d1, d2, horizon, family = "gaussian", trials = 1, main_effects = FALSE,
                      n_covariates = 0, tau = NULL, c_tau = C_TAU, c_lam = "cv", radius = NULL, explore_arms = NULL,
                      seed = NULL) {
  check_family(family, trials)
  if (!is.null(seed)) set.seed(seed)
  sizes <- main_groups(d1, d2, n_covariates, main_effects)
  p0 <- sum(sizes)
  p <- p0 + d1 * d2
  if (is.null(radius) && family == "binomial") radius <- rep(BINOMIAL_RADIUS, length(sizes) + 1)
  tau <- if (is.null(tau)) tau_rule(horizon, d1, d2, c_tau, p0) else as.integer(tau)
  n_tr <- (horizon - tau) %/% n_tracks                          # rounds per track
  if (tau < 2 || n_tr < 4) stop("the horizon is too short for tau = ", tau, " exploration rounds")
  horizon <- tau + n_tracks * n_tr                              # the rounds used (the n of the penalty level)
  cv <- is.character(c_lam) && c_lam == "cv"
  checkpoints <- if (cv) cv_checkpoints(tau, tau + n_tr) else integer(0)

  t <- 0                                                         # rounds observed
  pending <- NULL                                                # the open round: design rows, arm, track
  Z_ex <- matrix(0, tau, p)
  y_ex <- numeric(tau)
  arm_ex <- integer(tau)
  G_ex <- matrix(0, p, p)
  b_ex <- numeric(p)
  score <- NULL
  pilot <- NULL
  base <- NULL
  # the tracks, filled in when the exploration ends:
  #   Zs[[j]], ys[[j]]   the rows S_o followed by track j's own rounds (kept apart from `tracks`, so that observe()
  #            writes the new row in place: a matrix inside a copied list would be copied every round)
  #   k        the track's rounds so far;  arm  its arms
  #   theta    the track's current estimate, fitted on s_fit rows
  #   G, b     sum z z' and sum y z over the track's rows (the Gaussian fit needs only these)
  #   c, cv    the penalty constant in force and the cross-validation choices (rows s, c)
  #   L, H, LH, s_H   solver caches: the top eigenvalue of G / s (every round up to k = 200, then every tenth), the
  #            curvature-weighted Gram matrix (binomial; rebuilt exactly when s has grown by a quarter since s_H,
  #            updated row by row in between) and its top eigenvalue
  new_track <- function() list(k = 0, arm = integer(n_tr), theta = NULL, s_fit = 0, G = NULL, b = NULL, c = NULL,
                               cv = matrix(numeric(0), 0, 2), L = NULL, H = NULL, LH = NULL, s_H = 0)
  tracks <- lapply(seq_len(n_tracks), function(j) new_track())
  Zs <- NULL
  ys <- NULL

  fit_pilot <- function() {
    score <<- pilot_score(Z_ex, y_ex, d1, d2, family, trials, radius, sizes)
    base <<- function(s) penalty_level(1, score, tau, horizon, d1, d2, s)
    c0 <- if (cv) CV_START else c_lam
    lam <- c0 * base(tau)
    if (family == "gaussian") {
      pilot <<- fista(G_ex / tau, b_ex / tau, lam, d1, d2, rep(0, p), radius = radius, sizes = sizes)
    } else {
      pilot <<- proximal_newton(Z_ex, y_ex, lam, d1, d2, rep(0, p), trials, radius, sizes = sizes)
    }
    for (j in seq_len(n_tracks)) {
      tracks[[j]]$theta <<- pilot
      tracks[[j]]$s_fit <<- tau
      tracks[[j]]$c <<- c0
      tracks[[j]]$G <<- G_ex
      tracks[[j]]$b <<- b_ex
    }
    Zs <<- lapply(seq_len(n_tracks), function(j) rbind(Z_ex, matrix(0, n_tr, p)))
    ys <<- lapply(seq_len(n_tracks), function(j) c(y_ex, numeric(n_tr)))
  }

  refit <- function(j, s) {                                      # track j's estimate on its s = tau + k rows
    tr <- tracks[[j]]
    if (s == tr$s_fit) return(invisible(NULL))                 # already fitted on these rows
    tr$s_fit <- s
    k <- s - tau
    rows <- seq_len(s)
    if (cv && s %in% checkpoints) {                              # choose c again on the track's rows
      tr$c <- cv_constant(Zs[[j]][rows, , drop = FALSE], ys[[j]][rows], base, tr$theta, d1, d2, family, trials,
                          radius, sizes)
      tr$cv <- rbind(tr$cv, c(s, tr$c))
    }
    lam <- tr$c * base(s)
    if (k <= 200 || k %% 10 == 0 || is.null(tr$L)) tr$L <- top_eigenvalue(tr$G / s)
    if (family == "gaussian") {
      tr$theta <- fista(tr$G / s, tr$b / s, lam, d1, d2, tr$theta, L = tr$L, radius = radius, sizes = sizes)
    } else {
      Z <- Zs[[j]][rows, , drop = FALSE]
      if (s >= 1.25 * tr$s_H) {                                # exact curvature-weighted Gram, now and then
        w <- variance_function(as.vector(Z %*% tr$theta), family, trials)
        tr$H <- crossprod(Z, Z * w)
        tr$s_H <- s
        tr$LH <- top_eigenvalue(tr$H / s)
      }
      tr$theta <- proximal_newton(Z, ys[[j]][rows], lam, d1, d2, tr$theta, trials, radius, H = tr$H / s, LH = tr$LH,
                                  L_max = 0.25 * trials * tr$L, sizes = sizes)
    }
    tracks[[j]] <<- tr
  }

  choose <- function(arms, context, covariates = NULL) {
    if (!is.null(pending)) stop("observe() the reward of the previous round first")
    if (t >= horizon) stop("the horizon has been reached")
    Zc <- design_rows(arms, context, covariates, main_effects)
    if (ncol(Zc) != p) stop("the design rows have ", ncol(Zc), " columns, expected ", p)
    if (t < tau) {                                               # exploration: an arm drawn uniformly
      a <- if (is.null(explore_arms)) sample.int(nrow(Zc), 1) else as.integer(explore_arms[t + 1])
      pending <<- list(Zc = Zc, a = a, j = 0)
      return(a)
    }
    j <- (t - tau) %% n_tracks + 1
    k <- (t - tau) %/% n_tracks
    if (k > 0) refit(j, tau + k)
    a <- which.max(as.vector(Zc %*% tracks[[j]]$theta))         # the greedy choice
    pending <<- list(Zc = Zc, a = a, j = j)
    a
  }

  observe <- function(y) {
    if (is.null(pending)) stop("choose() an arm first")
    z <- pending$Zc[pending$a, ]
    j <- pending$j
    a <- pending$a
    pending <<- NULL
    if (j == 0) {                                                # an exploration round
      Z_ex[t + 1, ] <<- z
      y_ex[t + 1] <<- y
      arm_ex[t + 1] <<- a
      G_ex <<- G_ex + tcrossprod(z)
      b_ex <<- b_ex + y * z
      t <<- t + 1
      if (t == tau) fit_pilot()
      return(invisible(NULL))
    }
    tr <- tracks[[j]]
    tr$k <- tr$k + 1
    Zs[[j]][tau + tr$k, ] <<- z                                   # in place
    ys[[j]][tau + tr$k] <<- y
    tr$arm[tr$k] <- a
    tr$G <- tr$G + tcrossprod(z)
    tr$b <- tr$b + y * z
    if (family != "gaussian" && !is.null(tr$H))                  # running curvature-weighted Gram
      tr$H <- tr$H + variance_function(sum(z * tr$theta), family, trials) * tcrossprod(z)
    tracks[[j]] <<- tr
    t <<- t + 1
    invisible(NULL)
  }

  data <- function() {
    if (t < horizon) stop("the collection is not complete: ", t, " of ", horizon, " rounds observed")
    list(d1 = d1, d2 = d2, p0 = p0, sizes = sizes, family = family, trials = trials, radius = radius, tau = tau,
         horizon = horizon, n_tr = n_tr, score = score, c_lam = c_lam, pilot = pilot,
         Z_ex = Z_ex, y_ex = y_ex, arm_ex = arm_ex,
         Z = lapply(Zs, function(Z) Z[tau + seq_len(n_tr), , drop = FALSE]),
         y = lapply(ys, function(y) y[tau + seq_len(n_tr)]),
         arm = lapply(tracks, function(tr) tr$arm),
         cv = if (cv) lapply(tracks, function(tr) tr$cv) else NULL)
  }

  estimate <- function() {                                       # greedy_policy: the fit on all rounds so far
    k <- t - tau
    if (k < 0) stop("the exploration has not finished")
    if (k > 0 && is.null(pending)) refit(1, t)
    tracks[[1]]$theta
  }

  list(choose = choose, observe = observe, data = data, estimate = estimate, tau = tau, horizon = horizon,
       n_tr = n_tr, p0 = p0)
}

#' Algorithm 1: the greedy policy
#'
#' A short uniform exploration, then greedy choices from the nuclear-norm-penalized fit on all rounds so far.
#' In every round, call \code{a <- policy$choose(arms, context)} with the candidates' features and the context, show
#' arm \code{a}, and pass its reward to \code{policy$observe(y)}. \code{policy$estimate()} returns the current
#' estimate.
#'
#' @param d1,d2 Dimensions of the arm features m and of the context x.
#' @param horizon The number of rounds n.
#' @param family The reward model: \code{"gaussian"}, or \code{"binomial"} with \code{trials} trials
#'   (\code{trials = 1} for clicks).
#' @param trials The number of trials of the binomial family.
#' @param main_effects Include the main effects (m, x) in the model. Under the binomial family the exploration must
#'   then be long enough for the main-effects-only logistic fit to exist: about twice as many rounds as main-effect
#'   coefficients.
#' @param n_covariates The number q of further unpenalized covariates per arm, passed to \code{choose} as
#'   \code{covariates}.
#' @param tau The exploration length. Default: ceil(c_tau (d1 + d2 + p0) log(n d1 d2)).
#' @param c_tau The constant of the default exploration length.
#' @param c_lam The penalty constant: a number, or \code{"cv"} for 10-fold cross-validation at s = max(2 tau, 200),
#'   2s, 4s, ...
#' @param radius The domain C_R: radii c(R_1, ..., R_G, R_Theta) of the main-effect groups and of Theta. Default: none
#'   under the Gaussian family, 10 for every group under the binomial family.
#' @param explore_arms The arms of the exploration rounds (1, ..., K). Default: drawn uniformly.
#' @param seed If given, \code{set.seed(seed)} is called when the policy is created.
#' @return A policy: a list of the functions \code{choose}, \code{observe} and \code{estimate}, with \code{tau}
#'   and \code{horizon}.
#' @seealso \code{\link{two_track_policy}} for the data collection of Algorithm 2
#' @examples
#' set.seed(1)
#' Theta <- low_rank_truth(6, 6, rank = 1)
#' s <- simulate_stream(1000, 6, 6, K = 8, Theta = Theta)
#' policy <- greedy_policy(6, 6, horizon = 1000)
#' regret <- run(policy, s)
#' sum(regret)
#' theta_hat <- policy$estimate()
#' @export
greedy_policy <- function(d1, d2, horizon, family = "gaussian", trials = 1, main_effects = FALSE, n_covariates = 0,
                          tau = NULL, c_tau = 1 / 20, c_lam = "cv", radius = NULL, explore_arms = NULL, seed = NULL) {
  collector(1, d1, d2, horizon, family, trials, main_effects, n_covariates, tau, c_tau, c_lam, radius, explore_arms,
            seed)
}

#' Step 1 of Algorithm 2: data collection for inference
#'
#' A shared uniform exploration S_o, then two greedy tracks that take the rounds in turn; each track refits on S_o
#' and its own rounds only. After the last round, \code{policy$data()} returns the records for
#' \code{\link{two_track_inference}}. The policy uses tau + 2 floor((n - tau) / 2) rounds, one fewer than n when
#' n - tau is odd (\code{policy$horizon}).
#'
#' @inheritParams greedy_policy
#' @return A policy: a list of the functions \code{choose}, \code{observe} and \code{data}, with \code{tau} and
#'   \code{horizon}.
#' @seealso \code{\link{two_track_inference}}, which takes \code{policy$data()}
#' @examples
#' set.seed(1)
#' Theta <- low_rank_truth(6, 6, rank = 1)
#' s <- simulate_stream(1000, 6, 6, K = 8, Theta = Theta)
#' policy <- two_track_policy(6, 6, horizon = 1000)
#' regret <- run(policy, s)
#' data <- policy$data()
#' c(tau = data$tau, rounds_per_track = data$n_tr)
#' @export
two_track_policy <- function(d1, d2, horizon, family = "gaussian", trials = 1, main_effects = FALSE,
                             n_covariates = 0, tau = NULL, c_tau = 1 / 20, c_lam = "cv", radius = NULL,
                             explore_arms = NULL, seed = NULL) {
  collector(2, d1, d2, horizon, family, trials, main_effects, n_covariates, tau, c_tau, c_lam, radius, explore_arms,
            seed)
}
