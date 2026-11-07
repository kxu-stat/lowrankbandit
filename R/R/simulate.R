# Simulated data for examples and checks.
#
# A stream fixes every round in advance: the candidates' features, the contexts, and the randomness of the rewards
# (Gaussian noise, or uniform numbers for the binomial draws).  With the true parameter, the reward of any arm in any
# round is then determined, so different policies -- or the R and Python versions of this package -- can be run on the
# same rounds.  Rounds and arms are numbered from 1.

#' A fixed sequence of rounds
#'
#' The rounds are fixed in advance, so that several policies can be run on the same data.
#'
#' @param arms The n x K x d1 array of the candidates' features m_{t,a}.
#' @param contexts The n x d2 matrix of the contexts x_t.
#' @param theta The true parameter: main effects first, then vec(Theta*).
#' @param family The reward model, \code{"gaussian"} or \code{"binomial"}.
#' @param trials The number of trials of the binomial family.
#' @param noise Gaussian family: the n noise values, y = <z, theta> + noise_t.
#' @param uniforms Binomial family: n x trials uniform numbers, y = #{k : uniforms_{t,k} < sigmoid(<z, theta>)}.
#' @param covariates The n x K x q array of further covariates, or NULL.
#' @param main_effects Whether theta contains the main effects (m, x), after the covariates' coefficients.
#' @return A stream: a list with the functions \code{rows(t)}, \code{means(t)}, \code{reward(t, a)} and
#'   \code{regret(t, a)}, and the data.
#' @export
stream <- function(arms, contexts, theta, family = "gaussian", trials = 1, noise = NULL, uniforms = NULL,
                   covariates = NULL, main_effects = FALSE) {
  n <- dim(arms)[1]
  K <- dim(arms)[2]
  if (!is.null(uniforms)) uniforms <- matrix(uniforms, nrow = n)
  rows <- function(t) {
    c <- if (is.null(covariates)) NULL else matrix(covariates[t, , ], nrow = K)
    design_rows(matrix(arms[t, , ], nrow = K), contexts[t, ], c, main_effects)
  }
  means <- function(t) mean_function(as.vector(rows(t) %*% theta), family, trials)
  reward <- function(t, a) {
    u <- sum(rows(t)[a, ] * theta)
    if (family == "gaussian") return(u + noise[t])
    sum(uniforms[t, ] < sigmoid(u))
  }
  regret <- function(t, a) {
    m <- means(t)
    max(m) - m[a]
  }
  list(rows = rows, means = means, reward = reward, regret = regret, arms = arms, contexts = contexts,
       covariates = covariates, theta = theta, family = family, trials = trials, n = n, K = K)
}

#' Run a policy on a stream
#'
#' @param policy A policy from \code{\link{greedy_policy}} or \code{\link{two_track_policy}}.
#' @param s A stream from \code{\link{stream}} or \code{\link{simulate_stream}}.
#' @param rounds The number of rounds; default: the policy's horizon.
#' @return The regret of each round: the best mean reward minus the mean reward of the chosen arm.
#' @export
run <- function(policy, s, rounds = NULL) {
  if (is.null(rounds)) rounds <- min(s$n, policy$horizon)
  regret <- numeric(rounds)
  for (t in seq_len(rounds)) {
    c <- if (is.null(s$covariates)) NULL else matrix(s$covariates[t, , ], nrow = s$K)
    a <- policy$choose(matrix(s$arms[t, , ], nrow = s$K), s$contexts[t, ], c)
    policy$observe(s$reward(t, a))
    regret[t] <- s$regret(t, a)
  }
  regret
}

#' A random low-rank matrix
#'
#' Theta* = signal U V' / sqrt(rank), with U (d1 x rank) and V (d2 x rank) random orthonormal, so that
#' ||Theta*||_F = signal and the rank singular values are equal.
#'
#' @param d1,d2 The dimensions.
#' @param rank The rank.
#' @param signal The Frobenius norm.
#' @return A d1 x d2 matrix.
#' @export
low_rank_truth <- function(d1, d2, rank, signal = 1) {
  U <- qr.Q(qr(matrix(stats::rnorm(d1 * rank), d1, rank)))
  V <- qr.Q(qr(matrix(stats::rnorm(d2 * rank), d2, rank)))
  signal * U %*% t(V) / sqrt(rank)
}

sphere <- function(n, d) {
  z <- matrix(stats::rnorm(n * d), n, d)
  sqrt(d) * z / sqrt(rowSums(z^2))
}

#' Simulated rounds
#'
#' A stream of n rounds with features and contexts uniform on the sphere of radius sqrt(d) or standard normal
#' (both of second moment I), the true interaction Theta and, optionally, main effects. Uses R's random number
#' generator.
#'
#' @param n The number of rounds.
#' @param d1,d2 The dimensions of the arm features and of the context.
#' @param K The number of candidates per round.
#' @param Theta The true interaction, d1 x d2.
#' @param family The reward model, \code{"gaussian"} or \code{"binomial"}.
#' @param trials The number of trials of the binomial family.
#' @param noise_sd The standard deviation of the Gaussian noise.
#' @param main The main effects (theta_m, theta_x) of length d1 + d2, or NULL.
#' @param covariate_law \code{"sphere"} or \code{"normal"}.
#' @return A stream (see \code{\link{stream}}).
#' @export
simulate_stream <- function(n, d1, d2, K, Theta, family = "gaussian", trials = 1, noise_sd = 0.2, main = NULL,
                            covariate_law = "sphere") {
  if (covariate_law == "sphere") {
    arms <- array(sphere(n * K, d1), c(n * K, d1))
    contexts <- sphere(n, d2)
  } else {
    arms <- matrix(stats::rnorm(n * K * d1), n * K, d1)
    contexts <- matrix(stats::rnorm(n * d2), n, d2)
  }
  arms <- aperm(array(arms, c(K, n, d1)), c(2, 1, 3))           # row (t - 1) K + a is arm a of round t
  theta <- if (is.null(main)) vec(Theta) else c(main, vec(Theta))
  if (family == "gaussian")
    return(stream(arms, contexts, theta, family, trials, noise = noise_sd * stats::rnorm(n),
                  main_effects = !is.null(main)))
  stream(arms, contexts, theta, family, trials, uniforms = matrix(stats::runif(n * trials), n, trials),
         main_effects = !is.null(main))
}
