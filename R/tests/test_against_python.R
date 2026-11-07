# The R package against the Python package on the same rounds (fixtures written by python/dev/make_fixtures.py).
#
# For every case: the same exploration arms, the same greedy arms in every round of both tracks, the same pilot
# estimate, and the same n0, estimates and standard errors of Algorithm 2 under the three rules for n0.
library(lowrankbandit)

read <- function(folder, name) as.matrix(utils::read.csv(file.path(folder, paste0(name, ".csv")), header = FALSE))

check_case <- function(folder) {
  cfg <- utils::read.csv(file.path(folder, "config.csv"), stringsAsFactors = FALSE)
  n <- cfg$n; d1 <- cfg$d1; d2 <- cfg$d2; K <- cfg$K
  arms_rows <- read(folder, "arms")                                   # row (t - 1) K + a
  arms <- array(0, c(n, K, d1))
  for (t in 1:n) arms[t, , ] <- arms_rows[(t - 1) * K + 1:K, ]
  contexts <- read(folder, "contexts")
  theta <- as.vector(read(folder, "theta"))
  main_effects <- cfg$main_effects == 1
  if (cfg$family == "gaussian") {
    s <- stream(arms, contexts, theta, "gaussian", noise = as.vector(read(folder, "noise")), main_effects = main_effects)
  } else {
    s <- stream(arms, contexts, theta, "binomial", uniforms = read(folder, "uniforms"), main_effects = main_effects)
  }
  c_lam <- if (cfg$c_lam == "cv") "cv" else as.numeric(cfg$c_lam)
  policy <- two_track_policy(d1, d2, n, family = cfg$family, main_effects = main_effects, tau = cfg$tau, c_lam = c_lam,
                             explore_arms = as.vector(read(folder, "explore_arms")) + 1)
  run(policy, s)
  data <- policy$data()

  expected_arms <- read(folder, "expected_arms")
  same_arms <- all(data$arm[[1]] - 1 == expected_arms[1, ]) && all(data$arm[[2]] - 1 == expected_arms[2, ])
  pilot <- as.vector(read(folder, "expected_pilot"))
  d_pilot <- max(abs(data$pilot - pilot)) / max(1, max(abs(pilot)))
  d_score <- abs(data$score - read(folder, "expected_score")[1, 1]) / read(folder, "expected_score")[1, 1]
  if (!is.null(data$cv)) {
    cv_expected <- read(folder, "expected_cv")
    same_cv <- isTRUE(all.equal(unname(rbind(data$cv[[1]], data$cv[[2]])), unname(cv_expected)))
  } else {
    same_cv <- TRUE
  }

  # Algorithm 1 on the same rounds
  greedy <- greedy_policy(d1, d2, n, family = cfg$family, main_effects = main_effects, tau = cfg$tau, c_lam = c_lam,
                          explore_arms = as.vector(read(folder, "explore_arms")) + 1)
  greedy_regret <- run(greedy, s)
  same_greedy <- all(greedy$data()$arm[[1]] - 1 == as.vector(read(folder, "expected_greedy_arms")))
  d_greedy <- max(abs(greedy$estimate() - as.vector(read(folder, "expected_greedy_estimate"))))
  d_regret <- max(abs(greedy_regret - as.vector(read(folder, "expected_greedy_regret"))))

  As <- read(folder, "loadings")
  expected <- read(folder, "expected_inference")                   # rows: rule x loading; n0, mu, se
  d_mu <- 0; d_se <- 0; same_n0 <- TRUE
  for (k in 1:3) {
    res <- two_track_inference(data, As, rank = cfg$r, n0 = c("design", "information", "equal")[k], kappa_D = 0.2)
    e <- expected[(k - 1) * 3 + 1:3, , drop = FALSE]
    same_n0 <- same_n0 && all(res$n0 == e[, 1])
    d_mu <- max(d_mu, abs(res$estimate - e[, 2]) / e[, 3])
    d_se <- max(d_se, abs(res$se - e[, 3]) / e[, 3])
  }
  cat(sprintf(paste("%-14s two-track: arms identical %s, cv identical %s, pilot %.1e, score %.1e;",
                    "inference: n0 identical %s, |mu diff| / se %.1e, |se diff| / se %.1e;",
                    "greedy: arms identical %s, estimate %.1e, regret %.1e\n"),
              basename(folder), same_arms, same_cv, d_pilot, d_score, same_n0, d_mu, d_se, same_greedy, d_greedy,
              d_regret))
  same_arms && same_cv && same_n0 && same_greedy && d_pilot < 1e-8 && d_score < 1e-8 && d_mu < 1e-6 &&
    d_se < 1e-6 && d_greedy < 1e-8 && d_regret < 1e-8
}

folders <- list.dirs(file.path("fixtures"), recursive = FALSE)
ok <- sapply(folders, check_case)
stopifnot(length(folders) == 6, all(ok))
cat("R and Python agree on all", length(folders), "cases\n")
