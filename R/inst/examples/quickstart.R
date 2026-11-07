# Quick start: collect data with the two-track policy and compute a confidence interval.
#
#     Rscript quickstart.R      (or: source(system.file("examples", "quickstart.R", package = "lowrankbandit")))
#
# In your own application, replace the simulated stream by your system: in every round pass the candidates' features
# and the context to choose(), show the chosen arm, and pass its reward to observe().
library(lowrankbandit)
set.seed(1)

d1 <- 8; d2 <- 8; K <- 10; n <- 4000

# simulated rounds: arms and contexts uniform on spheres, a rank-2 interaction, Gaussian noise
Theta <- low_rank_truth(d1, d2, rank = 2)
s <- simulate_stream(n, d1, d2, K, Theta, family = "gaussian", noise_sd = 0.2)

# Algorithm 1 alone (regret only)
greedy <- greedy_policy(d1, d2, horizon = n)
regret <- run(greedy, s)
cat(sprintf("Algorithm 1: cumulative regret %.1f over %d rounds\n", sum(regret), length(regret)))

# Step 1 of Algorithm 2: the two-track collection, one round at a time
policy <- two_track_policy(d1, d2, horizon = n)
for (t in 1:policy$horizon) {
  a <- policy$choose(s$arms[t, , ], s$contexts[t, ])           # the arm to show (1, ..., K)
  policy$observe(s$reward(t, a))                               # its reward
}
data <- policy$data()

# Steps 2-4: intervals for two targets, the entry Theta[1, 1] and a contrast between two ads and two users
m <- diag(d1)[, 1]; m2 <- diag(d1)[, 2]
x <- diag(d2)[, 1]; x2 <- diag(d2)[, 2]
A <- rbind(interaction_loading(m %o% x, data$p0),
           interaction_loading((m - m2) %o% (x - x2), data$p0))
res <- two_track_inference(data, A, rank = 2)                  # n0 = "equal" by default
truth <- as.vector(A %*% vec(Theta))
for (k in 1:2)
  cat(sprintf("target %d: estimate %.4f, 95%% interval [%.4f, %.4f], truth %.4f\n",
              k, res$estimate[k], res$lower[k], res$upper[k], truth[k]))
cat(sprintf("n0 = %d of n_tr = %d rounds per track for the preliminary fit\n", res$n0, data$n_tr))

# other choices of n0: the information rule, the design-stage split with the planning value kappa_D = 1 (Gaussian
# rewards and E[z z'] = I, as here), or a number
for (rule in list("information", "design", data$n_tr %/% 3)) {
  r <- two_track_inference(data, A[1, ], rank = 2, n0 = rule, kappa_D = 1)
  cat(sprintf("n0 = %-12s -> %4d: estimate %.4f, se %.4f\n", format(rule), r$n0, r$estimate, r$se))
}
