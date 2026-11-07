# Properties of the building blocks that do not depend on the fixtures.
library(lowrankbandit)

set.seed(1)
m <- matrix(rnorm(12), 3, 4); x <- rnorm(5); Theta <- matrix(rnorm(20), 4, 5)
Z <- lowrankbandit:::design_rows(m, x, main_effects = TRUE)
theta <- c(rep(1, 4), rep(2, 5), vec(Theta))
stopifnot(isTRUE(all.equal(as.vector(Z %*% theta), rowSums(m) + 2 * sum(x) + as.vector(m %*% Theta %*% x))))

# the Gaussian fit is stationary: the gradient's interaction block is lam times a subgradient of the nuclear norm
d1 <- 4; d2 <- 3; s <- 300
Theta <- outer(rnorm(d1), rnorm(d2))
Z <- t(sapply(1:s, function(i) lowrankbandit:::design_rows(matrix(rnorm(d1), 1), rnorm(d2))[1, ]))
y <- as.vector(Z %*% vec(Theta)) + 0.1 * rnorm(s)
lam <- 0.05
th <- fit_nuclear(Z, y, lam, d1, d2, "gaussian", tol = 1e-10, iters = 20000)
G <- matrix(crossprod(Z, as.vector(Z %*% th) - y) / s, d1, d2)
sv <- svd(matrix(th, d1, d2))
r <- sum(sv$d > 1e-8)
stopifnot(max(abs(t(sv$u[, 1:r, drop = FALSE]) %*% (-G) %*% sv$v[, 1:r, drop = FALSE] - lam * diag(r))) < 1e-6)
stopifnot(max(svd(-G)$d) <= lam * (1 + 1e-6))

# a full run: collection and inference
Theta <- low_rank_truth(4, 4, 1)
s <- simulate_stream(600, 4, 4, 5, Theta, noise_sd = 0.3)
policy <- two_track_policy(4, 4, 600, c_lam = 0.25)
run(policy, s)
A <- interaction_loading(diag(4)[, 1, drop = FALSE] %*% diag(4)[1, , drop = FALSE], 0)
res <- two_track_inference(policy$data(), A, rank = 1)
stopifnot(res$lower < res$estimate, res$estimate < res$upper, abs(res$estimate - sum(A * vec(Theta))) < 5 * res$se)
stopifnot(inherits(try(two_track_inference(policy$data(), A, rank = 1, n0 = "design"), silent = TRUE),
                   "try-error"))                                # the design-stage split needs kappa_D
cat("basic checks passed\n")
