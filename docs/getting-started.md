# Getting started

## Installation

=== "Python"

    ```bash
    pip install "git+https://github.com/kxu-stat/lowrankbandit.git#subdirectory=python"
    ```

    Python 3.9 or later, with numpy and scipy.

=== "R"

    ```r
    # install.packages("remotes")
    remotes::install_github("kxu-stat/lowrankbandit", subdir = "R")
    ```

    R 4.0 or later; base R only.

## Choosing arms: Algorithm 1

The example simulates an ad platform: 3,000 rounds, each with 20 candidate ads described by 10 features and one user
described by 10 features. The expected reward of an ad is \(m^\top \Theta^* x\), with \(m\) the ad's features,
\(x\) the user's and \(\Theta^*\) a rank-2 matrix. The policy sees the features, chooses one ad, and observes only that
ad's reward.

=== "Python"

    ```python
    import numpy as np
    import lowrankbandit as lrb

    d1, d2, K, n = 10, 10, 20, 3000
    Theta = lrb.low_rank_truth(d1, d2, rank=2, rng=1)
    stream = lrb.simulate_stream(n, d1, d2, K, Theta, noise_sd=0.5, seed=2)

    policy = lrb.GreedyPolicy(d1, d2, horizon=n, seed=3)
    regret = np.zeros(n)
    for t in range(n):
        a = policy.choose(stream.arms[t], stream.contexts[t])  # the ad shown
        policy.observe(stream.reward(t, a))                    # its reward
        regret[t] = stream.regret(t, a)                        # known here

    theta_hat = policy.estimate()
    error = np.linalg.norm(theta_hat - lrb.vec(Theta)) / np.linalg.norm(Theta)
    print(f"exploration rounds: {policy.tau}")
    print(f"cumulative regret: {regret.sum():.1f}")
    print(f"regret in the last 1,000 rounds: {regret[-1000:].sum():.1f}")
    print(f"relative error of Theta: {error:.3f}")
    ```

    ```text
    exploration rounds: 13
    cumulative regret: 124.2
    regret in the last 1,000 rounds: 4.0
    relative error of Theta: 0.067
    ```

=== "R"

    ```r
    library(lowrankbandit)
    set.seed(1)

    d1 <- 10; d2 <- 10; K <- 20; n <- 3000
    Theta <- low_rank_truth(d1, d2, rank = 2)
    s <- simulate_stream(n, d1, d2, K, Theta, noise_sd = 0.5)

    policy <- greedy_policy(d1, d2, horizon = n)
    regret <- numeric(n)
    for (t in 1:n) {
      a <- policy$choose(s$arms[t, , ], s$contexts[t, ])  # the ad shown, 1..K
      policy$observe(s$reward(t, a))                      # its reward
      regret[t] <- s$regret(t, a)                         # known here
    }
    theta_hat <- policy$estimate()
    cat("exploration rounds:", policy$tau, "\n")
    cat("cumulative regret:", round(sum(regret), 1), "\n")
    cat("regret in the last 1,000 rounds:", round(sum(tail(regret, 1000)), 1), "\n")
    cat("relative error of Theta:", round(sqrt(sum((theta_hat - vec(Theta))^2) / sum(Theta^2)), 3), "\n")
    ```

The policy explores for 13 rounds, then chooses greedily. Over the last 1,000 rounds it lost a total of 4 units of
expected reward against the best ad of each round. `run(policy, stream)` runs the same loop and returns the regret
of each round.

In an application, the loop is your system: pass the candidates' features and the context to `choose`, show the
chosen arm, and pass its reward to `observe`. Arms are numbered from 0 in Python and from 1 in R.

## Confidence intervals: Algorithm 2

For inference, collect the data with the two-track policy, then call `two_track_inference` with the loading \(A\) of
each target \(\mu^* = \langle A, \theta^* \rangle\). Here the targets are one entry of \(\Theta^*\) and the
interaction effect \(m^\top \Theta^* x\) of a new ad \(m\) on a new user \(x\).

=== "Python"

    ```python
    policy = lrb.TwoTrackPolicy(d1, d2, horizon=n, seed=3)
    lrb.run(policy, stream)
    data = policy.data()

    # targets: the entry Theta[0, 0], and m' Theta x for a new ad m and user x
    rng = np.random.default_rng(4)
    m, x = rng.standard_normal(d1), rng.standard_normal(d2)
    e, f = np.eye(d1)[0], np.eye(d2)[0]
    A = np.vstack([lrb.interaction_loading(np.outer(e, f), data["p0"]),
                   lrb.interaction_loading(np.outer(m, x), data["p0"])])
    res = lrb.two_track_inference(data, A, rank=2)

    mu = A @ lrb.vec(Theta)                                    # the true values
    for k in range(2):
        est, lo, hi = res["estimate"][k], res["lower"][k], res["upper"][k]
        print(f"estimate {est:.3f}  CI [{lo:.3f}, {hi:.3f}]  truth {mu[k]:.3f}")
    ```

    ```text
    estimate 0.022  CI [0.003, 0.042]  truth 0.024
    estimate 1.204  CI [0.813, 1.595]  truth 0.999
    ```

=== "R"

    ```r
    policy <- two_track_policy(d1, d2, horizon = n)
    regret <- run(policy, s)
    data <- policy$data()

    # targets: the entry Theta[1, 1], and m' Theta x for a new ad m and user x
    m <- rnorm(d1); x <- rnorm(d2)
    e <- diag(d1)[, 1]; f <- diag(d2)[, 1]
    A <- rbind(interaction_loading(e %o% f, data$p0),
               interaction_loading(m %o% x, data$p0))
    res <- two_track_inference(data, A, rank = 2)

    mu <- as.vector(A %*% vec(Theta))                          # the true values
    round(cbind(estimate = res$estimate, lower = res$lower, upper = res$upper, truth = mu), 3)
    ```

For each target, the result holds `estimate`, `se` and the ends `lower` and `upper` of the 95% confidence interval
(CI), and the quantities behind them: the split `n0`, and each fold's estimate and standard error. The R example
draws its own stream from R's random numbers, so its numbers differ from those shown; given the same stream and
exploration arms, the two languages agree (see [Settings](settings.md#python-and-r)).

!!! warning "Use data collected by the two-track policy"

    The intervals rely on how the two tracks were collected: each track's choices depend only on the shared
    exploration and its own rounds. Data logged by another policy cannot be used.

## Binary rewards

For clicks, set `family="binomial"` (and `trials` for counts out of several trials). The reward model is then
logistic, and the domain of the estimator is bounded by default; see [Settings](settings.md).

=== "Python"

    ```python
    policy = lrb.TwoTrackPolicy(d1, d2, horizon=n, family="binomial")
    ```

=== "R"

    ```r
    policy <- two_track_policy(d1, d2, horizon = n, family = "binomial")
    ```

## Main effects and further covariates

With `main_effects=True` the model adds the main effects of the arm and context features,
\(m^\top\theta_m + x^\top\theta_x\); with `n_covariates=q`, a further \(q\) covariates per arm, passed to `choose` as
`covariates`. Both are unpenalized. Their coefficients come first in the parameter, so a target that involves only
the interaction is built with `interaction_loading(A_Theta, data["p0"])`, `p0` being their number.

## Complete scripts

`python/examples/quickstart.py` and `R/inst/examples/quickstart.R` run the whole workflow, including every choice of
`n0`.
