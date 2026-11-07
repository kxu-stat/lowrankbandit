# lowrankbandit (R)

Greedy contextual bandits with a low-rank interaction between arm features and contexts (Algorithm 1), and
asymptotically valid confidence intervals for linear functionals of the parameter after greedy data collection
(Algorithm 2). The Python package in the same repository has the same functions and, given the same rounds, returns
the same numbers.

## Installation

```r
# install.packages("remotes")
remotes::install_github("kxu-stat/lowrankbandit", subdir = "R")
```

## Example

```r
library(lowrankbandit)
set.seed(2)
Theta <- low_rank_truth(10, 10, rank = 2)
s <- simulate_stream(3000, 10, 10, K = 20, Theta = Theta, noise_sd = 0.5)

policy <- two_track_policy(10, 10, horizon = 3000)
for (t in 1:policy$horizon) {
  a <- policy$choose(s$arms[t, , ], s$contexts[t, ])                # the ad to show (1, ..., K)
  policy$observe(s$reward(t, a))                                    # its reward
}

m <- rnorm(10); x <- rnorm(10)                                      # a new ad m and a new user x
A <- interaction_loading(m %o% x, p0 = 0)
res <- two_track_inference(policy$data(), A, rank = 2)
cat(sprintf("estimate %.3f, 95%% interval [%.3f, %.3f]\ntruth %.3f\n",
            res$estimate, res$lower, res$upper, drop(m %*% Theta %*% x)))
```

| Function | Purpose |
|---|---|
| `greedy_policy()` | Algorithm 1: `choose()`, `observe()`, `estimate()` |
| `two_track_policy()` | data collection for inference: `choose()`, `observe()`, `data()` |
| `two_track_inference()` | estimates, standard errors and confidence intervals |
| `fit_nuclear()` | the nuclear-norm-penalized estimator on given rows |
| `interaction_loading()`, `vec()` | loadings of targets |
| `simulate_stream()`, `stream()`, `run()`, `low_rank_truth()` | simulated rounds |

Documentation: <https://kxu-stat.github.io/lowrankbandit/>. A complete script: `inst/examples/quickstart.R`.
