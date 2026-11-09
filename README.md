# lowrankbandit

[![Python tests](https://github.com/kxu-stat/lowrankbandit/actions/workflows/python.yml/badge.svg)](https://github.com/kxu-stat/lowrankbandit/actions/workflows/python.yml)
[![R-CMD-check](https://github.com/kxu-stat/lowrankbandit/actions/workflows/R-CMD-check.yml/badge.svg)](https://github.com/kxu-stat/lowrankbandit/actions/workflows/R-CMD-check.yml)

Python and R implementations of the methods in

> Ke Xu, Yuefeng Han and Zifeng Zhao. Regret Minimization and Statistical Inference for Low-Rank Contextual
> Generalized Linear Bandits, with Applications to Advertising and Product Design. Manuscript, 2026.

In round $t$ a policy shows one of $K$ arms, with features $m_{t,a} \in \mathbb{R}^{d_1}$, for a context
$x_t \in \mathbb{R}^{d_2}$; the reward follows a generalized linear model whose linear predictor contains the
interaction $m_{t,a}^\top \Theta x_t$, with $\Theta$ of low rank.

- `GreedyPolicy` (Algorithm 1): a short uniform exploration, then greedy choices from a nuclear-norm-penalized GLM
  estimate refitted every round, with the penalty constant chosen by cross-validation inside the run.
- `TwoTrackPolicy` and `two_track_inference` (Algorithm 2): data collection by two greedy tracks after a shared
  exploration, then estimates, standard errors and confidence intervals for linear functionals
  $\langle A, \theta^* \rangle$ of the parameter, asymptotically valid although the arms were chosen greedily.

The R package has the same functions, arguments and defaults; given the same rounds, the two packages choose the same
arms and return the same estimates up to rounding error.

## Installation

Python 3.11 or later, with numpy and scipy:

```bash
pip install "git+https://github.com/kxu-stat/lowrankbandit.git#subdirectory=python"
```

R 4.0 or later, with Rcpp and RcppEigen; installing from GitHub compiles C++ code, which needs Rtools on Windows and
the Xcode command line tools on macOS:

```r
remotes::install_github("kxu-stat/lowrankbandit", subdir = "R")
```

## Example

Simulate 3,000 rounds, collect the data with the two-track policy, and estimate the effect $m^\top \Theta x$ of a new
ad $m$ on a new user $x$:

```python
import numpy as np
import lowrankbandit as lrb

# each round shows one of 20 candidate ads (10 features each) to one user (10 features)
Theta = lrb.low_rank_truth(10, 10, rank=2, rng=1)                  # the rank-2 interaction
stream = lrb.simulate_stream(3000, 10, 10, K=20, Theta=Theta, noise_sd=0.5, seed=2)

policy = lrb.TwoTrackPolicy(10, 10, horizon=3000, seed=3)
for t in range(policy.horizon):
    a = policy.choose(stream.arms[t], stream.contexts[t])          # the ad to show
    policy.observe(stream.reward(t, a))                            # its reward

# the target: the effect m' Theta x of a new ad m on a new user x
rng = np.random.default_rng(4)
m, x = rng.standard_normal(10), rng.standard_normal(10)
A = lrb.interaction_loading(np.outer(m, x), p0=0)
res = lrb.two_track_inference(policy.data(), A, rank=2)
print(f"estimate {res['estimate']:.3f}, 95% interval [{res['lower']:.3f}, {res['upper']:.3f}]")
print(f"truth {m @ Theta @ x:.3f}")
```

```text
estimate 1.204, 95% interval [0.813, 1.595]
truth 0.999
```

<details>
<summary>The same in R</summary>

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

```text
estimate 1.542, 95% interval [1.302, 1.782]
truth 1.397
```

R draws its own random numbers, so its stream, and hence its numbers, differ from Python's.

</details>

In an application, the loop is your system: pass the candidates' features and the context to `choose`, show the
chosen arm, and pass its reward to `observe`.

## Documentation

<https://kxu-stat.github.io/lowrankbandit/>: the algorithms step by step, the settings and their defaults, and the
function reference for Python and R.

## Reproducing the paper

[`reproduction/`](reproduction) holds the code that produced the paper's tables and figures (the regret comparisons
with OFUL and ESTR, the inference simulations and the MIND news-recommendation study), with instructions for every
run.

## Citation

```bibtex
@unpublished{xu2026lowrankbandit,
  title  = {Regret Minimization and Statistical Inference for Low-Rank Contextual Generalized Linear Bandits,
            with Applications to Advertising and Product Design},
  author = {Xu, Ke and Han, Yuefeng and Zhao, Zifeng},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

MIT, except the two functions from the code of Jun, Willett, Wright and Nowak (2019) in
[`reproduction/regret/lrbandit/`](reproduction/regret/lrbandit), which keep their Apache-2.0 license.
