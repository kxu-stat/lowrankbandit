# lowrankbandit (Python)

Greedy contextual bandits with a low-rank interaction between arm features and contexts (Algorithm 1), and
asymptotically valid confidence intervals for linear functionals of the parameter after greedy data collection
(Algorithm 2). The R package in the same repository has the same functions and, given the same rounds, returns the
same numbers.

## Installation

```bash
pip install "git+https://github.com/kxu-stat/lowrankbandit.git#subdirectory=python"
```

## Example

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

| Function | Purpose |
|---|---|
| `GreedyPolicy` | Algorithm 1: `choose`, `observe`, `estimate` |
| `TwoTrackPolicy` | data collection for inference: `choose`, `observe`, `data` |
| `two_track_inference` | estimates, standard errors and confidence intervals |
| `fit_nuclear` | the nuclear-norm-penalized estimator on given rows |
| `interaction_loading`, `vec` | loadings of targets |
| `simulate_stream`, `Stream`, `run`, `low_rank_truth` | simulated rounds |

Documentation: <https://kxu-stat.github.io/lowrankbandit/>. A complete script: `examples/quickstart.py`.
Tests: `pip install -e ".[test]"`, then `pytest tests`.
