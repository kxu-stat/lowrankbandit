"""A small Monte Carlo check of Algorithm 2 with the package's defaults (run by hand; a few minutes).

    python3 coverage_check.py gaussian 200
    python3 coverage_check.py binomial 100

Each replicate draws new rounds, collects them with TwoTrackPolicy (cross-validated penalty) and computes the 95%
interval for an entry of Theta*.  Prints the coverage and the mean of (estimate - truth) / se.
"""
import sys
import numpy as np
import lowrankbandit as lrb

family = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 100
d, r, K, n = 6, 1, 8, 2000
Theta = lrb.low_rank_truth(d, d, r, rng=1) * (1.0 if family == "gaussian" else 2.0)
A = lrb.interaction_loading(np.outer(np.eye(d)[0], np.eye(d)[0]), 0)
mu_star = float(A @ lrb.vec(Theta))
T = []
for b in range(B):
    stream = lrb.simulate_stream(n, d, d, K, Theta, family=family, noise_sd=0.3, seed=1000 + b)
    policy = lrb.TwoTrackPolicy(d, d, n, family=family, seed=2000 + b)
    lrb.run(policy, stream)
    res = lrb.two_track_inference(policy.data(), A, rank=r)
    T.append((res["estimate"] - mu_star) / res["se"])
    if (b + 1) % 20 == 0:
        T_ = np.array(T)
        print("%s, %d replicates: coverage %.3f, mean T %.3f, sd T %.3f"
              % (family, b + 1, np.mean(np.abs(T_) <= 1.959964), T_.mean(), T_.std()), flush=True)
