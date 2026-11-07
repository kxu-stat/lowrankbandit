"""Check that the package lowrankbandit computes what the experiment code of this folder computes.

    python3 -m pip install ../python          # the package, once
    python3 check_package.py

On the same simulated rounds, this folder's collection (inference/algorithms.collect) and Algorithm 2
(fold_estimates, calibrate_loadings) are compared with the package's TwoTrackPolicy and two_track_inference: the arms
chosen in every round, the tracks' final estimates, and the estimates and standard errors of four loadings.  The
penalty constant is fixed (the package's cross-validation assigns the folds without random numbers, the experiment
code at random).  Cases: Gaussian and logistic rewards, without and with main effects.
"""
import os
import sys
import numpy as np
import lowrankbandit as lrb
from lowrankbandit.tuning import tau_rule

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference"))
import algorithms as paper                                           # noqa: E402  (inference/algorithms.py)


class StreamDesign:
    """The experiment code's design interface (next_round / reward_arm) reading the rounds of a Stream in order."""

    def __init__(self, stream, r0, main_effects, radius):
        self.stream, self.r0, self.main_sizes = stream, r0, ((stream.arms.shape[2], stream.contexts.shape[1])
                                                             if main_effects else None)
        self.d1, self.d2, self.K = stream.arms.shape[2], stream.contexts.shape[1], stream.K
        self.d = stream.rows(0).shape[1]
        self.family, self.nb, self.theta, self.radius = stream.family, stream.trials, stream.theta, radius
        self.sigma_eps = 1.0
        self.anchor_at_main_fit = main_effects
        self.g = -1

    def next_round(self, rng):
        self.g += 1
        return self.stream.rows(self.g)

    def reward_arm(self, rng, a):
        return self.stream.reward(self.g, a)


def one_case(family, main_effects, n=1200, d=5, r=1, K=6, seed=11):
    Theta = lrb.low_rank_truth(d, d, r, rng=seed)
    main = 0.3 * np.ones(2 * d) if main_effects else None
    scale = 1.0 if family == "gaussian" else 2.0
    stream = lrb.simulate_stream(n, d, d, K, scale * Theta, family=family, noise_sd=0.3, main=main, seed=seed + 1)
    p0 = 2 * d if main_effects else 0
    tau = 40 if main_effects else tau_rule(n, d, d, 1 / 20, p0)
    radius = None if family == "gaussian" else (10.0,) * (3 if main_effects else 1)
    c_lam = 0.25
    # the experiment code: its exploration arms come from its generator
    D = StreamDesign(stream, r, main_effects, radius)
    N_max = 2 * ((n - tau) // 2)
    rec = paper.collect(D, tau, N_max, np.random.default_rng(seed + 2), c_lam)
    # the package, with the same exploration arms
    explore = np.random.default_rng(seed + 2).integers(K, size=tau)
    pol = lrb.TwoTrackPolicy(d, d, n, family=family, main_effects=main_effects, tau=tau, c_lam=c_lam,
                             radius=radius, explore_arms=explore)
    lrb.run(pol, stream)
    data = pol.data()
    same_arms = (np.array_equal(rec["arm_ex"], data["arm_ex"]) and
                 all(np.array_equal(rec["arm"][j], data["arm"][j]) for j in (0, 1)))
    rows = max(float(np.max(np.abs(rec["Z"][j] - data["Z"][j]))) for j in (0, 1))
    # Algorithm 2 with n0 = n_tr / 2 on both sides, four loadings
    rng = np.random.default_rng(seed + 3)
    As = rng.standard_normal((4, data["Z_ex"].shape[1]))
    n0 = data["n_tr"] // 2
    F = paper.fold_estimates(D, rec, N_max, n0)
    out_paper = paper.calibrate_loadings(D, rec, N_max, As, F)
    out_pkg = lrb.two_track_inference(data, As, rank=r, n0=n0)
    mu_p = np.array([o["mu"] for o in out_paper])
    V_p = np.array([o["V"] for o in out_paper])
    d_mu = float(np.max(np.abs(mu_p - out_pkg["estimate"]) / V_p))
    d_V = float(np.max(np.abs(V_p - out_pkg["se"]) / V_p))
    d_th = max(float(np.linalg.norm(F["th_hat"][j] - out_pkg["theta"][j])) for j in (0, 1))
    print("%-9s main effects %-5s  arms identical: %s  rows max diff %.1e  theta_hat diff %.1e  "
          "|mu diff| / se %.1e  |se diff| / se %.1e"
          % (family, main_effects, same_arms, rows, d_th, d_mu, d_V))
    return same_arms and d_mu < 1e-6 and d_V < 1e-6


if __name__ == "__main__":
    ok = all([one_case("gaussian", False), one_case("binomial", False),
              one_case("gaussian", True), one_case("binomial", True)])
    print("ALL CASES AGREE" if ok else "SOME CASES DIFFER")
