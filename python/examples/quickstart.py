"""Quick start: collect data with the two-track policy and compute a confidence interval.

    python3 quickstart.py

In your own application, replace the simulated stream by your system: in every round pass the candidates' features
and the context to choose(), show the chosen arm, and pass its reward to observe().
"""
import numpy as np
import lowrankbandit as lrb

d1, d2, K, n = 8, 8, 10, 4000

# simulated rounds: arms and contexts uniform on spheres, a rank-2 interaction, Gaussian noise
Theta = lrb.low_rank_truth(d1, d2, rank=2, rng=1)
stream = lrb.simulate_stream(n, d1, d2, K, Theta, family="gaussian", noise_sd=0.2, seed=2)

# Algorithm 1 alone (regret only)
greedy = lrb.GreedyPolicy(d1, d2, horizon=n, seed=3)
regret = lrb.run(greedy, stream)
print("Algorithm 1: cumulative regret %.1f over %d rounds" % (regret.sum(), len(regret)))

# Step 1 of Algorithm 2: the two-track collection, one round at a time
policy = lrb.TwoTrackPolicy(d1, d2, horizon=n, seed=3)
for t in range(policy.horizon):
    a = policy.choose(stream.arms[t], stream.contexts[t])        # the arm to show
    policy.observe(stream.reward(t, a))                          # its reward
data = policy.data()

# Steps 2-4: intervals for two targets, the entry Theta[0, 0] and a contrast between two ads and two users
m, m2 = np.eye(d1)[0], np.eye(d1)[1]
x, x2 = np.eye(d2)[0], np.eye(d2)[1]
A = np.vstack([lrb.interaction_loading(np.outer(m, x), data["p0"]),
               lrb.interaction_loading(np.outer(m - m2, x - x2), data["p0"])])
res = lrb.two_track_inference(data, A, rank=2)                   # n0 = "equal" by default
truth = A @ lrb.vec(Theta)
for k in range(2):
    print("target %d: estimate %.4f, 95%% interval [%.4f, %.4f], truth %.4f"
          % (k + 1, res["estimate"][k], res["lower"][k], res["upper"][k], truth[k]))
print("n0 = %d of n_tr = %d rounds per track for the preliminary fit" % (res["n0"], data["n_tr"]))

# other choices of n0: the information rule, the design-stage split with the planning value kappa_D = 1 (Gaussian
# rewards and E[z z'] = I, as here), or a number
for rule in ("information", "design", data["n_tr"] // 3):
    r = lrb.two_track_inference(data, A[0], rank=2, n0=rule, kappa_D=1.0)
    print("n0 = %-12s -> %4d: estimate %.4f, se %.4f" % (rule, r["n0"], r["estimate"], r["se"]))
