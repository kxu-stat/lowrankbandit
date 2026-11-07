"""Write the shared test data (run by hand when the method changes).

    python3 make_fixtures.py          # from python/dev

For each case it simulates a stream, runs TwoTrackPolicy and two_track_inference, and writes the stream and the
results as CSV files into python/tests/fixtures/CASE and R/tests/fixtures/CASE.  The Python tests check that the
package still gives these results; the R tests check that the R package gives the same ones on the same rounds.
Arms are written numbered from 0 (the R tests add 1).
"""
import os
import numpy as np
import lowrankbandit as lrb

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [os.path.join(HERE, "..", "tests", "fixtures"), os.path.join(HERE, "..", "..", "R", "tests", "fixtures")]
KAPPA_D = 0.2           # the planning value for n0 = "design": small enough that the split is lengthened

CASES = {
    #  name                family      main   n    tau   c_lam  d1 d2 seed
    "gaussian":          ("gaussian", False, 500, None, 0.25,  4, 4, 101),
    "gaussian_cv":       ("gaussian", False, 640, None, "cv",  4, 4, 102),
    "gaussian_main":     ("gaussian", True,  500, 30,   0.25,  4, 4, 103),
    "binomial":          ("binomial", False, 700, None, 0.25,  4, 4, 104),
    "binomial_main":     ("binomial", True,  700, 40,   0.25,  4, 4, 105),
    "rectangular":       ("gaussian", True,  600, None, "cv",  5, 3, 106),
}
R, K = 1, 5


def write(folder, name, array):
    np.savetxt(os.path.join(folder, name + ".csv"), np.atleast_2d(array), delimiter=",", fmt="%.17g")


def one_case(name, family, main_effects, n, tau, c_lam, d1, d2, seed):
    Theta = lrb.low_rank_truth(d1, d2, R, rng=seed) * (1.0 if family == "gaussian" else 2.0)
    main = 0.3 * np.ones(d1 + d2) if main_effects else None
    stream = lrb.simulate_stream(n, d1, d2, K, Theta, family=family, noise_sd=0.3, main=main, seed=seed + 1)
    explore = np.random.default_rng(seed + 2).integers(K, size=n)
    policy = lrb.TwoTrackPolicy(d1, d2, n, family=family, main_effects=main_effects, tau=tau, c_lam=c_lam,
                                explore_arms=explore)
    lrb.run(policy, stream)
    data = policy.data()
    greedy = lrb.GreedyPolicy(d1, d2, n, family=family, main_effects=main_effects, tau=tau, c_lam=c_lam,
                              explore_arms=explore)                         # Algorithm 1 on the same rounds
    greedy_regret = lrb.run(greedy, stream)
    greedy_arms = greedy.data()["arm"][0]
    greedy_estimate = greedy.estimate()
    As = np.random.default_rng(seed + 3).standard_normal((3, data["Z_ex"].shape[1]))
    results = []
    for rule in ("design", "information", "equal"):
        res = lrb.two_track_inference(data, As, rank=R, n0=rule, kappa_D=KAPPA_D)
        results.append(np.column_stack([np.full(3, res["n0"]), res["estimate"], res["se"]]))
    for folder in TARGETS:
        out = os.path.join(folder, name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.csv"), "w") as f:
            f.write("family,main_effects,n,tau,c_lam,d1,d2,r,K\n")
            f.write("%s,%d,%d,%d,%s,%d,%d,%d,%d\n" % (family, main_effects, n, data["tau"], c_lam, d1, d2, R, K))
        write(out, "arms", stream.arms.reshape(n * K, d1))                      # row t K + a
        write(out, "contexts", stream.contexts)
        write(out, "theta", stream.theta)
        if family == "gaussian":
            write(out, "noise", stream.noise)
        else:
            write(out, "uniforms", stream.uniforms)
        write(out, "explore_arms", explore[:data["tau"]])
        write(out, "loadings", As)
        # the expected results
        write(out, "expected_arms", np.vstack([data["arm"][0], data["arm"][1]]))
        write(out, "expected_pilot", data["pilot"])
        write(out, "expected_score", [data["score"]])
        write(out, "expected_inference", np.vstack(results))                  # rows: rule x loading; n0, mu, se
        write(out, "expected_greedy_arms", greedy_arms)
        write(out, "expected_greedy_estimate", greedy_estimate)
        write(out, "expected_greedy_regret", greedy_regret)
        if data["cv"] is not None:
            write(out, "expected_cv", np.vstack([np.array(c) for c in data["cv"]]))
    print("%-15s tau %3d  n_tr %3d  cv %s" % (name, data["tau"], data["n_tr"], data["cv"]))


if __name__ == "__main__":
    for case, spec in CASES.items():
        one_case(case, *spec)
