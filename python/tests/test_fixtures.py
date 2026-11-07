"""The package against the shared fixtures (python/dev/make_fixtures.py): the same rounds give the same arms, pilot,
cross-validation choices and Algorithm 2 results as when the fixtures were written.  The R package's tests read the
same files."""
import os
import numpy as np
import pytest
import lowrankbandit as lrb

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASES = sorted(os.listdir(FIXTURES))


def read(folder, name):
    return np.loadtxt(os.path.join(folder, name + ".csv"), delimiter=",", ndmin=2)


@pytest.mark.parametrize("case", CASES)
def test_case(case):
    folder = os.path.join(FIXTURES, case)
    with open(os.path.join(folder, "config.csv")) as f:
        keys, values = [line.strip().split(",") for line in f]
    cfg = dict(zip(keys, values))
    n, d1, d2, K, tau = (int(cfg[key]) for key in ("n", "d1", "d2", "K", "tau"))
    main_effects = cfg["main_effects"] == "1"
    c_lam = "cv" if cfg["c_lam"] == "cv" else float(cfg["c_lam"])
    arms = read(folder, "arms").reshape(n, K, d1)
    theta = read(folder, "theta").ravel()
    if cfg["family"] == "gaussian":
        stream = lrb.Stream(arms, read(folder, "contexts"), theta, "gaussian", noise=read(folder, "noise").ravel(),
                            main_effects=main_effects)
    else:
        stream = lrb.Stream(arms, read(folder, "contexts"), theta, "binomial", uniforms=read(folder, "uniforms"),
                            main_effects=main_effects)
    explore = read(folder, "explore_arms").ravel().astype(int)
    policy = lrb.TwoTrackPolicy(d1, d2, n, family=cfg["family"], main_effects=main_effects, tau=tau, c_lam=c_lam,
                                explore_arms=explore)
    lrb.run(policy, stream)
    data = policy.data()
    expected_arms = read(folder, "expected_arms")
    assert np.array_equal(data["arm"][0], expected_arms[0]) and np.array_equal(data["arm"][1], expected_arms[1])
    assert np.allclose(data["pilot"], read(folder, "expected_pilot").ravel(), rtol=0, atol=1e-10)
    if data["cv"] is not None:
        assert np.array_equal(np.vstack(data["cv"]), read(folder, "expected_cv"))
    As = read(folder, "loadings")
    expected = read(folder, "expected_inference")
    for k, rule in enumerate(("design", "information", "equal")):
        res = lrb.two_track_inference(data, As, rank=int(cfg["r"]), n0=rule, kappa_D=0.2)
        e = expected[3 * k:3 * k + 3]
        assert np.all(res["n0"] == e[:, 0])
        assert np.all(np.abs(res["estimate"] - e[:, 1]) <= 1e-8 * e[:, 2])
        assert np.all(np.abs(res["se"] - e[:, 2]) <= 1e-8 * e[:, 2])
    greedy = lrb.GreedyPolicy(d1, d2, n, family=cfg["family"], main_effects=main_effects, tau=tau, c_lam=c_lam,
                              explore_arms=explore)
    lrb.run(greedy, stream)
    assert np.array_equal(greedy.data()["arm"][0], read(folder, "expected_greedy_arms").ravel())
    assert np.allclose(greedy.estimate(), read(folder, "expected_greedy_estimate").ravel(), rtol=0, atol=1e-10)
