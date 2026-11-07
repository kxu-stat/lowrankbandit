"""Properties of the building blocks that do not depend on the fixtures."""
import numpy as np
import pytest
import lowrankbandit as lrb
from lowrankbandit.model import design_rows, tangent_project, tangent_basis, truncate
from lowrankbandit.estimator import prox
from lowrankbandit.inference import fold_fit, choose_n0


def test_design_rows_give_the_bilinear_form():
    rng = np.random.default_rng(1)
    m, x, Theta = rng.standard_normal((3, 4)), rng.standard_normal(5), rng.standard_normal((4, 5))
    Z = design_rows(m, x, main_effects=True)
    theta = np.concatenate([np.ones(4), 2 * np.ones(5), lrb.vec(Theta)])
    assert np.allclose(Z @ theta, m.sum(axis=1) + 2 * x.sum() + m @ Theta @ x)


def test_prox_soft_thresholds_the_singular_values():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 3))
    out = prox(lrb.vec(M), 1.0, 0.7, 4, 3).reshape(4, 3, order="F")
    s, s_out = np.linalg.svd(M, compute_uv=False), np.linalg.svd(out, compute_uv=False)
    assert np.allclose(s_out, np.maximum(s - 0.7, 0.0))


def test_tangent_basis_spans_the_tangent_space():
    rng = np.random.default_rng(3)
    U = np.linalg.qr(rng.standard_normal((5, 2)))[0]
    V = np.linalg.qr(rng.standard_normal((4, 2)))[0]
    Q = tangent_basis(U, V, 5, 4, 3)
    assert Q.shape == (3 + 20, 3 + 2 * (5 + 4 - 2))
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]))
    v = rng.standard_normal(23)
    assert np.allclose(Q @ (Q.T @ v), tangent_project(v, U, V, 5, 4))


def test_fit_is_stationary():
    """At the Gaussian fit, the gradient's interaction block is lam times a subgradient of the nuclear norm."""
    rng = np.random.default_rng(4)
    d1, d2, s = 4, 3, 300
    Theta = np.outer(rng.standard_normal(d1), rng.standard_normal(d2))
    Z = np.array([design_rows(rng.standard_normal((1, d1)), rng.standard_normal(d2))[0] for _ in range(s)])
    y = Z @ lrb.vec(Theta) + 0.1 * rng.standard_normal(s)
    lam = 0.05
    th = lrb.fit_nuclear(Z, y, lam, d1, d2, "gaussian", tol=1e-10, iters=20000)
    G = (Z.T @ (Z @ th - y) / s).reshape(d1, d2, order="F")
    U, sv, Vt = np.linalg.svd(th.reshape(d1, d2, order="F"))
    r = int(np.sum(sv > 1e-8))
    assert np.allclose(U[:, :r].T @ (-G) @ Vt[:r].T, lam * np.eye(r), atol=1e-6)
    assert np.linalg.norm(-G, 2) <= lam * (1 + 1e-6)


def test_refit_and_calibration_run_and_rules_differ():
    d, K, n = 4, 5, 600
    Theta = lrb.low_rank_truth(d, d, 1, rng=5)
    stream = lrb.simulate_stream(n, d, d, K, Theta, noise_sd=0.3, seed=6)
    policy = lrb.TwoTrackPolicy(d, d, n, c_lam=0.25, seed=7)
    lrb.run(policy, stream)
    data = policy.data()
    assert choose_n0(data, "equal") == data["n_tr"] // 2
    with pytest.raises(ValueError):
        choose_n0(data, 1)                                            # n0 must be at least 2
    with pytest.raises(ValueError):
        choose_n0(data, "design")                                     # the design-stage split needs kappa_D
    fold = fold_fit(data, 0, data["n_tr"] // 2, 1)
    assert fold["rank"] == 1 and fold["kappa"] > 0
    A = lrb.interaction_loading(np.eye(d)[:, :1] @ np.eye(d)[:1], 0)
    res = lrb.two_track_inference(policy.data(), A, rank=1)
    assert res["lower"] < res["estimate"] < res["upper"]
    assert abs(res["estimate"] - float(A @ stream.theta)) < 5 * res["se"]


def test_greedy_policy_estimate_is_close():
    d, K, n = 4, 5, 800
    Theta = lrb.low_rank_truth(d, d, 1, rng=8)
    stream = lrb.simulate_stream(n, d, d, K, Theta, noise_sd=0.2, seed=9)
    policy = lrb.GreedyPolicy(d, d, n, c_lam=0.25, seed=10)
    regret = lrb.run(policy, stream)
    est = policy.estimate()
    assert np.linalg.norm(est - lrb.vec(Theta)) < 0.3
    assert regret[-200:].mean() < regret[:policy.tau].mean()
