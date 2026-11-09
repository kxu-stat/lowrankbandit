"""Simulated data for examples and checks.

A Stream fixes every round in advance: the candidates' features, the contexts, and the randomness of the rewards
(Gaussian noise, or uniform numbers for the binomial draws).  With the true parameter, the reward of any arm in any
round is then determined, so different policies -- or the Python and R versions of this package -- can be run on
the same rounds.
"""
import numpy as np
from .model import design_rows, mean_function, sigmoid, vec


class Stream:
    """A fixed sequence of n rounds, so that several policies can be run on the same data.

    Parameters
    ----------
    arms : array of shape (n, K, d1)
        The candidates' features m_{t,a}.
    contexts : array of shape (n, d2)
        The contexts x_t.
    theta : array of shape (p,)
        The true parameter: main effects first, then vec(Theta*).
    family : {"gaussian", "binomial"}
        The reward model.
    trials : int
        The number of trials of the binomial family.
    noise : array of shape (n,), optional
        Gaussian family: the noise, y = <z, theta> + noise_t.
    uniforms : array of shape (n, trials), optional
        Binomial family: uniform numbers, y = #{k : uniforms_{t,k} < sigmoid(<z, theta>)}.
    covariates : array of shape (n, K, q), optional
        Further covariates of the candidates.
    main_effects : bool
        Whether theta contains the main effects (m, x), after the covariates' coefficients.
    """

    def __init__(self, arms, contexts, theta, family="gaussian", trials=1, noise=None, uniforms=None,
                 covariates=None, main_effects=False):
        self.arms = np.asarray(arms, dtype=float)
        self.contexts = np.asarray(contexts, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.family, self.trials, self.main_effects = family, trials, main_effects
        self.noise = None if noise is None else np.asarray(noise, dtype=float)
        self.uniforms = None if uniforms is None else np.asarray(uniforms, dtype=float).reshape(len(self.arms), -1)
        self.covariates = None if covariates is None else np.asarray(covariates, dtype=float)
        self.n, self.K = self.arms.shape[0], self.arms.shape[1]
        # u[t, a] = <z_{t,a}, theta> = c' gamma + m' theta_m + x' theta_x + m' Theta x, for every round and arm at once
        d1, d2 = self.arms.shape[2], self.contexts.shape[1]
        q = 0 if self.covariates is None else self.covariates.shape[2]
        p0 = self.theta.size - d1 * d2
        Theta = self.theta[p0:].reshape(d1, d2, order="F")
        self.u = np.zeros((self.n, self.K))
        for a in range(self.K):
            m = self.arms[:, a]
            self.u[:, a] = np.sum((m @ Theta) * self.contexts, axis=1)
            if q > 0:
                self.u[:, a] += self.covariates[:, a] @ self.theta[:q]
            if main_effects:
                self.u[:, a] += m @ self.theta[q:q + d1] + self.contexts @ self.theta[q + d1:p0]

    def rows(self, t):
        """The design rows of round t."""
        c = None if self.covariates is None else self.covariates[t]
        return design_rows(self.arms[t], self.contexts[t], c, self.main_effects)

    def means(self, t):
        """The mean rewards psi'(<z_{t,a}, theta*>) of the K candidates of round t."""
        return mean_function(self.u[t], self.family, self.trials)

    def reward(self, t, a):
        """The reward of arm a in round t."""
        if self.family == "gaussian":
            return float(self.u[t, a] + self.noise[t])
        return float(np.sum(self.uniforms[t] < sigmoid(self.u[t, a])))

    def regret(self, t, a):
        """The best mean reward of round t minus the mean reward of arm a."""
        m = self.means(t)
        return float(m.max() - m[a])


def run(policy, stream, rounds=None):
    """Run a policy on a stream.

    Parameters
    ----------
    policy : GreedyPolicy or TwoTrackPolicy
    stream : Stream
    rounds : int, optional
        The number of rounds; default: the policy's horizon.

    Returns
    -------
    array of shape (rounds,)
        The regret of each round: the best mean reward minus the mean reward of the chosen arm.
    """
    rounds = min(stream.n, policy.horizon) if rounds is None else rounds
    regret = np.zeros(rounds)
    for t in range(rounds):
        c = None if stream.covariates is None else stream.covariates[t]
        a = policy.choose(stream.arms[t], stream.contexts[t], c)
        policy.observe(stream.reward(t, a))
        regret[t] = stream.regret(t, a)
    return regret


def low_rank_truth(d1, d2, rank, signal=1.0, rng=None):
    """A random rank-r matrix with equal singular values.

    Theta* = signal U V' / sqrt(rank), with U (d1 x rank) and V (d2 x rank) random orthonormal, so that
    ||Theta*||_F = signal.  `rng` is a seed or a numpy Generator.
    """
    rng = np.random.default_rng(rng)
    U = np.linalg.qr(rng.standard_normal((d1, rank)))[0]
    V = np.linalg.qr(rng.standard_normal((d2, rank)))[0]
    return signal * U @ V.T / np.sqrt(rank)


def sphere(rng, n, d):
    """n points uniform on the sphere of radius sqrt(d) in R^d (second moment I)."""
    z = rng.standard_normal((n, d))
    return np.sqrt(d) * z / np.linalg.norm(z, axis=1, keepdims=True)


def simulate_stream(n, d1, d2, K, Theta, family="gaussian", trials=1, noise_sd=0.2, main=None, seed=None,
                    covariate_law="sphere"):
    """A simulated stream of n rounds.

    Parameters
    ----------
    n : int
        The number of rounds.
    d1, d2 : int
        The dimensions of the arm features and of the context.
    K : int
        The number of candidates per round.
    Theta : array of shape (d1, d2)
        The true interaction.
    family : {"gaussian", "binomial"}
        The reward model.
    trials : int
        The number of trials of the binomial family.
    noise_sd : float
        The standard deviation of the Gaussian noise.
    main : array of shape (d1 + d2,), optional
        The main effects (theta_m, theta_x); none by default.
    seed : int, optional
        The seed of all random draws.
    covariate_law : {"sphere", "normal"}
        The features and contexts: uniform on the sphere of radius sqrt(d), or standard normal.  Both have second
        moment I.

    Returns
    -------
    Stream
    """
    rng = np.random.default_rng(seed)
    if covariate_law == "sphere":
        arms = sphere(rng, n * K, d1).reshape(n, K, d1)
        contexts = sphere(rng, n, d2)
    else:
        arms = rng.standard_normal((n, K, d1))
        contexts = rng.standard_normal((n, d2))
    theta = vec(Theta) if main is None else np.concatenate([np.asarray(main, dtype=float), vec(Theta)])
    if family == "gaussian":
        return Stream(arms, contexts, theta, family, trials, noise=noise_sd * rng.standard_normal(n),
                      main_effects=main is not None)
    return Stream(arms, contexts, theta, family, trials, uniforms=rng.random((n, trials)), main_effects=main is not None)
