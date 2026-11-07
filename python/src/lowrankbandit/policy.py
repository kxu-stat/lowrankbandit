"""Data collection: Algorithm 1 (GreedyPolicy) and Step 1 of Algorithm 2 (TwoTrackPolicy).

Both policies explore for tau rounds (the arm drawn uniformly), fit the pilot estimate on the exploration rows S_o,
and then choose greedily: the arm with the largest fitted linear predictor <z_{t,a}, theta_hat>.  GreedyPolicy has one
greedy track.  TwoTrackPolicy has two tracks that take the rounds in turn; each track refits on S_o and its own
rounds only, so the two records are independent given S_o (Step 1 of Algorithm 2).

Use, one round at a time:

    a = policy.choose(arms, context)        # arms: K x d1 features of the candidates, context: length d2
    policy.observe(y)                       # the reward of the chosen arm

Before a track chooses in its round k > 0 it refits its estimator on S_o and its first k rounds at
lambda_{tau + k}, starting from its previous estimate; the penalty constant is fixed (c_lam a number) or chosen by
cross-validation (c_lam = "cv").
"""
import numpy as np
from .model import (check_family, design_rows, main_groups, variance_function)
from .estimator import fista, proximal_newton, top_eigenvalue
from .tuning import (C_TAU, CV_START, tau_rule, pilot_score, penalty_level, cv_checkpoints, cv_constant)

BINOMIAL_RADIUS = 10.0   # the default domain C_R under the binomial family: every main-effect group and Theta in a
                         # ball of radius 10 (log-odds units), so that the fits stay finite when the exploration rows
                         # are separable; pass radius to change it


class _Collector:
    """Exploration followed by `n_tracks` greedy tracks that take the rounds in turn."""

    def __init__(self, n_tracks, d1, d2, horizon, family="gaussian", trials=1, main_effects=False, n_covariates=0,
                 tau=None, c_tau=C_TAU, c_lam="cv", radius=None, explore_arms=None, seed=None):
        check_family(family, trials)
        self.d1, self.d2, self.family, self.trials = d1, d2, family, trials
        self.main_effects, self.q = main_effects, n_covariates
        self.sizes = main_groups(d1, d2, n_covariates, main_effects)
        self.p0 = sum(self.sizes)
        self.p = self.p0 + d1 * d2
        if radius is None and family == "binomial":
            radius = (BINOMIAL_RADIUS,) * (len(self.sizes) + 1)
        self.radius = None if radius is None else tuple(radius)
        self.tau = tau_rule(horizon, d1, d2, c_tau, self.p0) if tau is None else int(tau)
        self.n_tracks = n_tracks
        self.n_tr = (horizon - self.tau) // n_tracks              # rounds per track
        if self.tau < 2 or self.n_tr < 4:
            raise ValueError("the horizon is too short for tau = %d exploration rounds" % self.tau)
        self.horizon = self.tau + n_tracks * self.n_tr           # the rounds used (the n of the penalty level)
        self.cv = isinstance(c_lam, str) and c_lam == "cv"
        self.c_lam = c_lam if self.cv else float(c_lam)
        self.explore_arms = None if explore_arms is None else [int(a) for a in explore_arms]
        self.rng = np.random.default_rng(seed)
        self.t = 0                                               # rounds observed
        self.pending = None                                      # (design rows, arm, track) of the open round
        # exploration record S_o
        self.Z_ex, self.y_ex, self.arm_ex = [], [], []
        self.G_ex, self.b_ex = np.zeros((self.p, self.p)), np.zeros(self.p)
        self.score = self.pilot = self.base = None
        # the tracks, filled in when the exploration ends:
        #   Z, y     the rows S_o followed by the track's own rounds;  k  the track's rounds so far;  arm  its arms
        #   theta    the track's current estimate, fitted on s_fit rows
        #   G, b     sum z z' and sum y z over the track's rows (the Gaussian fit needs only these)
        #   c, cv    the penalty constant in force and the cross-validation choices [(s, c), ...]
        #   L, H, LH, s_H   solver caches: the top eigenvalue of G / s (every round up to k = 200, then every tenth),
        #            the curvature-weighted Gram matrix (binomial; rebuilt exactly when s has grown by a quarter since
        #            s_H, updated row by row in between) and its top eigenvalue
        self.tracks = [dict(Z=None, y=None, k=0, arm=[], theta=None, s_fit=0, G=None, b=None, c=None, cv=[],
                            L=None, H=None, LH=None, s_H=0) for _ in range(n_tracks)]
        self.checkpoints = cv_checkpoints(self.tau, self.tau + self.n_tr) if self.cv else []

    # -------------------------------------------------------------- one round
    def choose(self, arms, context, covariates=None):
        """The arm to show in this round.

        Parameters
        ----------
        arms : array of shape (K, d1)
            The features of the K candidates.
        context : array of shape (d2,)
            The context of the round.
        covariates : array of shape (K, q), optional
            Further covariates of the candidates, when the policy was made with `n_covariates=q`.

        Returns
        -------
        int
            The index of the chosen arm (a row of `arms`).
        """
        if self.pending is not None:
            raise RuntimeError("observe() the reward of the previous round first")
        if self.t >= self.horizon:
            raise RuntimeError("the horizon has been reached")
        Zc = design_rows(arms, context, covariates, self.main_effects)
        if Zc.shape[1] != self.p:
            raise ValueError("the design rows have %d columns, expected %d" % (Zc.shape[1], self.p))
        if self.t < self.tau:                                    # exploration: an arm drawn uniformly
            a = self.explore_arms[self.t] if self.explore_arms is not None else int(self.rng.integers(len(Zc)))
            self.pending = (Zc, a, None)
            return a
        j, k = (self.t - self.tau) % self.n_tracks, (self.t - self.tau) // self.n_tracks
        if k > 0:
            self._refit(j, self.tau + k)
        a = int(np.argmax(Zc @ self.tracks[j]["theta"]))         # the greedy choice
        self.pending = (Zc, a, j)
        return a

    def observe(self, y):
        """Record the reward of the arm chosen in this round.

        Parameters
        ----------
        y : float
            The observed reward (for the binomial family, the number of successes).
        """
        if self.pending is None:
            raise RuntimeError("choose() an arm first")
        Zc, a, j = self.pending
        z, y = Zc[a], float(y)
        self.pending = None
        if j is None:                                            # an exploration round
            self.Z_ex.append(z); self.y_ex.append(y); self.arm_ex.append(a)
            self.G_ex += np.outer(z, z); self.b_ex += y * z
            self.t += 1
            if self.t == self.tau:
                self._fit_pilot()
            return
        tr = self.tracks[j]
        s = self.tau + tr["k"]
        tr["Z"][s], tr["y"][s] = z, y
        tr["k"] += 1
        tr["arm"].append(a)
        tr["G"] += np.outer(z, z); tr["b"] += y * z
        if self.family != "gaussian" and tr["H"] is not None:   # running curvature-weighted Gram
            tr["H"] += variance_function(z @ tr["theta"], self.family, self.trials) * np.outer(z, z)
        self.t += 1

    # -------------------------------------------------------------- fits
    def _fit_pilot(self):
        """The pilot estimate on S_o at lambda_tau, and the penalty levels lambda_s = c base(s)."""
        Z, y = np.array(self.Z_ex), np.array(self.y_ex)
        self.score = pilot_score(Z, y, self.d1, self.d2, self.family, self.trials, self.radius, self.sizes)
        self.base = lambda s: penalty_level(1.0, self.score, self.tau, self.horizon, self.d1, self.d2, s)
        c0 = CV_START if self.cv else self.c_lam
        lam = c0 * self.base(self.tau)
        if self.family == "gaussian":
            self.pilot = fista(self.G_ex / self.tau, self.b_ex / self.tau, lam, self.d1, self.d2, np.zeros(self.p),
                               radius=self.radius, sizes=self.sizes)
        else:
            self.pilot = proximal_newton(Z, y, lam, self.d1, self.d2, np.zeros(self.p), self.trials, self.radius,
                                         sizes=self.sizes)
        for tr in self.tracks:
            tr["theta"], tr["s_fit"], tr["c"] = self.pilot.copy(), self.tau, c0
            tr["G"], tr["b"] = self.G_ex.copy(), self.b_ex.copy()
            tr["Z"], tr["y"] = np.zeros((self.tau + self.n_tr, self.p)), np.zeros(self.tau + self.n_tr)
            tr["Z"][:self.tau], tr["y"][:self.tau] = Z, y

    def _rows(self, j):
        """S_o followed by track j's rounds so far."""
        tr = self.tracks[j]
        s = self.tau + tr["k"]
        return tr["Z"][:s], tr["y"][:s]

    def _refit(self, j, s):
        """Track j's estimate on its s = tau + k rows, started at its previous estimate."""
        tr, k = self.tracks[j], s - self.tau
        if s == tr["s_fit"]:                                     # already fitted on these rows
            return
        tr["s_fit"] = s
        if self.cv and s in self.checkpoints:                    # choose c again on the track's rows
            Z, y = self._rows(j)
            tr["c"] = cv_constant(Z, y, self.base, tr["theta"], self.d1, self.d2, self.family, self.trials,
                                  self.radius, self.sizes)
            tr["cv"].append((s, tr["c"]))
        lam = tr["c"] * self.base(s)
        if k <= 200 or k % 10 == 0 or tr["L"] is None:           # G / s moves by O(1/s): every tenth round is enough
            tr["L"] = top_eigenvalue(tr["G"] / s)
        if self.family == "gaussian":
            tr["theta"] = fista(tr["G"] / s, tr["b"] / s, lam, self.d1, self.d2, tr["theta"], L=tr["L"],
                                radius=self.radius, sizes=self.sizes)
            return
        Z, y = self._rows(j)
        if s >= 1.25 * tr["s_H"]:                                 # exact curvature-weighted Gram, now and then
            w = variance_function(Z @ tr["theta"], self.family, self.trials)
            tr["H"] = Z.T @ (Z * w[:, None])
            tr["s_H"] = s
            tr["LH"] = top_eigenvalue(tr["H"] / s)
        tr["theta"] = proximal_newton(Z, y, lam, self.d1, self.d2, tr["theta"], self.trials, self.radius,
                                      H=tr["H"] / s, LH=tr["LH"], L_max=0.25 * self.trials * tr["L"], sizes=self.sizes)

    # -------------------------------------------------------------- results
    def data(self):
        """The records of the collection, for `two_track_inference`.

        Returns
        -------
        dict
            The exploration record (`Z_ex`, `y_ex`, `arm_ex`), the two tracks (`Z`, `y`, `arm`), the pilot estimate
            and the tuning in force (`tau`, `c_lam`, the cross-validation choices `cv`), with the dimensions.
        """
        if self.t < self.horizon:
            raise RuntimeError("the collection is not complete: %d of %d rounds observed" % (self.t, self.horizon))
        return dict(d1=self.d1, d2=self.d2, p0=self.p0, sizes=self.sizes, family=self.family, trials=self.trials,
                    radius=self.radius, tau=self.tau, horizon=self.horizon, n_tr=self.n_tr, score=self.score,
                    c_lam=self.c_lam, pilot=self.pilot.copy(),
                    Z_ex=np.array(self.Z_ex), y_ex=np.array(self.y_ex), arm_ex=np.array(self.arm_ex),
                    Z=[tr["Z"][self.tau:].copy() for tr in self.tracks],
                    y=[tr["y"][self.tau:].copy() for tr in self.tracks],
                    arm=[np.array(tr["arm"]) for tr in self.tracks],
                    cv=[list(tr["cv"]) for tr in self.tracks] if self.cv else None)


class GreedyPolicy(_Collector):
    """Algorithm 1: a short uniform exploration, then greedy choices from the penalized fit on all rounds so far.

    In every round, call `choose` with the candidates' features and the context, show the returned arm, and pass its
    reward to `observe`.  `estimate` returns the current estimate.

    Parameters
    ----------
    d1, d2 : int
        Dimensions of the arm features m and of the context x.
    horizon : int
        The number of rounds n.
    family : {"gaussian", "binomial"}
        The reward model: Gaussian, or binomial with `trials` trials (`trials=1` for clicks).
    trials : int
        The number of trials of the binomial family.
    main_effects : bool
        Include the main effects (m, x) in the model.  Under the binomial family the exploration must then be long
        enough for the main-effects-only logistic fit to exist: about twice as many rounds as main-effect coefficients.
    n_covariates : int
        The number q of further unpenalized covariates per arm, passed to `choose` as `covariates`.
    tau : int, optional
        The exploration length.  Default: ceil(c_tau (d1 + d2 + p0) log(n d1 d2)).
    c_tau : float
        The constant of the default exploration length.
    c_lam : float or "cv"
        The penalty constant: a number, or "cv" for 10-fold cross-validation at s = max(2 tau, 200), 2s, 4s, ...
    radius : tuple of float, optional
        The domain C_R: radii (R_1, ..., R_G, R_Theta) of the main-effect groups and of Theta.  Default: none under
        the Gaussian family, 10 for every group under the binomial family.
    explore_arms : sequence of int, optional
        The arms of the exploration rounds.  Default: drawn uniformly by a generator seeded with `seed`.
    seed : int, optional
        The seed of the exploration draws.

    Examples
    --------
    >>> import lowrankbandit as lrb
    >>> stream = lrb.simulate_stream(2000, 6, 6, K=8, Theta=lrb.low_rank_truth(6, 6, 1, rng=1), seed=2)
    >>> policy = lrb.GreedyPolicy(6, 6, horizon=2000)
    >>> regret = lrb.run(policy, stream)
    >>> theta_hat = policy.estimate()
    """

    def __init__(self, d1, d2, horizon, family="gaussian", trials=1, main_effects=False, n_covariates=0, tau=None,
                 c_tau=1 / 20, c_lam="cv", radius=None, explore_arms=None, seed=None):
        super().__init__(1, d1, d2, horizon, family, trials, main_effects, n_covariates, tau, c_tau, c_lam, radius,
                         explore_arms, seed)

    def estimate(self):
        """The current estimate: main effects first, then vec(Theta), refitted on all rounds so far."""
        k = self.t - self.tau
        if k < 0:
            raise RuntimeError("the exploration has not finished")
        if k > 0 and self.pending is None:
            self._refit(0, self.t)
        return self.tracks[0]["theta"].copy()


class TwoTrackPolicy(_Collector):
    """Step 1 of Algorithm 2: data collection for inference.

    A shared uniform exploration S_o, then two greedy tracks that take the rounds in turn; each track refits on S_o
    and its own rounds only.  After the last round, `data` returns the records for `two_track_inference`.  The policy
    uses tau + 2 floor((n - tau) / 2) rounds, one fewer than n when n - tau is odd (`policy.horizon`).

    Parameters
    ----------
    d1, d2 : int
        Dimensions of the arm features m and of the context x.
    horizon : int
        The number of rounds n.
    family : {"gaussian", "binomial"}
        The reward model: Gaussian, or binomial with `trials` trials (`trials=1` for clicks).
    trials : int
        The number of trials of the binomial family.
    main_effects : bool
        Include the main effects (m, x) in the model.  Under the binomial family the exploration must then be long
        enough for the main-effects-only logistic fit to exist: about twice as many rounds as main-effect coefficients.
    n_covariates : int
        The number q of further unpenalized covariates per arm, passed to `choose` as `covariates`.
    tau : int, optional
        The exploration length.  Default: ceil(c_tau (d1 + d2 + p0) log(n d1 d2)).
    c_tau : float
        The constant of the default exploration length.
    c_lam : float or "cv"
        The penalty constant: a number, or "cv" for 10-fold cross-validation at s = max(2 tau, 200), 2s, 4s, ...
    radius : tuple of float, optional
        The domain C_R: radii (R_1, ..., R_G, R_Theta) of the main-effect groups and of Theta.  Default: none under
        the Gaussian family, 10 for every group under the binomial family.
    explore_arms : sequence of int, optional
        The arms of the exploration rounds.  Default: drawn uniformly by a generator seeded with `seed`.
    seed : int, optional
        The seed of the exploration draws.

    Examples
    --------
    >>> import lowrankbandit as lrb
    >>> Theta = lrb.low_rank_truth(6, 6, 1, rng=1)
    >>> stream = lrb.simulate_stream(4000, 6, 6, K=8, Theta=Theta, seed=2)
    >>> policy = lrb.TwoTrackPolicy(6, 6, horizon=4000)
    >>> regret = lrb.run(policy, stream)
    >>> A = lrb.interaction_loading(Theta / (Theta ** 2).sum() ** 0.5, p0=0)
    >>> result = lrb.two_track_inference(policy.data(), A, rank=1)
    """

    def __init__(self, d1, d2, horizon, family="gaussian", trials=1, main_effects=False, n_covariates=0, tau=None,
                 c_tau=1 / 20, c_lam="cv", radius=None, explore_arms=None, seed=None):
        super().__init__(2, d1, d2, horizon, family, trials, main_effects, n_covariates, tau, c_tau, c_lam, radius,
                         explore_arms, seed)
