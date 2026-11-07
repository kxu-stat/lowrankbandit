"""The comparison baselines OFUL and ESTR of Jun, Willett, Wright and Nowak (ICML 2019, "Bilinear bandits with low-rank
structure"), run on our contextual interface.

Source: github.com/kwangsungjun/lrbandit, commit 2f1f7ca (Apache-2.0).  `rankone` and `calc_sqrt_beta_det4` are vendored
in lrbandit/; line numbers refer to blbandits3.py.  The authors' classes keep a fixed catalog of arm pairs, while our
experiment shows K new arms and one context per round, so their per-round arithmetic is re-implemented here; every
departure the interface forces is marked [ADAPT].

Both methods pull the optimistic arm of a ridge regression on z = vec(m x') (row-major):

    index(a) = <z_a, theta_hat> + sqrt(multiplier) sqrt_beta ||z_a||_{V^-1},
    sqrt_beta = R sqrt(log det V - log det V_0 + log(1 / delta^2)) + Sp,   delta = 0.2.

  OFUL (455-604)  V_0 = lambda I, lambda = 1, Sp = sqrt(lambda) S_F.
  ESTR (618-833)  stage 1: T1 = ceil(C_T1 r(2d - r)) exploration rounds, the rank-r Burer-Monteiro recovery of the
                  observed (m, x, y) and its SVD; stage 2: LowOFUL in the rotated coordinates, regularization lambda on
                  the r(2d - r) coordinates in the first r rows or columns and lambda_perp = T2 / (d r log(1 + T2 /
                  lambda)) on the remaining (d - r)^2 (765, 790-795),
                  Sp = sqrt(lambda) S_F + S2 sqrt(lambda_perp) R^2 d^3 r / T1 ('simple2', 781-784), after replaying
                  the stage-1 observations (804-805).
S_F and S2 are the Frobenius and spectral norms of the link-scale truth (the authors take them from the true
parameter) and R the noise scale.  Tuning grids (expr01_defs.py, grid 1): OFUL multiplier in 10^{-2, -1.5, ..., 0};
ESTR C_T1 in 10^{0, 0.25, ..., 1} times the same multipliers.  The authors' selection rule is kept as
`select_configuration` for reference; the study reports every configuration and the best one by mean in hindsight.

Units.  The released constants are calibrated for unit-norm arms.  For features s times and rewards a times those of
unit-norm data (spec["arm_scale"] = s; a = s for Gaussian and 1 for binary rewards) the methods reproduce the authors'
code on the unit-scale data exactly when lambda -> s^2 lambda, lambda_perp -> s^2 lambda_perp, the BM regularization
C -> s C, its starts -> sqrt(a / s) starts and tolTh -> (a / s) tolTh, and 'simple2' uses R / a; R, S_F and S2 are
declared in the data's units.  arm_scale = 1 is the released arithmetic.

Binary rewards.  coding "pm1" (the authors' treatment, the supplement): the linear arithmetic on 2y - 1, as the
authors' binaryRewards flag does for OFUL (559-561) [ADAPT: ESTR gets the same coding], with R = 1, S_F = amp / 2,
S2 = amp / (2 sqrt r).  coding "glm" (our logistic extension, the main binary competitors): OFUL with the ridge-
penalized logistic MLE and its likelihood ellipsoid (Faury et al., 2020); ESTR with stage 1 the rank-r nuclear-norm
penalized logistic MLE computed by the authors' alternation (`rankone_glm`, lambda_1 = GLM_BM_C1 sqrt(T1); Lu, Meisami
and Tewari, 2021) and stage 2 LowGLM-UCB (Kang, Hsieh and Lee, 2022) with lambda_perp = L T2 / (d r log(1 + L T2 /
lambda)), L = nb / 4; R = 1/2; run in unit-norm units.  Every formula reduces to the authors' for Gaussian rewards.

Main effects (MIND).  With covariates c (env.main) OFUL regresses on (c, vec(m x')); ESTR uses the augmented arm (m, c)
and context (x, 1, 0, ..., 0), zero-padded to equal length because the authors' routines are square [ADAPT], and is
given rank r + 1 (realdata/mind.estr_matrix gives the augmented truth).
"""
import numpy as np
from scipy import stats
from scipy.linalg import cho_solve
from lrbandit.matrixrecovery import rankone, calc_sqrt_beta_det4
from algorithms import psi, psi1, psi2, curvature_bound

DELTA, LAMBDA = 0.2, 1.0
# ESTR's stage 1 minimizes the authors' Burer-Monteiro objective: their `rankone` with its defaults C = 0.1,
# tolPred = tolTh = 0.01, maxIter = 400 (blbandits3.py line 737; tolTh converted to the data's units).  The authors
# start it once from ra.randn(d, r); in their catalog design the problem is a low-rank approximation of a fully
# observed matrix and one start finds the minimizer.  With new arms every round the observations are random rank-one
# projections, and at the short T1 that ESTR's tuning selects the objective has spurious stationary points, so the
# minimizer is computed from 50 random starts (the smallest objective is kept); GLM-ESTR likewise.
BM = dict(C=0.1, tolPred=0.01, tolTh=0.01, maxIter=400, starts=50)
GLM_BM = dict(tolTh=1e-4, maxIter=8000, starts=50)
GLM_BM_C1 = 0.1          # GLM-ESTR's stage-1 penalty lambda_1 = 0.1 sqrt(T1)


# ================================================================ the grids
def specs(d, rank, K, noise_sd, amp, horizon, main_dim=0, S_F=None, S2=None, coding=None, arm_scale=1.0):
    """The released grid 1, one dict per configuration in the authors' order.  noise_sd, amp, S_F and S2 are in the
    data's units (arm_scale: see the module docstring).  main_dim = q > 0: main-effect covariates (S_F and S2 are then
    given).  coding = "pm1": the authors' binary treatment (R = 1, linear-scale amp / 2)."""
    if coding == "pm1":
        noise_sd, amp = 1.0, amp / 2.0
    multipliers = 10.0 ** (-2 + np.arange(5) / 2)
    d_est, rank_est = (d + main_dim, rank + 1) if main_dim else (d, rank)
    if S_F is None:
        S_F, S2 = amp, amp / np.sqrt(rank)
    k = rank_est * (2 * d_est - rank_est)
    out = []
    for j, m in enumerate(multipliers):
        out.append(dict(method="OFUL", config_id="OFUL_g1_m%02d" % j, C_T1=None, multiplier=m, T1=0,
                        source_parameter_index=j, eligible=True))
    for i, C in enumerate(10.0 ** (np.arange(5) / 4)):
        for j, m in enumerate(multipliers):
            T1 = int(np.ceil(C * k))
            out.append(dict(method="ESTR", config_id="ESTR_BM_g1_t%02d_m%02d" % (i, j), C_T1=C,
                            multiplier=m, T1=T1, source_parameter_index=i * 5 + j, eligible=T1 < horizon))
    for sp in out:
        sp.update(d=d, rank=rank, K=K, noise_sd=noise_sd, amp=amp, S_F=S_F, S2=S2, horizon=horizon, coding=coding,
                  arm_scale=float(arm_scale))
        if main_dim:
            sp.update(main_dim=main_dim, d_est=d_est, rank_est=rank_est)
    return out


def select_configuration(regrets, config_ids):
    """The authors' selection rule (analyze_expr01_20190119-9_paper.py): the configuration minimizing
    mean + t_.975 sd / sqrt(B) over the B replications, first minimum in source order.  regrets: B x n_configs."""
    B = regrets.shape[0]
    score = regrets.mean(axis=0) + stats.t.ppf(0.975, B - 1) * regrets.std(axis=0, ddof=1) / np.sqrt(B)
    return config_ids[int(np.argmin(score))], score


# ================================================================ the ridge regression of LowOFUL
class Ridge:
    """The regression state of (Low)OFUL: V, its inverse, W'y, theta_hat and log det V.  Under the logistic extension V
    is the working information H = Lambda + sum_i psi''(z_i' theta_(i-1)) z_i z_i' (each weight at the estimate when
    the row arrived), rebuilt exactly at the current estimate when the prefix has grown by a quarter."""

    def __init__(self, regularization, R, Sp, multiplier, family, nb, capacity, scale=1.0):
        p = len(regularization)
        self.Lambda, self.R, self.Sp, self.multiplier = np.asarray(regularization, float), R, Sp, multiplier
        self.scale = scale                               # arm_scale s, for the logistic stopping test
        self.family, self.nb, self.n = family, nb, 0
        self.V = np.diag(self.Lambda)                    # 505-509: diagonal regularization D
        self.invV = np.diag(1.0 / self.Lambda)
        self.WTy = np.zeros(p)
        self.logdetV0 = self.logdetV = np.sum(np.log(self.Lambda))
        self.theta = np.zeros(p)
        if family == "binomial":                          # logistic extension: the rows
            self.Z, self.y = np.zeros((capacity, p)), np.zeros(capacity)
            self.H_n = 0                                  # the prefix length at the last exact rebuild of V

    def radius(self):
        """sqrt(multiplier) sqrt_beta (line 528)."""
        beta = calc_sqrt_beta_det4(self.n + 1, self.R, LAMBDA, DELTA, self.Sp,
                                   max(self.logdetV, self.logdetV0), self.logdetV0)
        return np.sqrt(self.multiplier) * beta

    def leverage(self, Z):
        """||z||^2_{V^-1} for the rows of Z [ADAPT: computed for the K current features]."""
        return np.einsum("ij,ij->i", Z @ self.invV, Z)

    def index(self, Z):
        """The optimistic index of the K candidates (532-537)."""
        return Z @ self.theta + self.radius() * np.sqrt(np.maximum(self.leverage(Z), 0.0))

    def update(self, z, y):
        self.n += 1
        if self.family == "gaussian":                     # 555-590
            self.WTy += y * z
            self.V += np.outer(z, z)
            tempval1 = self.invV @ z
            tempval2 = tempval1 @ z
            self.logdetV += np.log(1.0 + tempval2)
            if self.n % 100 == 0:
                self.invV = np.linalg.inv(self.V)
            else:
                a = tempval1 / np.sqrt(1.0 + tempval2)
                self.invV -= np.outer(a, a)
            self.theta = self.invV @ self.WTy
        else:
            self._update_glm(z, y)

    def _rebuild(self, theta):
        """The working information rebuilt exactly at theta from every row; its inverse and log determinant."""
        Z = self.Z[:self.n]
        w = psi2(Z @ theta, self.family, self.nb)
        self.V = np.diag(self.Lambda) + (Z * w[:, None]).T @ Z
        self.H_n = self.n
        self._invert()

    def _invert(self):
        """V^-1 and log det V from the Cholesky factor of V."""
        chol = np.linalg.cholesky(self.V)
        self.invV = cho_solve((chol, True), np.eye(len(self.Lambda)))
        self.logdetV = 2.0 * np.sum(np.log(np.diag(chol)))

    def _update_glm(self, z, y, tol=1e-7):
        """Ridge-penalized logistic MLE by inexact Newton steps (line-search Newton-CG, Nocedal and Wright, 2006,
        Algorithm 7.1): each step solved by conjugate gradients with the exact Hessian, preconditioned by the working
        information V, to the relative residual min(0.5, sqrt(||g|| / ||g_0||)) -- that algorithm's forcing sequence
        min(0.5, sqrt(||g||)) with the gradient taken relative to g_0, the gradient at the previous round's estimate,
        so that it does not depend on the scale of the data; the steps continue until the gradient test holds.  V gains
        the new row with the weight at the current estimate (V^-1 and log det V by Sherman-Morrison and the
        determinant lemma, recomputed from V every 100 rows, as the authors do for OFUL); it is rebuilt exactly when
        the prefix has grown by 25 % or a solve needed 20 iterations."""
        j = self.n - 1
        self.Z[j], self.y[j] = z, y
        w_new = psi2(z @ self.theta, self.family, self.nb)
        self.V += w_new * np.outer(z, z)
        if self.n >= 1.25 * self.H_n:
            self._rebuild(self.theta)
        elif self.n % 100 == 0:
            self._invert()
        else:
            u = self.invV @ z
            c = w_new * (u @ z)
            self.logdetV += np.log1p(c)
            self.invV -= (w_new / (1.0 + c)) * np.outer(u, u)
        Z, ys, theta, Lam = self.Z[:self.n], self.y[:self.n], self.theta, self.Lambda

        def objective(th, e):                             # e = Z th
            return np.sum(psi(e, self.family, self.nb) - ys * e) + 0.5 * np.sum(Lam * th * th)

        eta = Z @ theta
        f = objective(theta, eta)
        g0 = None
        for _ in range(30):
            grad = Z.T @ (psi1(eta, self.family, self.nb) - ys) + Lam * theta
            s, gnorm = self.scale, np.linalg.norm(grad)   # the test in unit-scale terms
            if gnorm / s <= tol * max(1.0, s * np.linalg.norm(theta)):
                break
            g0 = gnorm if g0 is None else g0
            forcing = min(0.5, np.sqrt(gnorm / g0))
            w = psi2(eta, self.family, self.nb)
            step, res = np.zeros_like(theta), grad.copy()   # conjugate gradients, preconditioned by V^-1
            zv = self.invV @ res; direction = zv; rz = res @ zv; cg = 0
            while cg < 50:
                Hd = Lam * direction + Z.T @ (w * (Z @ direction)); a = rz / (direction @ Hd)
                step += a * direction; res -= a * Hd; cg += 1
                if np.linalg.norm(res) <= forcing * gnorm:
                    break
                zv = self.invV @ res; rz_new = res @ zv
                direction = zv + (rz_new / rz) * direction; rz = rz_new
            alpha, dec = 1.0, grad @ step                  # Armijo backtracking
            new = theta - step
            e_new = Z @ new
            f_new = objective(new, e_new)
            while f_new > f - 1e-4 * alpha * dec and alpha > 1e-3:
                alpha /= 2.0
                new = theta - alpha * step
                e_new = Z @ new
                f_new = objective(new, e_new)
            theta, eta, f = new, e_new, f_new
            if cg >= 20:
                self._rebuild(theta)
        self.theta = theta


# ================================================================ stage 1 of ESTR
def draw_starts(d, r, seed, starts):
    """`starts` standard-normal d x r starting factors from one seeded stream (the authors' ra.randn(d, r))."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((d, r)) for _ in range(starts)]


def rankone_glm(X, Z, y, r, init_U, family, nb, reg, tolTh, maxIter, newton_tol=1e-8, tolObj=1e-10):
    """The authors' `rankone` alternation for the logistic likelihood (GLM-ESTR's stage 1): minimize
    sum_i [psi(x_i' U V' z_i) - y_i x_i' U V' z_i] + (reg / 2)(||U||_F^2 + ||V||_F^2) over the d x r factors, each
    half-step a ridge-penalized logistic regression solved by Newton.  Stops when U V' moves by less than tolTh or the
    objective falls by less than tolObj of itself.  Returns U, V, the iterations and the objective path."""
    N, d = X.shape
    U = V = init_U

    def objective(U, V):
        eta = np.einsum("ij,ij->i", X @ (U @ V.T), Z)
        return float(np.sum(psi(eta, family, nb) - y * eta) + 0.5 * reg * (np.sum(U * U) + np.sum(V * V)))

    def block(D, beta):
        for _ in range(50):
            eta = D @ beta
            grad = D.T @ (psi1(eta, family, nb) - y) + reg * beta
            if np.linalg.norm(grad) <= newton_tol * max(1.0, np.linalg.norm(beta)):
                break
            w = psi2(eta, family, nb)
            Hm = (D * w[:, None]).T @ D + reg * np.eye(D.shape[1])
            step = np.linalg.solve(Hm, grad)
            f = lambda b: np.sum(psi(D @ b, family, nb) - y * (D @ b)) + 0.5 * reg * (b @ b)
            f0, alpha, dec, new = f(beta), 1.0, grad @ step, beta - step
            while f(new) > f0 - 1e-4 * alpha * dec and alpha > 1e-3:
                alpha /= 2.0
                new = beta - alpha * step
            beta = new
        return beta

    hatTh, objs = U @ V.T, [objective(U, V)]
    for it in range(1, maxIter + 1):
        if it % 2 == 0:
            D = (X[:, :, None] * (Z @ V)[:, None, :]).reshape(N, d * r); beta0 = U.ravel()
        else:
            D = (Z[:, :, None] * (X @ U)[:, None, :]).reshape(N, d * r); beta0 = V.ravel()
        sol = block(D, beta0).reshape(d, r)
        if it % 2 == 0:
            U = sol
        else:
            V = sol
        prev, hatTh = hatTh, U @ V.T
        objs.append(objective(U, V))
        if np.linalg.norm(hatTh - prev) < tolTh or objs[-2] - objs[-1] <= tolObj * abs(objs[-1]):
            break
    return U, V, it, dict(objs=objs)


def recover(M, X, y, r, R, seed, family, nb, s=1.0, a=1.0):
    """ESTR's stage-1 recovery: the BM routine from BM["starts"] (GLM-ESTR: GLM_BM["starts"]) seeded random starts,
    keeping the smallest objective.  s, a: the data's units (module docstring).  Returns Theta_hat and a record of the
    starts (the winner, and the relative objective gaps of the first start and of the first five)."""
    fits = []
    starts = BM["starts"] if family == "gaussian" else GLM_BM["starts"]
    for init_U in draw_starts(M.shape[1], r, seed, starts):
        if family == "gaussian":
            U, V, _, stat = rankone(M, X, y, r, R=R, C=s * BM["C"], tolPred=BM["tolPred"],
                                    tolTh=(a / s) * BM["tolTh"], maxIter=BM["maxIter"], init_U=np.sqrt(a / s) * init_U)
        elif s != 1.0:
            raise NotImplementedError("the logistic BM runs in unit-norm units (Baseline converts GLM-ESTR's data)")
        else:                                            # GLM-ESTR
            U, V, _, stat = rankone_glm(M, X, y, r, init_U, family, nb, GLM_BM_C1 * np.sqrt(len(y)),
                                        GLM_BM["tolTh"], GLM_BM["maxIter"])
        fits.append((stat["objs"][-1], U @ V.T))
    objectives = [f[0] for f in fits]
    if not np.all(np.isfinite(objectives)):
        raise RuntimeError("non-finite BM objective")
    best = int(np.argmin(objectives))
    info = dict(bm_starts=len(fits), bm_best_start=best + 1,
                bm_gap1=float(objectives[0] / objectives[best] - 1.0),
                bm_gap5=float(min(objectives[:5]) / objectives[best] - 1.0))
    return fits[best][1], info


# ================================================================ the policies
def rotated_features(M, x, U, V):
    """z_a = vec((U'm_a)(V'x)') for the K candidates; U = V = I gives the plain features."""
    left, right = M @ U, V.T @ x
    return (left[:, :, None] * right[None, None, :]).reshape(M.shape[0], -1)


class Baseline:
    """One configuration of OFUL or ESTR, run online: `select(t, M, x)` then `update(y)`."""

    def __init__(self, spec, horizon, family, nb, solver_seed):
        self.spec, self.n, self.horizon = spec, 0, horizon
        self.family, self.nb, self.seed = family, nb, solver_seed
        self.q = spec.get("main_dim", 0)                    # main-effect covariates (0: none)
        d, self.d, self.r = spec["d"], spec["d"], spec["rank"]
        if self.q and spec["method"] == "ESTR":            # the augmented square design
            d, self.d, self.r = spec["d_est"], spec["d_est"], spec["rank_est"]
        self.R, self.T1 = spec["noise_sd"], spec["T1"]
        self.s = spec.get("arm_scale", 1.0)                  # features s times the unit-norm ones
        binary = spec.get("coding") == "pm1" or family == "binomial"
        self.a = 1.0 if binary else self.s                  # rewards a times
        self.glm_estr = family == "binomial" and spec["method"] == "ESTR"
        self.unit = 1.0
        if self.glm_estr:                                   # GLM-ESTR runs in unit-norm units (the same logits)
            self.unit, self.s = self.s, 1.0
        self.U = self.V = np.eye(d)
        p = d * d + (self.q if spec["method"] == "OFUL" else 0)
        lam = self.s ** 2 * LAMBDA                          # lambda = 1 in the data's units
        if spec["method"] == "OFUL":                       # expr01_defs.py 176-185
            self.ridge = Ridge(np.full(p, lam), self.R, np.sqrt(lam) * spec["S_F"],
                               spec["multiplier"], family, nb, horizon, self.s)
        else:                                               # stage 1 first; the ridge is built at T1
            self.ridge = None
            self.obs_M, self.obs_x, self.obs_y = np.zeros((self.T1, d)), np.zeros((self.T1, d)), np.zeros(self.T1)

    def select(self, t, M, x, exploration_arm, C=None):
        """The arm of round t.  ESTR's stage 1 and OFUL's first round pull the environment's uniform arm [ADAPT: the
        authors draw catalog pairs (526, 671-700)].  C: the round's K x q main-effect covariates or None."""
        if self.unit != 1.0:                                # GLM-ESTR in unit-norm units
            M, x = M / np.sqrt(self.unit), x / np.sqrt(self.unit)
        if self.q and self.spec["method"] == "ESTR":       # augmented arm (m, c), context (x, 1, 0...)
            M = np.hstack([M, C])
            x = np.r_[x, 1.0, np.zeros(self.q - 1)]
        Z = rotated_features(M, x, self.U, self.V)
        if self.q and self.spec["method"] == "OFUL":
            Z = np.hstack([C, Z])
        self.explore = self.ridge is None                       # ESTR stage 1
        if self.explore or (self.spec["method"] == "OFUL" and self.n == 0):
            arm = exploration_arm
        else:
            arm = int(np.argmax(self.ridge.index(Z)))
        self.m, self.x, self.z = M[arm], x, Z[arm]
        return arm

    def update(self, y):
        if self.ridge is None:                                  # ESTR stage 1: store the observation
            self.obs_M[self.n], self.obs_x[self.n], self.obs_y[self.n] = self.m, self.x, y
            if self.n + 1 == self.T1:
                self._start_stage2()
        else:
            self.ridge.update(self.z, y)
        self.n += 1

    def _start_stage2(self):
        """Lines 705-814: recover Theta, rotate by its SVD, set up LowOFUL, replay stage 1 [ADAPT: the recovery runs on
        the observed (m, x, y) rows, not on averaged entries of a catalog matrix]."""
        sp, d, r, T, s, a = self.spec, self.d, self.r, self.horizon, self.s, self.a
        Theta, self.bm = recover(self.obs_M, self.obs_x, self.obs_y, r, self.R, self.seed, self.family, self.nb, s, a)
        self.U, _, Vt = np.linalg.svd(Theta)
        self.V = Vt.T
        T2 = T - self.T1
        lam = s ** 2 * LAMBDA
        lamp = s ** 2 * (T2 / d / r / np.log(1 + T2 / LAMBDA))             # 765
        S_F, S2 = sp["S_F"], sp["S2"]
        if self.glm_estr:                     # the logistic curvature bound in lambda_perp; unit-norm units
            L = curvature_bound(self.family, self.nb)
            lamp = L * T2 / d / r / np.log(1 + L * T2 / LAMBDA)
            S_F, S2 = self.unit * S_F, self.unit * S2
        Sp = np.sqrt(lam) * S_F + S2 * np.sqrt(lamp) * (self.R / a) ** 2 * d ** 3 * r / self.T1   # 'simple2'
        D = np.full((d, d), lam)                                           # 790-795
        D[r:, r:] = lamp
        self.ridge = Ridge(D.ravel(), self.R, Sp, sp["multiplier"], self.family, self.nb, T, s)
        for i in range(self.T1):                                           # 804-805: replay stage 1
            z = rotated_features(self.obs_M[i][None, :], self.obs_x[i], self.U, self.V)[0]
            self.ridge.update(z, self.obs_y[i])


def run_baseline(env, spec, solver_seed):
    """One configuration on one environment: per-round arm, reward, regret, exploration and the cumulative regret
    (with ESTR, the stage-1 record).  coding "pm1": the linear arithmetic updated with 2y - 1."""
    n = env.n
    pm1 = spec.get("coding") == "pm1"
    policy = Baseline(spec, n, "gaussian" if pm1 else env.family, env.nb, solver_seed)
    trace = dict(arm=np.zeros(n, int), reward=np.zeros(n), regret=np.zeros(n), exploration=np.zeros(n, bool))
    main = getattr(env, "main", None)
    for t in range(n):
        means = env.means(t)
        a = policy.select(t, env.candidates[t], env.contexts[t], env.exploration_arms[t],
                          None if main is None else main[t])
        y = env.reward(t, a, means)
        trace["arm"][t], trace["reward"][t], trace["regret"][t] = a, y, means.max() - means[a]
        trace["exploration"][t] = policy.explore
        policy.update(2.0 * y - 1.0 if pm1 else y)
    trace["cumulative_regret"] = np.cumsum(trace["regret"])
    if spec["method"] == "ESTR":
        trace["bm"] = dict(getattr(policy, "bm", {}))
        truth = env.link_truth() if hasattr(env, "link_truth") else None
        if truth is not None and not policy.q and policy.ridge is not None:
            trace["bm"]["bm_sin"] = stage1_subspace_error(policy.U, policy.V, policy.r, truth)
    return trace


def stage1_subspace_error(U, V, r, truth):
    """Offline diagnostic of ESTR's stage 1 (uses the truth): max over the two sides of ||(I - U_r U_r') U*||_2, U*
    the truth's singular vectors with singular value at least half the largest (0: contained; 1: a direction missed)."""
    Us, sv, Vst = np.linalg.svd(truth)
    k = int(np.sum(sv >= 0.5 * sv[0]))
    left = np.linalg.norm(Us[:, :k] - U[:, :r] @ (U[:, :r].T @ Us[:, :k]), 2)
    right = np.linalg.norm(Vst[:k].T - V[:, :r] @ (V[:, :r].T @ Vst[:k].T), 2)
    return float(max(left, right))
