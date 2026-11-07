# Settings

Every tuning rule has the form given by the theory, with a fixed constant or one chosen by cross-validation inside
the run, so the defaults need no tuning by the user. The arguments are the same in Python and R.

## Data collection (`GreedyPolicy`, `TwoTrackPolicy`)

| Setting | Default | Argument |
|---|---|---|
| reward model | Gaussian | `family` (`"gaussian"`, `"binomial"`), `trials` |
| exploration length | \(\tau = \lceil c_\tau (d_1 + d_2 + p_0) \log(n d_1 d_2) \rceil\), \(c_\tau = 1/20\) | `tau`, `c_tau` |
| penalty level | \(\lambda_s = c\, \hat\sigma \sqrt{(d_1 + d_2)\log(n d_1 d_2)/s}\) | |
| penalty constant \(c\) | 10-fold cross-validation over \(\{1, 1/2, \ldots, 1/64\}\) at \(s = \max(2\tau, 200), 2s, 4s, \ldots\); 1/8 before the first | `c_lam` (a number fixes it) |
| main effects | not included | `main_effects`, `n_covariates` |
| domain of the estimator | none (Gaussian); each coefficient block in a ball of radius 10 (binomial) | `radius` |
| exploration arms | uniform at random | `explore_arms`, `seed` |

Here \(n\) is the horizon, \(p_0\) the number of main-effect coefficients and \(s\) the number of rounds fitted. The
scale \(\hat\sigma = \|G_\tau\|_{\rm op} \sqrt{\tau/(d_1 + d_2)}\) comes from the exploration rounds, \(G_\tau\) being
the interaction block of the score of the likelihood at \(\Theta = 0\), with the main effects (if any) fitted.

Cross-validation assigns round \(i\) to fold \(i \bmod 10\), scores each fold by squared error (Gaussian) or deviance
(binomial), and uses only data the policy has already seen, so it can run inside a live system. Under the binomial
family, the bounded domain keeps the fit finite when the exploration rounds happen to be separable.

## Inference (`two_track_inference`)

| Setting | Default | Argument |
|---|---|---|
| rank \(r_0\) of \(\Theta\) | required | `rank` (a number, or `"gic"` to choose it in each fold) |
| split \(n_0\) | `"equal"` | `n0`, `kappa_D` |
| refit and calibration constants | \(C_{\rm ref} = 1/15\), \(C_{\rm cal} = 1\) | `C_ref`, `C_cal` |
| confidence level | 0.95 | `level` |

## Choosing `n0`

Each track's \(n_{\rm tr}\) rounds are split: the first \(n_0\) give the preliminary fit of Step 2, the remaining
\(n_{\rm tr} - n_0\) the refit. Four choices:

- `"equal"` (default): \(n_0 = \lfloor n_{\rm tr}/2 \rfloor\).
- `"information"`: \(n_0 = \lceil p_T \log(n d_1 d_2) / (3 \hat\kappa_0) \rceil\), kept within
  \([\max(n_{\rm tr}/4, 10),\, 3n_{\rm tr}/4]\). Here \(\hat\kappa_0\), the information per round and coordinate, is
  estimated from the exploration, and \(p_T = p_0 + r_0(d_1 + d_2 - r_0)\) is the dimension of the tangent space.
- `"design"`: the equal split, lengthened to \(p_T \log(n d_1 d_2) / (3 \kappa_D)\) when half a track is shorter, at
  most \(3n_{\rm tr}/4\), for a planning value `kappa_D` of the information per round and coordinate (for example 1
  for Gaussian rewards with \(E[zz^\top] = I\)).
- a number of rounds between 2 and \(n_{\rm tr} - 2\).

With `rank="gic"`, \(p_T\) is not known in advance, so `n0` must be `"equal"` or a number.

## Python and R

The two packages have the same defaults and, given the same rounds and the same exploration arms, return the same
numbers. The tests check this on six shared cases (Gaussian and binomial rewards, with and without main effects,
square and rectangular \(\Theta\), fixed and cross-validated penalty), comparing the arms chosen in every round, the
cross-validation choices, the estimates and the standard errors.

Two differences in use:

- Arms are numbered from 0 in Python and from 1 in R.
- Random numbers. In Python, the policies and `simulate_stream` take a `seed` and `low_rank_truth` an `rng`, each
  feeding its own NumPy generator. In R these functions draw from R's generator, which `set.seed` fixes (the
  policies' `seed` argument calls `set.seed`). The same seed therefore gives different streams and exploration arms
  in the two languages; pass the same stream and `explore_arms` to compare them.
