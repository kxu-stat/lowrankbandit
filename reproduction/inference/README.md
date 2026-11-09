# inference/ -- the inference simulations

Algorithm 2 on data collected by the two-track policy: a shared exploration, then two greedy tracks that take the
rounds in turn.  Every choice of the split n0 runs on the same collections and is judged by the same standard.

## Files

| file | what it holds |
|---|---|
| `algorithms.py` | Algorithm 2: Step 1 `collect()` (exploration and the two tracks, with the tuning rules of `../regret/`), Steps 2-4 `two_track_inference()` (nuisance fit, truncation, predictable refit, calibration on the other track, aggregation), `calibrate_loadings()` (Steps 3-4 for many loadings at once), the rules for n0, the rank chosen by an information criterion (`rank_rule`), and the one-step comparator |
| `simulation.py` | the blocks (grids of n, replications, seeds), the designs, the truth and the four targets, the choices of n0 (`N0_CHOICES`) |
| `run_inference.py` | one array task: a range of replicates of one block, one table part; each collection is also recorded (`records/`) |
| `replay_inference.py` | regenerates a recorded collection from its seed without refitting |
| `collect_inference.py` | completeness, summaries for every (block, choice of n0, n), the choices compared, the verification standard |
| `figures_inference.py` | the figures (its docstring lists them) |
| `check_inference.py` | a health check of a collected run (judges nothing) |
| `tie_ablation.py`, `ablation_step3.py`, `figure_ablation.py` | the ablation of Step 3: Algorithm 2 against the one-step debiasing of the policy's own estimator, whole sample and cross-fitted, on the same collections |
| `submit_*.sge`, `notify.sge` | the cluster jobs, the collect among them (`../RUNNING.md`) |

## Design

The settings are defined in `simulation.py` and the constants of Algorithm 2 in `algorithms.py` (`C_REF = 1/15`,
`C_CAL = 1`).  The environment is the regret cell d1 = d2 = 16, r = 2, K = 16, interaction only, covariates on the
sphere or N(0, I):

| block | rewards | rounds n |
|---|---|---|
| I1 | Gaussian | 500 to 10,000 |
| I2l | binary (logistic) | 1,000 to 60,000 |
| I1d32 | Gaussian, d1 = d2 = 32 | 1,000 to 10,000 |
| I3 | the scalar design with frequent ties (appendix) | 50 to 12,800 after the exploration |

1,000 replicates per block; one run per replicate to the block's largest n, evaluated at every n.  The collection uses
the regret study's exploration length (c_tau = 1/20) and penalty (cross-validated in each track); I3 has its own.
The choices of n0: the design-stage split with cap 3/4 (`design_75`, displayed) and 0.9, the information rule
(`info_rule`), the fixed shares 0.1, 0.3, 0.5, 0.7, 0.9 of a track, and no refit (`no_refit`, for comparison); the
appendix adds the rank chosen in each fold (`rank_gic`, `rank_aic`).  Four targets on every collection: a dense
loading, the two entries with the largest and smallest tangent part, and a rank-one contrast.

## Output

`inference_results/parts/` (gzip-compressed): one row per (replicate, n, choice of n0, target) with its seed, mu, the
standard errors, the true value and the exact decomposition of mu - mu*; `inference_results/records/`: every
collection; `inference_results/collected/`: the merged tables (gzip-compressed) and the summaries, among them the
verification standard for every choice.  T is standardized by the oracle standard error (the predictable quadratic
variation with the true variance).  `figures/ARM/`: the figures of one choice of n0 (default `design_75`);
`inference_trend` and `inference_qq` are the main text's.  The ablation writes `ablation_results/tie`
(`ablation_tie_hist`, the main text's) and `ablation_results/appendix`.
