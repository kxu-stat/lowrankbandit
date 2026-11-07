# regret/ -- the regret simulations

Algorithm 1 against OFUL (Abbasi-Yadkori, Pál and Szepesvári, 2011) and ESTR (Jun, Willett, Wright and Nowak,
2019), on Gaussian and binary rewards.  Our method uses the theory's rates with constants fixed in advance or chosen
inside the run; each competitor runs its authors' whole grid of configurations, and every configuration is reported.

## Files

| file | what it holds |
|---|---|
| `algorithms.py` | Algorithm 1 and its estimator (FISTA under the identity link, proximal Newton under the logit link); the exploration length tau = ceil(c_tau (d1 + d2) log(n d1 d2)) (`tau_rule`); the penalty lambda_s = c sigma_hat sqrt((d1 + d2) log(n d1 d2) / s), with c chosen inside the run by 10-fold cross-validation on the run's own observations at s_1 = max(2 tau, 200), 2 s_1, 4 s_1, ... over 1, 1/2, ..., 1/64 (`cv_constant`).  Imports nothing of the experiment. |
| `baselines.py` | OFUL and ESTR on this contextual interface, on the authors' released grids (OFUL: 5 multipliers; ESTR: 5 values of T1, 5 multipliers each); binary rewards: our logistic versions (logistic OFUL, GLM-ESTR) and the authors' own linear model on 2y - 1.  Departures forced by the interface are marked `[ADAPT]`. |
| `lrbandit/` | two functions from the authors' code (Apache-2.0; see `../NOTICE.md`) |
| `simulation.py` | the design: the grid of cases, seeds, environments (truth, covariates, arms, noise), rewards; the arms of the study (`ARMS`, `C_TAU`, `C_TAU_RUN`); the further tiers (`TIER_TRUTH`, `TIER_LEVELS`, `TIER_HORIZONS`) |
| `run_regret.py` | one array task: a set of (case, replication, arm) cells, one JSON file each |
| `collect_regret.py` | completeness; every configuration's mean regret (`summary.tsv`); each competitor's best configuration in hindsight (`best.tsv`); the paired comparisons; the regret curves |
| `verdict_regret.py` | the comparisons stated in advance (`collect_regret.PREDICTIONS`), at declared thresholds |
| `figures_regret.py` | the figures (its docstring lists them) |
| `check_regret.py` | a health check of a collected run (judges nothing) |
| `submit_*.sge`, `notify.sge` | the cluster jobs (`../RUNNING.md`) |

## Design

The settings are defined in `simulation.py`.  Horizon n = 10,000; d1 = d2 = 16 with rank r = 1, 2, 4 at K = 16
candidate arms per round, and r = 2 at K = 32, 64.  Two covariate designs with E[m m'] = E[x x'] = I: uniform on the
sphere of radius sqrt(d) (bounded: every method's assumptions hold) and N(0, I) (unbounded).  Gaussian noise at three
levels relative to the signal, or binary rewards (the greedy arm's mean success probability 0.8).  50 replications
per case.  For each case and replication:

| arm | what |
|---|---|
| `A_t2` | our method, c_tau = 1/20, c by cross-validation (the reported configuration) |
| `A_t1`, `A_t3` | the same at c_tau = 1/40 and 1/10 (sensitivity) |
| `M_k0` ... `M_k4` | the matched arm: our learner with tau = ESTR's k-th exploration length T1 |
| `oful` | OFUL over its 5 multipliers (binary rewards: logistic OFUL) |
| `estr_k0` ... `estr_k4` | ESTR at its k-th T1, 5 multipliers each (binary rewards: GLM-ESTR) |
| `estr_tau` | ESTR with our exploration length, 5 multipliers |
| `oful_pm1`, `estr_pm1_k0` ... `k4` | binary rewards only: the authors' 2y - 1 versions |

Further tiers, each its own run: the regret against the horizon (n = 1,000 to 100,000, every run tuned to its own n),
approximately low-rank truths (singular values decaying like j^-2 and j^-1.5), and d = 32.

## Output

`regret_FAMILY/cells/KEY.json`, one per cell: the task, one summary row per configuration (regret, the exploration
length, the penalty, our fitted rank and estimation error, the cross-validation choices), the cumulative-regret curves,
the software versions; with `--trace` the per-round trajectory.  `regret_FAMILY/collected/`: the tables.
`figures/`: the figures, PDF and PNG; `regret_curves_main_identity` is the main text's.
