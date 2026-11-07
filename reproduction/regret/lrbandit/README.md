# lrbandit (vendored extract)

Source: Kwang-Sung Jun, Rebecca Willett, Stephen Wright and Robert Nowak,
"Bilinear Bandits with Low-rank Structure", ICML 2019.
Code: https://github.com/kwangsungjun/lrbandit, commit
2f1f7ca4bbefe2bfd3e0bc50c4423a9791bfcde8 (2019-05-12), Apache License 2.0
(LICENSE in this directory is the authors' file, unchanged).

Files taken:

* `matrixrecovery.py` -- `rankone`, the Burer-Monteiro matrix-recovery stage
  of ESTR (`matrixrecovery/matrixrecovery.py`).  Three marked modifications:
  the Cython helper `calcRowwiseKron` is replaced by the equivalent numpy
  expression, an optional `init_U` argument replaces the internal
  `numpy.random.randn` draw so that starts can be seeded from outside, and
  the `tqdm` progress bar is removed.
* `calc_sqrt_beta_det4` (`blbandits3.py` line 37): the confidence radius of
  LowOFUL, copied verbatim into `matrixrecovery.py` for convenience.

`../baselines.py` calls `rankone` with the controls of the authors' `BilinearTwoStage` (`blbandits3.py` line 737:
the defaults C = 0.1, tolPred = 0.01, tolTh = 0.01, maxIter = 400), tolTh converted to the data's units, from 50
random starts (the authors start once; with new arms every round one start does not find the minimizer at short T1);
GLM-ESTR (our extension) has its own routine and controls.

The bandit classes themselves (`BilinearOful`, `BilinearTwoStage`) are
written for a fixed catalog of left and right arms.  Our experiment draws K
new left arms and one context each round, so `../baselines.py` re-implements
their per-round update with the authors' arithmetic, citing the source
lines; each departure forced by the contextual interface is marked [ADAPT].
