# Third-party code

`regret/lrbandit/` contains two functions extracted from

> Kwang-Sung Jun, Rebecca Willett, Stephen Wright, Robert Nowak,
> "Bilinear Bandits with Low-rank Structure", ICML 2019.
> https://github.com/kwangsungjun/lrbandit, commit
> 2f1f7ca4bbefe2bfd3e0bc50c4423a9791bfcde8 (2019-05-12)

used under the Apache License 2.0, whose text is `regret/lrbandit/LICENSE`.
The modifications are marked `[MODIFIED]` in `regret/lrbandit/matrixrecovery.py`
and listed in `regret/lrbandit/README.md`: the Cython helper `calcRowwiseKron` is
replaced by the equivalent numpy expression, an optional `init_U` argument
replaces the internal random draw so that starts can be seeded from outside, and
the `tqdm` progress bar is removed.

`regret/baselines.py` re-implements the authors' bandit classes on this
experiment's contextual interface, citing the source line numbers; every
departure the interface forces is marked `[ADAPT]`.

Everything else in this package is the authors' own work, under the license in
the repository's top-level `LICENSE` file.
