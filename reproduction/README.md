# Reproducing the numerical results of the paper

This folder holds the code of the numerical study in

> *Regret Minimization and Statistical Inference for Low-Rank Contextual Generalized Linear Bandits, with Applications
> to Advertising and Product Design.*

It is the code that produced the paper's tables and figures, as it was run (its version is in `VERSION`).  It is
separate from the package `lowrankbandit` at the top of this repository: the package implements the two algorithms
for use on new data; this folder runs them, with the competing methods, on the paper's simulation designs and on the
MIND news data, and summarizes the results.  `check_package.py` confirms that the package and this code choose the
same arms and give the same estimates and standard errors on the same rounds.

## What is where

| the paper | folder | what runs |
|---|---|---|
| Numerical Experiments: Regret Comparison; Appendix: Regret: Design, Tuning, and Complete Comparisons | `regret/` | Algorithm 1 against OFUL (Abbasi-Yadkori, Pál and Szepesvári, 2011) and ESTR (Jun, Willett, Wright and Nowak, 2019), Gaussian and binary rewards |
| Numerical Experiments: Confidence Intervals and Normal Approximation; Appendix: Inference: Implementation and Diagnostics, A Scalar Design with Frequent Ties | `inference/` | Algorithm 2 on data collected by the two-track policy, and the ablation of its Step 3 |
| News Recommendation with MIND; Appendix: Implementation Details for the MIND Study | `realdata/` | both studies on MIND-small, semi-synthetic: clicks drawn from a logistic model fitted to the data |

Each folder's README lists its files, where its settings are defined, what a run writes and which figures it draws.
`RUNNING.md` gives every run in order, with its size.

The three folders are organized the same way:

| file | what it does |
|---|---|
| `algorithms.py` | the methods: Algorithm 1 (`regret/`), Algorithm 2 (`inference/`); numpy and scipy only |
| `simulation.py` (`realdata/mind.py`) | the designs: settings, seeds, environments |
| `run_*.py` | one array task: a set of cells (setting x replication x method), one file per cell |
| `collect_*.py`, `verdict_*.py` | the tables, and the comparisons stated in advance |
| `figures_*.py` | the figures |
| `check_*.py` | a health check of a collected run (judges nothing) |
| `submit_*.sge` | the cluster jobs (Sun Grid Engine) |

## Installation and self-test

Python 3.11 or later:

    python3 -m pip install -r requirements.txt     # numpy, scipy, matplotlib
    python3 selftest.py                            # must end with ALL CHECKS PASS

The self-test recomputes, along every numerical path, numbers recorded on the authors' installation
(`selftest_reference.json`); it takes under a minute.  This matters for bandits: a policy chooses by argmax, so a
difference in the last bits of a linear-algebra routine can flip one choice and send a trajectory elsewhere.  Every
cell records the software versions it was computed with.

## Data

The simulations need no data.  The MIND study needs MIND-small and the GloVe word vectors, downloaded from their
sites; `realdata/README.md` describes how its two input files are built from them.

## Third-party code

`regret/lrbandit/` holds two functions from the code of Jun, Willett, Wright and Nowak (2019), under the Apache
License 2.0; `NOTICE.md` lists them and the changes made.
