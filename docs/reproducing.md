# Reproducing the paper

The code of the paper's numerical study is in the repository's
[`reproduction/`](https://github.com/kxu-stat/lowrankbandit/tree/main/reproduction) folder, separate from the
package. It runs Algorithms 1 and 2 with the competing methods on the paper's simulation designs and on the MIND
news-recommendation data, and draws every figure and table.

| Part of the paper | Folder |
|---|---|
| Regret comparison (simulations) | [`reproduction/regret/`](https://github.com/kxu-stat/lowrankbandit/tree/main/reproduction/regret) |
| Confidence intervals and normal approximation (simulations) | [`reproduction/inference/`](https://github.com/kxu-stat/lowrankbandit/tree/main/reproduction/inference) |
| News recommendation with MIND | [`reproduction/realdata/`](https://github.com/kxu-stat/lowrankbandit/tree/main/reproduction/realdata) |

The competing methods are OFUL (Abbasi-Yadkori, Pál and Szepesvári, 2011) and ESTR (Jun, Willett, Wright and
Nowak, 2019), run on their authors' grids of tuning constants; two functions are taken from the ESTR authors' code.

## In short

```bash
cd reproduction
python3 -m pip install -r requirements.txt     # numpy, scipy, matplotlib
python3 selftest.py                            # checks that this installation reproduces the authors' numbers
```

Every experiment is a grid of independent cells, each a function of its setting and replication number, run as an
array job on a cluster and then collected into tables and figures. Any single cell can be recomputed on a laptop.
The folder's [README](https://github.com/kxu-stat/lowrankbandit/blob/main/reproduction/README.md) and
[RUNNING.md](https://github.com/kxu-stat/lowrankbandit/blob/main/reproduction/RUNNING.md) give every command, the
size of every run and the preparation of the MIND data.

The package and the experiment code compute the same thing: `reproduction/check_package.py` runs both on the same
rounds and compares the arms chosen in every round and the resulting intervals.
