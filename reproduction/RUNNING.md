# Running the study

Every experiment is a grid of independent cells (a setting, a replication or stream, a method).  A run computes the
cells as an array job, one file per cell, then a collect gathers them into tables and the figure scripts draw from
those tables.  A finished cell is never computed again, so submitting an interrupted array again finishes it.

The job scripts `submit_*.sge` are for Sun Grid Engine.  Each holds a load line for Python, a `cd` into its folder and
one Python call, for example

    module load python/3.12.15
    cd $SGE_O_WORKDIR
    python3 run_regret.py gaussian --chunk $SGE_TASK_ID --chunks 1100

so on another cluster only the header (queue, module) changes, and without a cluster the same call runs for each
`--chunk` from 1 to the number of tasks.  `notify*.sge` runs after an array (`qsub -hold_jid`) and counts its cells;
add `#$ -M <address>` and `#$ -m e` to be e-mailed then.

## The runs

Run each command from the folder named in the first column.  Each full run is preceded by its short check run
(minutes; it checks the pipeline end to end and makes no statistical claim).

| folder | experiment | check run | full run | cells | CPU hours |
|---|---|---|---|---|---|
| `regret/` | Gaussian rewards | `submit_debug_gaussian.sge` | `submit_gaussian.sge` | 22,500 (1,100 tasks) | 470-590 |
| `regret/` | binary rewards | `submit_debug_binomial.sge` | `submit_binomial.sge` | 10,500 (1,100 tasks) | 1,950-2,200 |
| `regret/` | regret against the horizon | `submit_debug_scale.sge` | `submit_gaussian_scale.sge` | 9,800 (1,100 tasks) | 230-320 |
| `regret/` | approximately low-rank truths, Gaussian | `submit_debug_approx_d32.sge` | `submit_gaussian_approx.sge` | 7,200 (1,200 tasks) | 85-115 |
| `regret/` | approximately low-rank truths, binary | `submit_debug_approx_binomial.sge` | `submit_binomial_approx.sge` | 1,400 | about 600 |
| `regret/` | d = 32 | (with the Gaussian approx) | `submit_gaussian_d32.sge` | 1,400 | 280-420 |
| `inference/` | inference simulations | `submit_debug.sge` | `submit_inference.sge` | 611 tasks | 1,300-1,500 |
| `realdata/` | MIND regret, down-sampled world | `submit_debug_mind.sge` | `submit_mind.sge` | 1,150 | 4,500-5,400 |
| `realdata/` | MIND regret, main world | `submit_debug_mind_main.sge` | `submit_mind_main.sge` | 250 | 1,250-1,350 |
| `realdata/` | MIND inference | `submit_debug_mind_inference.sge` | `submit_mind_inference_eval.sge` | 1,000 tasks | about 1,400 |
| `inference/` | ablation of Step 3, main text | by hand (below) | `submit_tie_ablation.sge` | 10 tasks | under 1 |
| `inference/` | ablation of Step 3, appendix | by hand (below) | `submit_ablation.sge` | 20 tasks | 8-10 |

The CPU hours are those of the authors' cluster, about 12,000-13,500 in all.  The runs are independent except:
the MIND main world runs the competitors' configurations that were best in the down-sampled world, so it follows
that run's collect; the appendix ablation replays the records of the inference simulations; the inference collect
compares with the regret runs' single track if they are there.

## Commands

The header of each `submit_*.sge` names the collect that follows it.  The full runs:

    # regret/
    qsub submit_gaussian.sge;       python3 collect_regret.py gaussian
    qsub submit_binomial.sge;       python3 collect_regret.py binomial
    qsub submit_gaussian_scale.sge; python3 collect_regret.py gaussian --tiers scale --out regret_gaussian_scale
    qsub submit_gaussian_approx.sge; python3 collect_regret.py gaussian --tiers approx --out regret_gaussian_approx
    qsub submit_binomial_approx.sge; python3 collect_regret.py binomial --tiers approx --out regret_binomial_approx
    qsub submit_gaussian_d32.sge;   python3 collect_regret.py gaussian --tiers d32 --out regret_gaussian_d32
    python3 verdict_regret.py;      python3 figures_regret.py
    python3 figures_regret.py regret_gaussian_scale
    python3 figures_regret.py regret_gaussian_approx regret_binomial_approx regret_gaussian_d32

    # inference/
    qsub submit_inference.sge
    python3 collect_inference.py --regret-curves ../regret/regret_gaussian/collected/regret_curves.tsv,../regret/regret_binomial/collected/regret_curves.tsv
    python3 figures_inference.py

    # realdata/ (the data files first: realdata/README.md)
    qsub submit_mind.sge;           python3 collect_mind.py mind_regret --setup data/mind_setup_D16.npz
    qsub submit_mind_main.sge
    python3 collect_mind.py mind_regret_main --setup data/mind_setup_D16.npz --block main --best mind_regret/collected/best.tsv
    python3 figures_mind.py mind_regret;  python3 figures_mind.py mind_regret_main
    qsub submit_mind_inference_eval.sge
    python3 collect_mind_inference.py evaluation --setup data/mind_setup_D16.npz
    python3 oracle_mind_inference.py evaluation --setup data/mind_setup_D16.npz --world down4
    python3 oracle_mind_inference.py evaluation --setup data/mind_setup_D16.npz --world main

    # inference/, the ablation of Step 3: its check run by hand (about a minute, after the check run of the inference)
    python3 tie_ablation.py 200 1 2 ablation_debug/tie
    python3 ablation_step3.py inference_debug ablation_debug/parts --every-n
    python3 figure_ablation.py ablation_debug/parts --run inference_debug --name debug
    # then the two runs (the appendix part after submit_inference.sge)
    qsub submit_tie_ablation.sge;   python3 figure_ablation.py ablation_results/tie --blocks "" --name ablation_tie
    qsub submit_ablation.sge
    python3 figure_ablation.py ablation_results/parts --run inference_results --name ablation_appendix \
        --show I2l:dense:40000,I2l:entry_max:40000,I2l:rank1:40000,I2l:entry_min:40000,I2l:dense:10000,I1:dense:10000,I1d32:dense:10000

Each collect reports the cells it found, failed and missing; a failed cell keeps its traceback.  After a collect,
`check_*.py RUN_DIR` prints the run's health check.  The check runs collect with the options in their script's
header, and `python3 ../selftest.py --cells RUN_DIR` compares the regret check runs' cells with the authors' values.

## Recomputing one cell

Every cell is a function of its key (its file name), so it can be recomputed alone, into another directory:

    python3 run_regret.py gaussian --out replay --only sph_d16_r2_K16_rho020_b601_A_t2 --trace      # regret/
    python3 run_inference.py --only I1_lowrank_r0001-0100 --out replay                             # inference/
    python3 run_mind.py --setup data/mind_setup_D16.npz --out replay --only D16_K10_down4_b001_A_t2  # realdata/

and `inference/replay_inference.py` regenerates a recorded inference collection from its seed without refitting, so
Steps 2-4 can be run again on the same data under another choice.
