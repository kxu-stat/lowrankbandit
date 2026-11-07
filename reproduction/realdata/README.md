# realdata/ -- the MIND study

Both studies on MIND-small news impressions (Wu et al., 2020), semi-synthetic: the rounds, articles and users are the
data's; the clicks are drawn from a rank-2 logistic model fitted to all the data.  Nothing is tuned on MIND: every
method uses the simulations' rules, and every constant a method is given comes from the fitted truth.

## Files

| file | what it holds |
|---|---|
| `mind.py` | loading the prepared data, the covariates, drawing rounds, the environment, the down-sampled world, the competitors' declared constants |
| `setup_mind.py` | run once: the kept users and their vectors, the covariates' standardization, the rank-2 truth and every constant derived from it, the 32 inference targets (the setup file) |
| `add_world_mind.py` | run once: adds the down-sampled world's constants to the setup file |
| `run_mind.py`, `collect_mind.py`, `figures_mind.py`, `check_mind_regret.py` | the regret study |
| `run_mind_inference.py`, `collect_mind_inference.py`, `oracle_mind_inference.py`, `check_mind_inference.py` | the inference study (the oracle script recomputes the correction martingale with the true parameter, as a check) |
| `submit_*.sge`, `notify_mind.sge` | the cluster jobs (`../RUNNING.md`) |

## Design

D = 16, K = 10 articles per round; the 27,087 users with at least 10 history articles; articles and users encoded by
the GloVe vectors of the titles, projected to D dimensions and whitened (E[m m'] = E[x x'] = I).  The model has
unpenalized main effects (an intercept, the same-category share, the log impression size, m and x; 35 coefficients)
and the low-rank interaction.  Two worlds: the main world, the fitted truth; the down-sampled world, the click model of
logs that keep every click and a quarter of the non-clicks (every log-odds plus log 4).

- Regret (`run_mind.py`): 50 streams of rounds drawn with replacement.  The down-sampled world (main text), 40,000
  rounds: our method and its sensitivity, the matched arm, logistic OFUL and GLM-ESTR over the authors' grids, the
  authors' 2y - 1 versions, two greedy comparators without the interaction, and the two-track collection of the
  inference.  The main world (appendix), 80,000 rounds: our method, the two-track collection, the two competitors at
  their best configuration in the down-sampled world, and the greedy comparator on the covariates c alone.
- Inference (`run_mind_inference.py`): 500 streams in each world, Algorithm 2 at n = 10,000 to 80,000 for every
  choice of n0 of the simulations, 32 targets (log odds ratios and their parts for the 8 largest categories).

## Data

Download MIND-small (https://msnews.github.io/; the training part) and the GloVe 6B vectors
(https://nlp.stanford.edu/projects/glove/, `glove.6B.zip`).  Put the folder `MINDsmall_train` (`news.tsv`,
`behaviors.tsv`, `entity_embedding.vec`) into `MIND.zip`.  Then, with pandas installed:

    python3 prepare_mind.py MIND.zip mind_small.npz
    python3 encode_glove.py MIND.zip glove.6B.zip mind_small.npz mind_small_glove300.npz
    python3 prepare_mind_rounds.py mind_small.npz mind_small_glove300.npz mind_small_prepared_glove.npz
    python3 setup_mind.py mind_small_prepared_glove.npz mind_small.npz mind_small_glove300.npz 16 mind_setup_main_D16.npz
    python3 add_world_mind.py mind_setup_main_D16.npz mind_small_prepared_glove.npz 0.25 mind_setup_D16.npz

and move `mind_small_prepared_glove.npz` and `mind_setup_D16.npz` to `data/`.  Each step also writes a text file
of checks.  The SHA-256 of the authors' files begin with `01abc1c7958c48ba` (`mind_small.npz`), `fccacba54d67027f`
(`mind_small_prepared_glove.npz`) and `298d2e6e3e9640cb` (`mind_setup_D16.npz`).

## Output

`mind_regret*/cells/` and `collected/` (the tables, `verdict_mind.md`); `figures_mind/`: the regret figures.
`mind_inference_evaluation/collected/WORLD/`: the tables and figures of each world, among them `paper_qq_news_n<n>`
for the paper, and `reported_n.md`.
