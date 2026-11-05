# Exact versus genetic search for capacitated indoor edge node placement

Code and raw results for the article "When does genetic search pay off in capacitated indoor edge node placement?" (R. Tsarov, submitted to the Journal of Edge Computing).

The scripts compare branch and bound (HiGHS through `scipy.optimize.milp`), a genetic algorithm and a greedy local search on the same capacitated maximal covering problem: a 10 x 10 grid of candidate sites, 200 users, a load limit of 40 users per node and four wall scenarios (64 instances).

## Requirements

Python 3.10 or later, NumPy, SciPy (1.17 or later, which bundles HiGHS), Matplotlib. See `requirements.txt`.

## Files

| Script | Output | Content |
|---|---|---|
| `model.py`, `run_exp.py` | | radio model, walls, user positions (seed 1000 + r) |
| `e37.py` | `e37.json` | series 1: B&B and GA with 6 000 evaluations, M = 3, 5, 8 |
| `e37b.py` | `e37b.json` | series 2: 20 m / 19 dB and 10 m / 10 dB, M = 4..7 |
| `e38lp.py` | `e38lp.json` | LP relaxation, 64 distinct instances |
| `e38ga.py` | `e38ga.json` | GA with 30 000 evaluations and independent seeds |
| `e39.py` | `e39.json` | B&B with continuous and binary assignment variables |
| `e40.py` | `e40.jsonl` | paired GA trajectories (150 / 750 generations), greedy local search, 5 000 random layouts |
| `e41.py` | `e41.json` | weak-duality check of the LP bound |
| `ana40.py` | | summary statistics of `e40.jsonl` |
| `layout.py`, `fig4.py` | `layout_0.json`, `fig/fig4.*` | layouts of figure 4 |
| `jec_figs.py`, `common.py` | `fig/fig1-3.*` | figures 1-3 |

Series 1 and 2 contain the instances 20 m / 19 dB, M = 5, realizations 0-2 twice; `e38lp.py` keeps one copy, which gives 64 distinct instances.

## Reproduction

Run the scripts from this folder in the order of the table, for example `python e37.py`. The result files are included, so the analysis and figure scripts can be run directly.

Times in `e37*.json`, `e38*.json` and `e39.json` were measured with SciPy 1.17 on a 2.1 GHz Intel Xeon virtual server; `e40.jsonl` and `e41.json` were produced with SciPy 1.18.1 (HiGHS 1.12.0) on a 2.8 GHz Intel Xeon virtual server. Times from the two environments are not comparable. Solver times vary between repeated runs by several seconds.

`e40.py` stops a GA trajectory once it reaches the known optimum; because of elitism this does not change the recorded values, but the stored times of such trajectories are partial.

## License

MIT, see `LICENSE`.
