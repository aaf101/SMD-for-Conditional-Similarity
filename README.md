# Approximate minimax regret for conditionally similar tests

Replication materials for **Andrés Aradillas Fernández, “Approximate minimax regret for conditionally similar tests.”**

Repository: https://github.com/aaf101/SMD-for-Conditional-Similarity

The release tagged **v1.1.0** contains the manuscript, its LaTeX source, the Python implementation, and the simulation results used in the paper. All data in the numerical illustration are simulated; no external data, credentials, or proprietary software are needed for the computations.

## Reproduce from a clean checkout

The numerical example was run with Python 3.12. Dependency versions are pinned in `requirements.txt`. Install Python 3.12 and Git, then run:

```sh
git clone https://github.com/aaf101/SMD-for-Conditional-Similarity.git
cd SMD-for-Conditional-Similarity
git checkout v1.1.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python example.py --epsilon 0.02 --delta 0.05 --draws 512 --seed 20261102
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The program writes all outputs to `results/` relative to its own location and prints the summary and table. It runs on a CPU; no GPU is needed. Different library/platform implementations can cause small floating-point differences.

To compile the paper, install a current TeX Live or MacTeX distribution with `latexmk`, BibTeX, and the packages listed in `main.tex`. The supplied PDF was compiled with TeX Live 2026. Run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The compiled paper is `build/main.pdf`. A checked copy is included at `main.pdf`. `highlights.txt` contains the separate editable submission highlights.

## Experiment and numerical method

The example uses size 0.05, null coefficient zero, identity covariance, and alternatives `(beta, pi) = (-3, 0.5)` and `(1, 2)`. The user specifies an excess-regret tolerance `epsilon`, failure probability `delta`, and number of fresh independent draws per alternative per iteration `draws`. The program computes the iteration count and step size from the iteration-budget formula in Proposition 1. At the supplied settings, `epsilon=0.02`, `delta=0.05`, `draws=512`, and seed `20261102`, it selects `K=4251` and `eta=0.0180585`.

For each conditioning value, 44 bisection steps equate the mixture likelihood ratios at the two critical values, with the left-tail null probability restricted to `[1e-14, 0.05 - 1e-14]`. Left and right null tail probabilities sum to 0.05 by construction. The program evaluates every iteration's power separately using 320-node Gauss–Hermite quadrature and compares it with 160-node quadrature. It averages the powers of the individual tests to evaluate the returned average test. It does not average critical values or use the best response to average weights.

The script checks conditional size, quadrature stability, the pointwise power envelopes, and that the selected iteration count attains the requested theoretical tolerance bound. The proposition assumes exact conditional best responses; it does not bound numerical error in their computation. Results describe one training run and the two specified alternatives; they do not estimate variability across runs or performance over an unrestricted alternative space. The dual lower bound is evaluated numerically and does not incorporate a rigorous bound on critical-value or integration error.

## Outputs and expected results

| Output | Description |
| --- | --- |
| `results/figure_1_convergence.pdf` / `.png` | Average-test maximum regret and the running numerical dual lower bound |
| `results/figure_2_tail_allocation.pdf` / `.png` | Conditional left-tail allocation for initial and final individual best responses |
| `results/example_table.tex` / `.csv` | Powers and maximum regret of benchmarks and SMD averages |
| `results/example_history.csv` | Weights, individual powers, average powers, maximum regret, and dual bound for every iteration |
| `results/example_summary.json` | Configuration, final results, and numerical checks |

The PDF figures are separate vector files suitable for manuscript submission. PNGs are 300-dpi previews. The LaTeX table is editable text.

For the supplied run, maximum regret is approximately **0.07755** after 4,251 iterations, compared with **0.08189** for the initial best response and **0.12275** for the equal-tail test. The numerical dual lower bound is **0.07691**, leaving a numerical gap of **0.00065**. The theoretical guarantee is that, with probability at least 0.95, maximum grid regret is at most `V_I + 0.02`, under exact best responses. Increasing quadrature order changes computed powers by about `7e-15`, and the conditional-size error is about `1.2e-16`.

## Repository contents and citation

- `main.tex`, `references.bib`, and `main.pdf`: current paper and compiled version.
- `original.tex`: the original manuscript preserved unchanged.
- `example.py` and `requirements.txt`: numerical implementation and dependencies.
- `results/`: generated figures, table, and underlying numerical results.
- `highlights.txt`: four submission highlights.
- `johd.sty` and `johd.bst`: legacy files retained from the original manuscript; the rewritten paper does not use them.

Please cite the author, paper title, repository URL, and version **v1.1.0** when referring to these replication materials. A machine-readable citation is provided in `CITATION.cff`.

OpenAI Codex assisted with the production of this code.
