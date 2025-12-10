# GNN-Enhanced Stock Portfolio Optimization

Lean, results-focused version of the project that adds a minimal Graph Neural Network (GNN) on top of the parent paper’s SHLO and Hill Climbing algorithms. The codebase is trimmed to only what is needed to reproduce the new GNN results and graphs.

## What’s New vs. Parent Paper
- Added a lightweight NumPy-based GNN to learn stock-to-stock relationships (category + feature similarity).
- Plugged GNN-enhanced scores into both SHLO and Hill Climbing objective functions.
- Produced side-by-side baselines vs. GNN runs and publication-ready visualizations.
- Removed legacy notebooks, split utilities, and unused datasets to keep the repo focused on the new results.

## Repo Contents
- `gnn_portfolio_optimization.py` — runs four experiments: baseline SHLO, GNN-SHLO, baseline HC, GNN-HC.
- `Final_Input_dataset_for_DSS.csv` — stock universe with fundamentals, intrinsic values, and fitness metrics.
- `GNN_EXPERIMENT_RESULTS.md` — detailed tables, portfolios, and conclusions from the GNN runs.
- `generate_graphs.py` — regenerates comparison plots (non-GUI backend).
- `new_results/` — generated PNG charts.
- `LICENSE`, `README.md`.

## Key Experiment Settings
- Portfolio size: 10 (Large 5, Mid 2, Small 3)
- Budget: $10,000; per-stock bounds: 5%–20%
- Objective weights: fitness 0.50, percent-change-to-intrinsic 0.20, revenue growth 0.25, budget use 0.05
- GNN: 2 layers (16 hidden, 8 output), symmetric-normalized adjacency, cosine + category edges, seed 42
- Enhanced score = 0.8 × original metric + 0.2 × GNN embedding (per metric)

## Results (from latest run)
- SHLO: 5.5655 → **6.3696** with GNN (+14.45%)
- Hill Climbing: 7.0068 → **7.5926** with GNN (+8.36%)
- Budget use: 57–69% (baselines) vs. 63–65% (GNN variants)
- Best performer: GNN-HC (objective 7.5926)
See `GNN_EXPERIMENT_RESULTS.md` for full tables and portfolios.

## How to Run
Install deps (Python 3.9+):
```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

Run the four experiments (prints results to console):
```
python gnn_portfolio_optimization.py
```

Regenerate charts:
```
python generate_graphs.py
```
Outputs are saved under `new_results/`.

## Change Log (cleaned branch)
- Added: minimal GNN feature builder + message passing; enhanced objective integration for SHLO/HC.
- Added: `generate_graphs.py` and `new_results/` plots; `GNN_EXPERIMENT_RESULTS.md`.
- Removed: legacy notebooks, dataset split utilities, old intrinsic/fitness CSVs, archives, and unused docs to keep only what supports the new GNN results.
