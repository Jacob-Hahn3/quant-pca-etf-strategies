# Rolling PCA ETF Strategies (2014–2025)

This project applies rolling PCA to a diversified ETF universe to:
- diagnose low-dimensional structure in returns (explained variance + factor stability),
- construct eigenportfolios (dollar-neutral factor portfolios),
- build minimum-variance and factor-neutral minimum-variance portfolios,
- stabilize optimization using Ledoit–Wolf covariance shrinkage and a gross-leverage constraint,
- evaluate performance net of transaction costs in a walk-forward monthly backtest.

## How to reproduce
1. Create a virtual environment and install dependencies:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`

2. Download data and build the cleaned return panel:
   - `python src/download_etf_data.py`

3. Run analysis notebooks in order (see `notebooks/`).

## Repo structure
- `src/` scripts (data download, PCA, backtest, optimization)
- `notebooks/` research notebooks
- `results/` summary CSV outputs (robustness grid, EVR/stability series)
- `figures/` saved plots used in the writeup
- `paper/` LaTeX + bibliography (optional)
