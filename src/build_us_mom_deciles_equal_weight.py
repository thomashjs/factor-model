"""
Runs the Fama-Macbeth regression for the 12-1 momentum signal on US stocks, and builds the cumulative returns for the W-L portfolio.

Reads: 
- data/processed/us_monthly_returns.parquet
- data/processed/us_momentum_12_1.parquet

Writes: 
- figures/us_mom_cumulative.png
- reports/us_fmb_summary.csv
"""

from pathlib import Path
import pandas as pd
from cross_sectional_utils import (
    run_fama_macbeth,
    build_deciles,
    cumulative_returns
)
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]

    rets = pd.read_parquet(root / "data/processed/us_monthly_returns.parquet")
    mom = pd.read_parquet(root / "data/processed/us_momentum_12_1.parquet")

    gamma_mean, t_stat, gammas = run_fama_macbeth(mom, rets)
    wml = build_deciles(mom, rets)
    cumulative = cumulative_returns(wml)

    (root / "reports").mkdir(exist_ok=True)

    print("Gamma:", gamma_mean)
    print("t-stat:", t_stat)
    print("Approx cumulative W-L:", float(cumulative.iloc[-1] - 1))

    # Summary
    pd.DataFrame({
        "gamma_mean": [gamma_mean],
        "t_stat": [t_stat],
        "n_months": [len(gammas)]
    }).to_csv(root / "reports/us_mom_fmb_summary.csv", index=False)

    # Full gamma series
    gammas.to_csv(root / "reports/us_mom_fmb_gamma_series.csv", header=["gamma"])
    print("Saved FMB summary and gamma series.")

    plt.figure()
    cumulative.plot()
    plt.savefig(root / "figures/us_mom_cumulative.png")

if __name__ == "__main__":
    main()