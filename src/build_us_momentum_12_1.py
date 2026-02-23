"""
Reads the monthly returns for US stocks, calculates the 12-1 momentum signal, and saves it to data/processed.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    rets = pd.read_parquet(root / "data/processed/us_monthly_returns.parquet")

    mom = np.expm1(np.log1p(rets).rolling(11).sum()).shift(1)
    
    mom.to_parquet(root / "data/processed/us_momentum_12_1.parquet")
    print("Saved momentum signal.")


if __name__ == "__main__":
    main()