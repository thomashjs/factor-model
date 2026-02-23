"""
Reads the daily price data for US stocks, resamples it to monthly frequency, and calculates the monthly returns. Saved to data/processed.
"""
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_parquet(root / "data/raw/us_prices_daily.parquet")

    raw = raw.dropna(axis=1, how="all")

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    prices_m = raw.resample("ME").last()
    rets = prices_m.pct_change(fill_method=None)
    rets = rets.loc[:, rets.count() >= 60]  # at least 5 years history

    processed_path = root / "data/processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    rets.to_parquet(processed_path / "us_monthly_returns.parquet")
    print("Saved monthly returns.")


if __name__ == "__main__":
    main()