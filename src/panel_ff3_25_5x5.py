"""
Run FF3 time-series regressions for each of the 25 Size-B/M portfolios.

Inputs (decimal units):
- data/processed/portfolios_25_5x5_monthly.parquet  (date + 25 returns)
- data/processed/factors_monthly.parquet           (date + factors incl RF)

Output:
- reports/ff3_25_5x5_results.csv
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

MAXLAGS = 3  # Newey-West lags for monthly data

def run_one(df: pd.DataFrame, ret_col: str) -> dict:
    y = df[ret_col] - df["RF"]  # excess return
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": MAXLAGS})

    return {
        "portfolio": ret_col,
        "n": int(res.nobs),
        "alpha": float(res.params["const"]),
        "alpha_t": float(res.tvalues["const"]),
        "beta_mkt": float(res.params["Mkt-RF"]),
        "beta_mkt_t": float(res.tvalues["Mkt-RF"]),
        "beta_smb": float(res.params["SMB"]),
        "beta_smb_t": float(res.tvalues["SMB"]),
        "beta_hml": float(res.params["HML"]),
        "beta_hml_t": float(res.tvalues["HML"]),
        "r2": float(res.rsquared),
    }

def main():
    root = Path(__file__).resolve().parents[1]
    ports = pd.read_parquet(root / "data/processed/portfolios_25_5x5_monthly.parquet")
    factors = pd.read_parquet(root / "data/processed/factors_monthly.parquet")

    df = ports.merge(factors, on="date", how="inner").dropna()
    ret_cols = [c for c in ports.columns if c != "date"] # list of 25 portfolio return columns

    results = [run_one(df, c) for c in ret_cols] # run regression for each portfolio, store in list of dicts
    out = pd.DataFrame(results).sort_values("portfolio")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_path = reports / "ff3_25_5x5_results.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(out.head())

if __name__ == "__main__":
    main()