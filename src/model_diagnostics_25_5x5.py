"""
Model diagnostics for 25 Size-B/M portfolios.

Outputs:
  reports/model_diagnostics_25_5x5.csv
  reports/model_diagnostics_25_5x5.md
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_time_series_regression(Y, F):
    X = sm.add_constant(F)
    model = sm.OLS(Y, X).fit()
    model_hac = model.get_robustcov_results(cov_type="HAC", maxlags=12)
    return model, model_hac


def run_model(df, ret_cols, factor_cols, model_name):
    rows = []

    for p in ret_cols:
        Y = df[p] - df["RF"]
        F = df[factor_cols]

        ols, hac = run_time_series_regression(Y, F)

        resid = ols.resid
        lb_p = acorr_ljungbox(resid, lags=[12], return_df=True)["lb_pvalue"].iloc[0]

        alpha_t_ols = ols.tvalues.iloc[0]  # OLS attributes are PD series, use iloc for strictly position-based indexing
        alpha_t_hac = hac.tvalues[0]       # HAC results are numpy arrays, so use [0]

        se_ratio = hac.bse[0] / ols.bse.iloc[0] if ols.bse.iloc[0] > 0 else np.nan

        row = {
            "model": model_name,
            "portfolio": p,
            "alpha_t_ols": alpha_t_ols,
            "alpha_t_hac": alpha_t_hac,
            "hac_to_ols_se_ratio": se_ratio,
            "ljungbox_pvalue_lag12": lb_p,
        }

        if "Mom" in factor_cols:
            row["mom_loading"] = ols.params.get("Mom", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]

    ports = pd.read_parquet(root / "data/processed/portfolios_25_5x5_monthly.parquet")
    factors = pd.read_parquet(root / "data/processed/factors_monthly.parquet")

    ports["date"] = pd.to_datetime(ports["date"])
    factors["date"] = pd.to_datetime(factors["date"])

    df = ports.merge(factors, on="date", how="inner")

    ret_cols = [c for c in ports.columns if c != "date"]

    ff3 = run_model(df, ret_cols, ["Mkt-RF", "SMB", "HML"], "FF3")
    carhart = run_model(df, ret_cols, ["Mkt-RF", "SMB", "HML", "Mom"], "Carhart")

    res = pd.concat([ff3, carhart], ignore_index=True)

    # Cross-sectional summary
    summ = (
        res.groupby("model")
        .agg(
            mean_alpha_t_hac=("alpha_t_hac", "mean"),
            frac_sig_alpha_5pct=("alpha_t_hac", lambda x: np.mean(np.abs(x) > 1.96)),
            mean_hac_to_ols_se_ratio=("hac_to_ols_se_ratio", "mean"),
            frac_resid_autocorr_5pct=("ljungbox_pvalue_lag12", lambda x: np.mean(x < 0.05)),
        )
        .reset_index()
    )

    (root / "reports").mkdir(parents=True, exist_ok=True)
    res.to_csv(root / "reports/model_diagnostics_25_5x5.csv", index=False)

    # Markdown
    lines = []
    lines.append("# Model Diagnostics (25 Size-B/M Portfolios)")
    lines.append("")
    lines.append("Cross-sectional summary statistics:")
    lines.append("")

    for _, row in summ.iterrows():
        lines.append(f"## {row['model']}")
        lines.append(f"- Mean alpha t-stat (HAC): {row['mean_alpha_t_hac']:.3f}")
        lines.append(f"- Fraction |t_alpha| > 1.96: {row['frac_sig_alpha_5pct']:.3f}")
        lines.append(f"- Mean HAC/OLS SE ratio: {row['mean_hac_to_ols_se_ratio']:.3f}")
        lines.append(f"- Fraction residual autocorr (p<0.05): {row['frac_resid_autocorr_5pct']:.3f}")
        lines.append("")

    lines.append(
        "Interpretation: large HAC/OLS ratios indicate serial correlation; "
        "high residual autocorrelation fractions suggest time-series dependence "
        "in regression errors."
    )

    (root / "reports/model_diagnostics_25_5x5.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("Wrote reports/model_diagnostics_25_5x5.*")


if __name__ == "__main__":
    main()