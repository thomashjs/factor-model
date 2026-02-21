"""
Mean-Variance Spanning and Sharpe Ratio Comparison

Reads:
  data/processed/factors_monthly.parquet

Writes:
  reports/meanvar_sharpe_comparison.csv
  reports/tangency_weights.csv
  reports/mom_spanning_wald.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ANNUALIZE = 12

def max_sharpe(mu: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Max Sharpe (with rf=0) for excess-return assets.
    SR* = sqrt(mu' Sigma^{-1} mu)
    """
    invS = np.linalg.inv(Sigma)
    sr2 = float(mu.T @ invS @ mu)
    return np.sqrt(max(sr2, 0.0))

def tangency_weights(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Unnormalized tangency weights proportional to Sigma^{-1} mu.
    Scale doesn't matter for Sharpe; normalize to sum to 1 for readability.
    """
    w = np.linalg.solve(Sigma, mu)
    s = w.sum()
    return w / s if abs(s) > 1e-12 else w

def wald_spanning_test_mom_on_ff3(df_excess: pd.DataFrame) -> dict:
    """
    Regression: Mom = alpha + beta' * FF3 + eps
    Wald test for alpha=0 and 1'beta = 1.
    """
    y = df_excess["Mom"].values
    X = df_excess[["Mkt-RF", "SMB", "HML"]].values
    X = sm.add_constant(X)  # [alpha, betas...]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    # Restriction matrix R * params = q
    # params = [alpha, b_mkt, b_smb, b_hml]
    R = np.array([
        [1, 0, 0, 0],         # alpha = 0
        [0, 1, 1, 1],         # b_mkt + b_smb + b_hml = 1
    ], dtype=float)
    q = np.array([0.0, 1.0], dtype=float)

    wald = model.wald_test((R, q),scalar=True)
    return {
        "alpha": model.params[0],
        "beta_mkt": model.params[1],
        "beta_smb": model.params[2],
        "beta_hml": model.params[3],
        "wald_stat": float(wald.statistic),
        "wald_pvalue": float(wald.pvalue),
        "n_obs": int(model.nobs),
    }

def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(root / "data/processed/factors_monthly.parquet")

    df = df.dropna().copy()

    # --- FF3 vs Carhart max Sharpe ---
    ff3 = df[["Mkt-RF", "SMB", "HML"]]
    carhart = df[["Mkt-RF", "SMB", "HML", "Mom"]]

    mu_ff3 = ff3.mean().values
    mu_c = carhart.mean().values

    S_ff3 = ff3.cov().values
    S_c = carhart.cov().values

    sr_ff3 = max_sharpe(mu_ff3, S_ff3)
    sr_c = max_sharpe(mu_c, S_c)

    out = pd.DataFrame([{
        "SR_max_monthly_FF3": sr_ff3,
        "SR_max_monthly_Carhart": sr_c,
        "SR_max_annual_FF3": sr_ff3 * np.sqrt(ANNUALIZE),
        "SR_max_annual_Carhart": sr_c * np.sqrt(ANNUALIZE),
        "Delta_SR_annual": (sr_c - sr_ff3) * np.sqrt(ANNUALIZE),
        "Delta_SR2_monthly": (sr_c**2 - sr_ff3**2),
    }])

    # Optional: show tangency weights (normalized for readability)
    w_ff3 = tangency_weights(mu_ff3, S_ff3)
    w_c = tangency_weights(mu_c, S_c)

    w_tbl = pd.DataFrame({
        "FF3_tan_w": np.r_[w_ff3, [np.nan]],
        "Carhart_tan_w": w_c
    }, index=["Mkt-RF", "SMB", "HML", "Mom"])

    # --- Spanning regression diagnostic ---
    wald = wald_spanning_test_mom_on_ff3(df[["Mkt-RF","SMB","HML","Mom"]])

    # --- Save outputs ---
    (root / "reports").mkdir(exist_ok=True)
    out.to_csv(root / "reports/meanvar_sharpe_comparison.csv", index=False)
    w_tbl.to_csv(root / "reports/tangency_weights.csv")
    pd.DataFrame([wald]).to_csv(root / "reports/mom_spanning_wald.csv", index=False)

    print("Wrote reports/meanvar_sharpe_comparison.csv, tangency_weights.csv, mom_spanning_wald.csv")
    print(out.to_string(index=False))
    print("\nSpanning Wald test:", wald)

if __name__ == "__main__":
    main()