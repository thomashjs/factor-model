"""
Expanding-window walk-forward out-of-sample (OOS) validation for
FF3 vs Carhart on Ken French 25 Size–B/M (5x5) portfolios.

Reads:
  data/processed/portfolios_25_5x5_monthly.parquet
  data/processed/factors_monthly.parquet

Writes:
  reports/oos_walkforward_25_5x5_by_portfolio.csv
  reports/oos_walkforward_25_5x5_summary.csv
  reports/oos_walkforward_25_5x5.md
  figures/oos_delta_mae_by_portfolio.png

Notes:
- Portfolio returns are already excess returns per your message, but we still
  follow your project convention of subtracting RF (safe/consistent).
- OOS here is "walk-forward pricing error": fit betas on history, predict next
  month given realized factors, record residual. This is model validation, not
  return forecasting.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fit_ols_params(Y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    OLS with intercept.
    Y: (T,) excess returns
    F: (T, K) factor returns
    returns beta: (K+1,) where beta[0] is alpha
    """
    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])  # (T, K+1)
    # Solve (X'X)b = X'Y
    return np.linalg.solve(X.T @ X, X.T @ Y)


def walkforward_oos(
    df: pd.DataFrame,
    ret_cols: list[str],
    factor_cols: list[str],
    burn_in: int = 120,
) -> pd.DataFrame:
    """
    Returns tidy OOS residuals with columns:
      date, portfolio, model, y, yhat, resid
    where each row is a 1-step-ahead prediction at date t using params fit on < t.
    """
    d = (
        df.dropna(subset=["date", "RF"] + ret_cols + factor_cols)
          .sort_values("date")
          .reset_index(drop=True)
    )

    out = []
    # Predict date t using fit up through t-1
    for t in range(burn_in, len(d)):
        train = d.iloc[:t]
        test = d.iloc[t]

        F_train = train[factor_cols].to_numpy(float)
        rf_train = train["RF"].to_numpy(float)

        F_test = test[factor_cols].to_numpy(float)
        rf_test = float(test["RF"])

        for p in ret_cols:
            Y_train = train[p].to_numpy(float) - rf_train  # project convention
            beta = _fit_ols_params(Y_train, F_train)

            # 1-step ahead: yhat_t = alpha + beta' * F_t
            y = float(test[p]) - rf_test
            yhat = float(beta[0] + beta[1:] @ F_test)
            resid = y - yhat

            out.append((test["date"], p, y, yhat, resid))

    oos = pd.DataFrame(out, columns=["date", "portfolio", "y", "yhat", "resid"])
    return oos


def summarize_oos(oos: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (by_portfolio, summary_one_row)
    """
    byp = (
        oos.groupby("portfolio")
           .agg(
               n=("resid", "size"),
               mean_resid=("resid", "mean"),
               mae=("resid", lambda x: float(np.mean(np.abs(x)))),
               rmse=("resid", lambda x: float(np.sqrt(np.mean(np.square(x))))),
           )
           .reset_index()
    )
    byp["model"] = model_name

    summ = pd.DataFrame([{
        "model": model_name,
        "n_total": int(len(oos)),
        "mae_avg_portfolio": float(byp["mae"].mean()),
        "rmse_avg_portfolio": float(byp["rmse"].mean()),
        "alpha_dispersion_oos_std_mean_resid": float(byp["mean_resid"].std(ddof=1)),
        "mean_abs_alpha_oos_mean_resid": float(np.mean(np.abs(byp["mean_resid"]))),
    }])

    return byp, summ


def main(burn_in: int = 120):
    root = Path(__file__).resolve().parents[1]

    ports = pd.read_parquet(root / "data/processed/portfolios_25_5x5_monthly.parquet")
    factors = pd.read_parquet(root / "data/processed/factors_monthly.parquet")

    ports["date"] = pd.to_datetime(ports["date"])
    factors["date"] = pd.to_datetime(factors["date"])

    df = ports.merge(factors, on="date", how="inner")
    ret_cols = [c for c in ports.columns if c != "date"]

    # --- FF3 ---
    oos_ff3 = walkforward_oos(df, ret_cols, ["Mkt-RF", "SMB", "HML"], burn_in=burn_in)
    oos_ff3["model"] = "FF3"

    # --- Carhart ---
    oos_car = walkforward_oos(df, ret_cols, ["Mkt-RF", "SMB", "HML", "Mom"], burn_in=burn_in)
    oos_car["model"] = "Carhart"

    # --- Summaries ---
    byp_ff3, summ_ff3 = summarize_oos(oos_ff3, "FF3")
    byp_car, summ_car = summarize_oos(oos_car, "Carhart")

    byp = pd.concat([byp_ff3, byp_car], ignore_index=True)
    summ = pd.concat([summ_ff3, summ_car], ignore_index=True)

    # Delta MAE by portfolio (Carhart - FF3)
    mae_delta = (
        byp.pivot(index="portfolio", columns="model", values="mae")
           .dropna()
           .assign(delta_mae=lambda x: x["Carhart"] - x["FF3"])
           .reset_index()
           .sort_values("delta_mae")
    )

    # --- Write outputs ---
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)

    byp_path = root / "reports/oos_walkforward_25_5x5_by_portfolio.csv"
    summ_path = root / "reports/oos_walkforward_25_5x5_summary.csv"

    byp.to_csv(byp_path, index=False)
    summ.to_csv(summ_path, index=False)

    # Plot: Delta MAE by portfolio
    plt.figure(figsize=(8, 10))
    plt.scatter(mae_delta["delta_mae"], mae_delta["portfolio"])
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("ΔMAE (Carhart − FF3)  [lower is better for Carhart]")
    plt.ylabel("portfolio")
    plt.title(f"OOS pricing error change by portfolio (burn-in={burn_in} months)")
    fig_path = root / "figures/oos_delta_mae_by_portfolio.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Markdown report
    lines = []
    lines.append("# Walk-forward OOS Validation (25 Size–B/M Portfolios)")
    lines.append("")
    lines.append(f"- Scheme: expanding-window, 1-step-ahead residuals (burn-in = {burn_in} months)")
    lines.append("- Metric: pricing error residual = realized excess return − model-implied return")
    lines.append("")
    for _, r in summ.sort_values("model").iterrows():
        lines.append(f"## {r['model']}")
        lines.append(f"- Avg portfolio MAE: {r['mae_avg_portfolio']:.6g}")
        lines.append(f"- Avg portfolio RMSE: {r['rmse_avg_portfolio']:.6g}")
        lines.append(f"- OOS alpha dispersion (std of mean residuals): {r['alpha_dispersion_oos_std_mean_resid']:.6g}")
        lines.append(f"- Mean |OOS alpha| (mean residual): {r['mean_abs_alpha_oos_mean_resid']:.6g}")
        lines.append("")

    lines.append("### Interpretation")
    lines.append(
        "Lower MAE/RMSE indicates improved out-of-sample pricing. "
        "Reduced dispersion of mean residuals suggests alphas are closer to zero across portfolios."
    )
    lines.append("")
    lines.append("### Files")
    lines.append(f"- {byp_path.name}")
    lines.append(f"- {summ_path.name}")
    lines.append(f"- {fig_path.name}")

    md_path = root / "reports/oos_walkforward_25_5x5.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {byp_path}")
    print(f"Wrote {summ_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()