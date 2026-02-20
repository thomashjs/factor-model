"""
60-month rolling alphas for 25 Size-B/M (5x5) portfolios: FF3 vs Carhart.

Reads:
  data/processed/portfolios_25_5x5_monthly.parquet
  data/processed/factors_monthly.parquet

Writes:
  figures/rolling_alpha_ff3.png
  figures/rolling_alpha_carhart.png
  figures/rolling_alpha_dispersion.png
  reports/rolling_alpha_60m_summary.csv

Notes:
- Rolling t-stats here use classic OLS (no HAC) for speed/stability in short windows.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ols_alpha_tstat(Y: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    """
    Single-asset OLS with intercept.
    Y: (T,) excess returns
    F: (T, K) factor returns
    Returns (alpha, t_alpha) using classic OLS SE.
    """
    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])          # (T, K+1)
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ Y)          # (K+1,)
    resid = Y - X @ beta
    k1 = X.shape[1]
    dof = T - k1
    if dof <= 0:
        return float(beta[0]), np.nan

    s2 = (resid @ resid) / dof
    covb = s2 * np.linalg.inv(XtX)
    se_alpha = np.sqrt(covb[0, 0])
    t_alpha = beta[0] / se_alpha if se_alpha > 0 else np.nan
    return float(beta[0]), float(t_alpha)


def rolling_alphas(
    df: pd.DataFrame,
    ret_cols: list[str],
    factor_cols: list[str],
    window: int = 60,
) -> pd.DataFrame:
    """
    Returns tidy frame with one row per (date_end, portfolio):
      date, portfolio, alpha, t_alpha
    """
    d = df.dropna(subset=ret_cols + factor_cols + ["RF"]).sort_values("date").reset_index(drop=True)

    out = []
    for end in range(window - 1, len(d)):
        w = d.iloc[end - window + 1 : end + 1]
        F = w[factor_cols].to_numpy(float)
        rf = w["RF"].to_numpy(float)

        date_end = w["date"].iloc[-1]

        for p in ret_cols:
            Y = w[p].to_numpy(float) - rf
            a, t = _ols_alpha_tstat(Y, F)
            out.append((date_end, p, a, t))

    return pd.DataFrame(out, columns=["date", "portfolio", "alpha", "t_alpha"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ports_path = root / "data/processed/portfolios_25_5x5_monthly.parquet"
    factors_path = root / "data/processed/factors_monthly.parquet"

    ports = pd.read_parquet(ports_path)
    factors = pd.read_parquet(factors_path)

    # defensive: ensure datetime + inner-merge
    ports["date"] = pd.to_datetime(ports["date"])
    factors["date"] = pd.to_datetime(factors["date"])

    df = ports.merge(factors, on="date", how="inner")

    ret_cols = [c for c in ports.columns if c != "date"]

    window = 60

    roll_ff3 = rolling_alphas(df, ret_cols, ["Mkt-RF", "SMB", "HML"], window=window)
    roll_ff3["model"] = "FF3"

    roll_car = rolling_alphas(df, ret_cols, ["Mkt-RF", "SMB", "HML", "Mom"], window=window)
    roll_car["model"] = "Carhart"

    roll = pd.concat([roll_ff3, roll_car], ignore_index=True)

    # Summary by date/model
    summ = (
        roll.groupby(["model", "date"])
        .agg(
            mean_abs_alpha=("alpha", lambda x: float(np.mean(np.abs(x)))),
            std_alpha=("alpha", lambda x: float(np.std(x, ddof=1))),
            frac_sig_5pct=("t_alpha", lambda x: float(np.mean(np.abs(x) > 1.96))),
            mean_t_alpha=("t_alpha", "mean"),
        )
        .reset_index()
        .sort_values(["model", "date"])
    )

    (root / "reports").mkdir(parents=True, exist_ok=True)
    summ.to_csv(root / "reports/rolling_alpha_60m_summary.csv", index=False)

    (root / "figures").mkdir(parents=True, exist_ok=True)

    # Plot 1: mean |alpha| over time (FF3)
    ff3_s = summ[summ["model"] == "FF3"]
    plt.figure()
    plt.plot(ff3_s["date"], ff3_s["mean_abs_alpha"])
    plt.title("60-month rolling mean |alpha| (FF3) across 25 portfolios")
    plt.xlabel("Date")
    plt.ylabel("Mean |alpha|")
    plt.tight_layout()
    plt.savefig(root / "figures/rolling_alpha_ff3.png", dpi=200)
    plt.close()

    # Plot 2: mean |alpha| over time (Carhart)
    car_s = summ[summ["model"] == "Carhart"]
    plt.figure()
    plt.plot(car_s["date"], car_s["mean_abs_alpha"])
    plt.title("60-month rolling mean |alpha| (Carhart) across 25 portfolios")
    plt.xlabel("Date")
    plt.ylabel("Mean |alpha|")
    plt.tight_layout()
    plt.savefig(root / "figures/rolling_alpha_carhart.png", dpi=200)
    plt.close()

    # Plot 3: dispersion comparison (std of alpha)
    plt.figure()
    plt.plot(ff3_s["date"], ff3_s["std_alpha"], label="FF3")
    plt.plot(car_s["date"], car_s["std_alpha"], label="Carhart")
    plt.title("60-month rolling alpha dispersion (cross-sectional std)")
    plt.xlabel("Date")
    plt.ylabel("Std(alpha) across 25 portfolios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "figures/rolling_alpha_dispersion.png", dpi=200)
    plt.close()

    print("Wrote figures/rolling_alpha_*.png and reports/rolling_alpha_60m_summary.csv")


if __name__ == "__main__":
    main()