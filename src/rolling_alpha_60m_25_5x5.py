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
- Rolling t-stats here use classic OLS for speed/stability in short windows.
"""

from __future__ import annotations

from logging import root
from logging import root
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ols_alpha_tstat(Y: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    # can optimize by doing Multiple Linear Regression, but this is fast enough for 25 assets.
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

# Plot mean alpha over time for given model with 95% CI
def plot_rolling_alphas(summ: pd.DataFrame, model_name: str, root: Path) -> None:
    model_summ = summ[summ["model"] == model_name]
    plt.figure()
    plt.plot(model_summ["date"], model_summ["mean_alpha"], label="Mean α")
    plt.fill_between(
        model_summ["date"],
        model_summ["ci_low"],
        model_summ["ci_high"],
        alpha=0.25,
        label="95% CI"
    )

    plt.title(f"60-month rolling mean alpha ({model_name}) across 25 portfolios")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Mean alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / f"figures/rolling_alpha_{model_name.lower()}.png", dpi=200)
    plt.close()

    print(f"Wrote figures/rolling_alpha_{model_name.lower()}.png")

# Plot mean |alpha| over time for given model
def plot_rolling_alpha_abs(summ, model_name, root) -> None:
    plt.figure()
    model_summ = summ[summ["model"] == model_name]
    plt.plot(model_summ["date"], model_summ["mean_abs_alpha"])
    idx_min = (model_summ["mean_abs_alpha"]).idxmin()
    date_min = model_summ.loc[idx_min, "date"]
    val_min = model_summ.loc[idx_min, "mean_abs_alpha"]

    plt.scatter(date_min, val_min)
    plt.annotate(f"mean |α| = {val_min:.3f}", (date_min, val_min))

    plt.title(f"60-month rolling mean |alpha| across 25 portfolios ({model_name})")
    plt.xlabel("Date")
    plt.ylabel("Mean |alpha|")
    plt.tight_layout()
    plt.savefig(root / f"figures/rolling_alpha_abs_{model_name.lower()}.png", dpi=200)
    plt.close()

    print(f"Wrote figures/rolling_alpha_abs_{model_name.lower()}.png")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ports_path = root / "data/processed/portfolios_25_5x5_monthly.parquet"
    factors_path = root / "data/processed/factors_monthly.parquet"

    ports = pd.read_parquet(ports_path)
    factors = pd.read_parquet(factors_path)

    ports["date"] = pd.to_datetime(ports["date"])
    factors["date"] = pd.to_datetime(factors["date"]) # ensure datetime in case not already

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
            mean_alpha=("alpha", "mean"),
            mean_abs_alpha=("alpha", lambda x: float(np.mean(np.abs(x)))),
            std_alpha=("alpha", lambda x: float(np.std(x, ddof=1))),
            frac_sig_5pct=("t_alpha", lambda x: float(np.mean(np.abs(x) > 1.96))),
            mean_t_alpha=("t_alpha", "mean"),
        )
        .reset_index()
        .sort_values(["model", "date"])
    )
    
    # Standard error of cross-sectional mean
    N = len(ret_cols)
    # 95% CI (t = 1.96): population mean +- t*SE/sqrt(N)
    summ["se_mean_alpha"] = summ["std_alpha"] / np.sqrt(N)
    summ["ci_low"] = summ["mean_alpha"] - 1.96 * summ["se_mean_alpha"] 
    summ["ci_high"] = summ["mean_alpha"] + 1.96 * summ["se_mean_alpha"]

    (root / "reports").mkdir(parents=True, exist_ok=True)
    summ.to_csv(root / "reports/rolling_alpha_60m_summary.csv", index=False)
    print("Wrote reports/rolling_alpha_60m_summary.csv")

    (root / "figures").mkdir(parents=True, exist_ok=True)

    # Plot mean alpha with CI and mean |alpha| over time for each model
    plot_rolling_alphas(summ, "FF3", root)
    plot_rolling_alpha_abs(summ, "FF3", root)
    plot_rolling_alphas(summ, "Carhart", root)
    plot_rolling_alpha_abs(summ, "Carhart", root)
    
    # Plot dispersion comparison (std of alpha)
    ff3_s = summ[summ["model"] == "FF3"]
    car_s = summ[summ["model"] == "Carhart"]
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

    print("Wrote figures/rolling_alpha_dispersion.png")


if __name__ == "__main__":
    main()