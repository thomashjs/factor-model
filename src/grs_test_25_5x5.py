"""
GRS joint alpha test for the 25 Size-B/M (5x5) portfolios.

Produces a report on both FF3 and Carhart.

Inputs:
  data/processed/portfolios_25_5x5_monthly.parquet
  data/processed/factors_monthly.parquet

Outputs:
  reports/grs_test.md

Caution:
  Do not run after markdown is written and manually edited, as this will overwrite the report.
  
Implements the Gibbons-Ross-Shanken (1989) F-test using OLS residual
covariance (no HAC).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import f as f_dist
except Exception:
    f_dist = None


@dataclass
class GRSResult:
    model_name: str
    factors: List[str]
    start: str                             # start date
    end: str                               # end date
    T: int                                 # number of time periods
    N: int                                 # number of assets
    K: int                                 # number of factors
    F_stat: float                          # GRS F-statistic
    p_value: float | None                  # p-value

def _ols_alphas_resids(Y: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    OLS with intercept in matrix form.

    Y: (T, N) excess returns
    F: (T, K) factor returns

    Returns:
      alpha: (N,)
      E: (T, N) residuals
    """
    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])  # (T, K+1)

    XtX = X.T @ X
    XtY = X.T @ Y
    B = np.linalg.solve(XtX, XtY)         # (K+1, N)

    alpha = B[0, :]                       # (N,)
    E = Y - X @ B                         # (T, N)
    return alpha, E


def grs_test(
    df: pd.DataFrame,
    ret_cols: Iterable[str],
    factor_cols: List[str],
    model_name: str,
) -> GRSResult:
    """
    GRS F-test:
      F = ((T - N - K)/N) * (1 + mu_f' Sigma_f^{-1} mu_f)^{-1} * alpha' Sigma_e^{-1} alpha

    where Sigma_e is the residual covariance from the system of time-series regressions.
    """
    d = df.copy()
    d = d.sort_values("date").dropna(subset=list(ret_cols) + factor_cols + ["RF"])

    # Excess returns (T, N)
    Y = d[list(ret_cols)].to_numpy(dtype=float) - d[["RF"]].to_numpy(dtype=float)

    # Factors (T, K)
    F = d[factor_cols].to_numpy(dtype=float)

    T, N = Y.shape
    K = F.shape[1]

    alpha, E = _ols_alphas_resids(Y, F)

    # Residual covariance (unbiased): Sigma_e = E'E / (T - K - 1)
    denom = T - K - 1
    if denom <= 0:
        raise ValueError(f"Not enough observations: T={T}, K={K}. Need T > K+1.")
    Sigma_e = (E.T @ E) / denom           # (N, N)

    # Factor mean/covariance
    mu_f = F.mean(axis=0).reshape(-1, 1)  # (K, 1)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)  # (K, K)

    term = 1.0 + float((mu_f.T @ np.linalg.solve(Sigma_f, mu_f)).item())

    a = alpha.reshape(-1, 1)              # (N, 1)
    try:
        quad = float((a.T @ np.linalg.solve(Sigma_e, a)).item())
    except np.linalg.LinAlgError:
        quad = float((a.T @ (np.linalg.pinv(Sigma_e) @ a)).item())

    df1 = N
    df2 = T - N - K
    if df2 <= 0:
        raise ValueError(f"Not enough observations: T={T}, N={N}, K={K}. Need T > N+K.")

    F_stat = ((T - N - K) / N) * (1.0 / term) * quad

    p_value = None
    if f_dist is not None:
        p_value = float(f_dist.sf(F_stat, df1, df2))

    start = d["date"].min().strftime("%Y-%m")
    end = d["date"].max().strftime("%Y-%m")

    return GRSResult(
        model_name=model_name,
        factors=factor_cols,
        start=start,
        end=end,
        T=T,
        N=N,
        K=K,
        F_stat=float(F_stat),
        p_value=p_value,
    )


def write_report(out_path: Path, results: List[GRSResult]) -> None:
    lines: List[str] = []
    lines.append("# GRS joint test of alphas (25 Size-B/M portfolios)")
    lines.append("")
    lines.append(
        "This report tests whether the 25 portfolio alphas are jointly zero under each model "
        "using the classic Gibbons–Ross–Shanken (GRS) F-test."
    )
    lines.append("")
    lines.append(
        "**Important:** This is the *classic* GRS test based on OLS residual covariance (no HAC). "
        "Your panel scripts use Newey–West HAC for per-portfolio t-stats; GRS is reported here in "
        "the standard form used in many papers."
    )
    lines.append("")

    for r in results:
        df2 = r.T - r.N - r.K
        lines.append(f"## {r.model_name}")
        lines.append("")
        lines.append(f"- Sample: **{r.start}** to **{r.end}**")
        lines.append(f"- T = {r.T}, N = {r.N}, K = {r.K}")
        lines.append(f"- Factors: {', '.join(r.factors)}")
        lines.append("")
        p_str = "N/A (install scipy)" if r.p_value is None else f"{r.p_value:.6g}"
        lines.append(f"**GRS F-stat:** {r.F_stat:.6g}  (df1 = {r.N}, df2 = {df2})")
        lines.append(f"**p-value:** {p_str}")
        lines.append("")

    if len(results) == 2:
        lines.append("## Interpretation")
        lines.append("")
        lines.append(
            "Lower GRS statistics/higher p-values indicate that alphas are jointly closer to zero "
            "across the 25 portfolios. Comparing FF3 and Carhart assesses whether the inclusion of "
            "the momentum factor reduces joint mispricing."
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ports_path = root / "data/processed/portfolios_25_5x5_monthly.parquet"
    factors_path = root / "data/processed/factors_monthly.parquet"

    ports = pd.read_parquet(ports_path)
    factors = pd.read_parquet(factors_path)

    df = ports.merge(factors, on="date", how="inner")
    ret_cols = [c for c in ports.columns if c != "date"]

    ff3 = grs_test(df, ret_cols, ["Mkt-RF", "SMB", "HML"], "FF3")
    carhart = grs_test(df, ret_cols, ["Mkt-RF", "SMB", "HML", "Mom"], "Carhart (4-factor)")

    out_path = root / "reports/grs_test.md"
    write_report(out_path, [ff3, carhart])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()