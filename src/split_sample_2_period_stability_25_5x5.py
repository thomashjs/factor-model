"""
Split-sample stability for 25 Size-B/M portfolios.

Inputs:
  rewrite_csv: bool = False   # if True, overwrite the CSV report
  rewrite_md: bool = False    # if True, overwrite the markdown report

Reads:
  data/processed/portfolios_25_5x5_monthly.parquet
  data/processed/factors_monthly.parquet

Splits:
  - pre/post 2009-01
  - pre/post 2020-01

Writes:
  reports/split_sample_stability.csv
  reports/split_sample_stability.md

Caution: do NOT carelessly run after markdown is written and manually edited, as this overwrites the report.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _ols_alpha(Y: np.ndarray, F: np.ndarray) -> float:
    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])
    beta = np.linalg.solve(X.T @ X, X.T @ Y)
    return float(beta[0])


def run_model(df, ret_cols, factor_cols, model_name):
    rows = []
    for p in ret_cols:
        Y = df[p].to_numpy(float) - df["RF"].to_numpy(float)
        F = df[factor_cols].to_numpy(float)
        a = _ols_alpha(Y, F)
        rows.append((model_name, p, a))
    return pd.DataFrame(rows, columns=["model", "portfolio", "alpha"])


def evaluate_period(df, ret_cols, factor_cols, model_name, label):
    result = run_model(df, ret_cols, factor_cols, model_name)
    result["period"] = label
    return result


def main(rewrite_csv=False, rewrite_md=False):
    root = Path(__file__).resolve().parents[1]
    ports = pd.read_parquet(root / "data/processed/portfolios_25_5x5_monthly.parquet")
    factors = pd.read_parquet(root / "data/processed/factors_monthly.parquet")

    ports["date"] = pd.to_datetime(ports["date"])
    factors["date"] = pd.to_datetime(factors["date"])

    df = ports.merge(factors, on="date", how="inner").sort_values("date")

    ret_cols = [c for c in ports.columns if c != "date"]

    splits = {
        "pre_2009": df[df["date"] < "2009-01-01"],
        "post_2009": df[df["date"] >= "2009-01-01"],
        "pre_2020": df[df["date"] < "2020-01-01"],
        "post_2020": df[df["date"] >= "2020-01-01"],
    }

    all_results = []

    for label, dsub in splits.items():
        if len(dsub) < 60:
            continue

        all_results.append(
            evaluate_period(dsub, ret_cols, ["Mkt-RF", "SMB", "HML"], "FF3", label)
        )
        all_results.append(
            evaluate_period(dsub, ret_cols, ["Mkt-RF", "SMB", "HML", "Mom"], "Carhart", label)
        )

    res = pd.concat(all_results, ignore_index=True)

    # Add dispersion summary
    summ = (
        res.groupby(["model", "period"])
        .agg(
            mean_alpha=("alpha", "mean"),
            std_alpha=("alpha", "std"),
            mean_abs_alpha=("alpha", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )

    if rewrite_csv:
        (root / "reports").mkdir(parents=True, exist_ok=True)
        res.to_csv(root / "reports/split_sample_2_period_stability.csv", index=False)
    else:
        print("CSV report not rewritten (rewrite_csv=False).")

    # Markdown report
    lines = []
    lines.append("# Split-Sample Stability (25 Size-B/M Portfolios)")
    lines.append("")
    lines.append("Models estimated separately across subperiods.")
    lines.append("")

    for _, row in summ.iterrows():
        lines.append(
            f"- **{row['model']} | {row['period']}**: "
            f"mean α = {row['mean_alpha']:.6g}, "
            f"std α = {row['std_alpha']:.6g}, "
            f"mean |α| = {row['mean_abs_alpha']:.6g}"
        )

    lines.append("")
    lines.append(
        "Interpretation: stability is indicated if alpha dispersion and magnitude do not materially increase "
        "in later subsamples."
    )

    if rewrite_md:
        (root / "reports/split_sample_2_period_stability.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        print("Wrote reports/split_sample_2_period_stability.*")
    else:
        print("Markdown report not rewritten (rewrite_md=False).")

if __name__ == "__main__":
    main()
