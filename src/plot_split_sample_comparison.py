"""
Plots comparing mean alpha across regimes for the split-sample stability analyses.

Reads:
  reports/split_sample_2_period_stability.csv
  reports/split_sample_3_period_stability.csv

Writes:
  figures/split_sample_2_period_comparison.png
  figures/split_sample_3_period_comparison.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_split_sample_comparison(csv_path: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)

    # Aggregate to mean alpha by model & period
    summ = (
        df.groupby(["model", "period"])
        .agg(mean_alpha=("alpha", "mean"))
        .reset_index()
    )

    periods = sorted(summ["period"].unique())
    models = sorted(summ["model"].unique())

    fig, ax = plt.subplots()

    width = 0.35
    x = range(len(periods))

    for i, model in enumerate(models):
        model_vals = [
            summ[(summ["model"] == model) & (summ["period"] == p)]["mean_alpha"].values[0]
            for p in periods
        ]
        ax.bar(
            [xi + i * width for xi in x],
            model_vals,
            width=width,
            label=model
        )

    ax.axhline(0)
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(periods)
    ax.set_title(title)
    ax.set_ylabel("Mean alpha")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Wrote {output_path}")


def main():
    root = Path(__file__).resolve().parents[1]

    # 2-period
    plot_split_sample_comparison(
        root / "reports/split_sample_2_period_stability.csv",
        root / "figures/split_sample_2_period_comparison.png",
        "Mean alpha by regime (2-period split)"
    )

    # 3-period
    plot_split_sample_comparison(
        root / "reports/split_sample_3_period_stability.csv",
        root / "figures/split_sample_3_period_comparison.png",
        "Mean alpha by regime (3-period split)"
    )


if __name__ == "__main__":
    main()