"""
Stock-level 12–1 momentum on US stocks of a chosen quantity (equal-weight)

Pipeline:
get_us_tickers() ------------ Pull current US tickers
download_prices() ----------- Download daily data from Stooq
build_monthly_returns() ----- Convert to monthly returns
build_momentum() ------------ Build 12–1 momentum signal
build_decile_returns() ------ Form equal-weight decile portfolios
fama_macbeth() -------------- Compute decile spread and Fama–MacBeth slope

Outputs:
- reports/sp500_mom_deciles_summary.csv
- reports/sp500_fmb_summary.csv
- figures/sp500_mom_cumulative.png
"""

from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import statsmodels.api as sm
import requests


# --------------------------------------------------
# Get US tickers
# --------------------------------------------------

def get_us_tickers(max_n=1800):
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text), sep="|")

    # Drop last summary row
    df = df[df["Symbol"] != "File Creation Time"]

    tickers = df["Symbol"].tolist()

    # Remove test issues
    tickers = df[df["Test Issue"] == "N"]["Symbol"].tolist()

    return tickers[:max_n]


# --------------------------------------------------
# Download prices from Stooq
# --------------------------------------------------

def download_prices(tickers, start="2000-01-01"):
    prices = {}
    for i, t in enumerate(tickers):
        try:
            df = pdr.DataReader(t, "stooq", start=start)
            df = df.sort_index()
            prices[t] = df["Close"]
        except Exception:
            continue

        if i % 100 == 0:
            print(f"Downloaded {i} tickers...")

    prices = pd.DataFrame(prices)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[:, prices.count() > 36]  # at least 3 years history
    return prices


# --------------------------------------------------
# Monthly returns
# --------------------------------------------------

def build_monthly_returns(prices):
    prices_m = prices.resample("ME").last()
    rets = prices_m.pct_change()
    return rets


# --------------------------------------------------
# 12–1 Momentum
# --------------------------------------------------

def build_momentum(rets):
    mom = (
        (1 + rets)
        .rolling(11)
        .apply(np.prod, raw=True)
        .shift(1)
        - 1
    )
    return mom


# --------------------------------------------------
# Decile portfolios (equal-weight)
# --------------------------------------------------

def build_decile_returns(rets, mom):
    decile_returns = {}

    for date in mom.index[:-1]:
        signal = mom.loc[date]
        next_ret = rets.shift(-1).loc[date]

        valid = signal.dropna().index.intersection(next_ret.dropna().index)
        if len(valid) < 50:
            continue

        ranks = pd.qcut(signal[valid], 10, labels=False)

        for d in range(10):
            members = valid[ranks == d]
            if len(members) > 0:
                r = next_ret[members].mean()
                decile_returns.setdefault(d+1, []).append((date, r))

    decile_df = {
        d: pd.DataFrame(vals, columns=["date", "ret"]).set_index("date")
        for d, vals in decile_returns.items()
    }

    return decile_df


# --------------------------------------------------
# Fama–MacBeth
# --------------------------------------------------

def fama_macbeth(rets, mom):
    gammas = []

    for date in mom.index[:-1]:
        signal = mom.loc[date]
        next_ret = rets.shift(-1).loc[date]

        df = pd.DataFrame({
            "ret": next_ret,
            "mom": signal
        }).dropna()

        if len(df) < 50:
            continue

        X = sm.add_constant(df["mom"])
        model = sm.OLS(df["ret"], X).fit()
        gammas.append(model.params["mom"])

    gammas = pd.Series(gammas)
    gamma_mean = gammas.mean()
    t_stat = gamma_mean / (gammas.std(ddof=1) / np.sqrt(len(gammas)))

    return gamma_mean, t_stat, gammas


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    root = Path(__file__).resolve().parents[1]
    (root / "reports").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    tickers = get_us_tickers()
    prices = download_prices(tickers)
    rets = build_monthly_returns(prices)
    mom = build_momentum(rets)

    deciles = build_decile_returns(rets, mom)

    # Winner minus Loser
    wml = deciles[10]["ret"] - deciles[1]["ret"]
    cum = (1 + wml).cumprod()

    # Plot
    plt.figure(figsize=(8, 5))
    cum.plot()
    plt.title("US 1800 Momentum (12–1) Winner – Loser")
    plt.tight_layout()
    plt.savefig(root / "figures/us1800_mom_cumulative.png", dpi=200)
    plt.close()

    # Fama–MacBeth
    gamma_mean, t_stat, gammas = fama_macbeth(rets, mom)

    # Save summaries
    pd.DataFrame({
        "gamma_mean": [gamma_mean],
        "t_stat": [t_stat],
        "n_months": [len(gammas)]
    }).to_csv(root / "reports/us1800_fmb_summary.csv", index=False)

    print("Done.")
    print("Gamma:", gamma_mean)
    print("t-stat:", t_stat)
    print("Approx cumulative W-L return:", float(cum.iloc[-1] - 1))


if __name__ == "__main__":
    main()