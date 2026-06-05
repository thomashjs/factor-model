"""
Utility functions for cross-sectional analysis.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm


def run_fama_macbeth(signal: pd.DataFrame,
                     returns: pd.DataFrame,
                     min_obs: int = 100,
                     use_rank: bool = False,
                     winsorize: list = [0.01,0.99]):
    gammas = []
    future_returns = returns.shift(-1)

    common_dates = signal.index.intersection(returns.index)
    future_returns = returns.shift(-1)
    for date in common_dates:
        next_ret = future_returns.loc[date]
        if next_ret.notna().sum() == 0:
            continue

        df = pd.DataFrame({
            "ret": next_ret,
            "signal": signal.loc[date]
        }).dropna()

        df["ret"] = df["ret"].clip(
            lower=df["ret"].quantile(winsorize[0]),
            upper=df["ret"].quantile(winsorize[1])
        )

        if len(df) < min_obs:
            continue

        if use_rank:
            df["rank"] = df["signal"].rank(pct=True) # use rank instead of raw signal
            model = sm.OLS(df["ret"], sm.add_constant(df["rank"])).fit()
            gammas.append(model.params["rank"])
        else:
            model = sm.OLS(df["ret"], sm.add_constant(df["signal"])).fit()
            gammas.append(model.params["signal"])

    gammas = pd.Series(gammas)

    gamma_mean = gammas.mean()
    t_stat = gamma_mean / (gammas.std(ddof=1) / np.sqrt(len(gammas)))

    return gamma_mean, t_stat, gammas


def build_deciles(signal: pd.DataFrame,
                  returns: pd.DataFrame,
                  n_deciles: int = 10,
                  min_obs: int = 100,
                  winsorize: list = [0.01,0.99]):

    wml_series = []

    for date in signal.index[:-1]:
        next_ret = returns.shift(-1).loc[date]
        df = pd.DataFrame({
            "ret": next_ret,
            "signal": signal.loc[date]
        }).dropna()

        if len(df) < min_obs:
            continue

        df["ret"] = df["ret"].clip(lower=df["ret"].quantile(winsorize[0]),
                           upper=df["ret"].quantile(winsorize[1]))
        
        df["decile"] = pd.qcut(
            df["signal"],
            n_deciles,
            labels=False,
            duplicates="drop"
        )

        high = df[df["decile"] == n_deciles - 1]["ret"].mean()
        low = df[df["decile"] == 0]["ret"].mean()

        wml_series.append(high - low)

    return pd.Series(wml_series)


def cumulative_returns(series: pd.Series):
    return (1 + series).cumprod()