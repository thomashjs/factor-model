"""
Downloads daily price data for US stocks from Yahoo Finance

Writes: 
- data/raw/us_prices_daily.parquet
"""

from pathlib import Path
import pandas as pd
import requests
from io import StringIO
import yfinance as yf
import time


def get_us_tickers(max_n=1200):
    import requests
    from io import StringIO

    # NASDAQ
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    r1 = requests.get(nasdaq_url)
    df1 = pd.read_csv(StringIO(r1.text), sep="|")
    df1 = df1[df1["Symbol"] != "File Creation Time"]
    df1 = df1[df1["Test Issue"] == "N"]
    tickers1 = df1["Symbol"].tolist()

    # NYSE + others
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    r2 = requests.get(other_url)
    df2 = pd.read_csv(StringIO(r2.text), sep="|")
    df2 = df2[df2["ACT Symbol"] != "File Creation Time"]
    df2 = df2[df2["Test Issue"] == "N"]
    tickers2 = df2["ACT Symbol"].tolist()

    tickers = list(set(tickers1 + tickers2))

    # Remove NaNs and non-strings
    tickers = [t for t in tickers if isinstance(t, str)]

    # Remove problematic Yahoo symbols
    tickers = [
        t for t in tickers
        if "^" not in t and "/" not in t and "$" not in t
    ]

    return tickers[:max_n]


def download_prices(tickers, start="2000-01-01", batch_size=200):
    all_prices = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i} to {i+len(batch)}")

        data = yf.download(
            batch,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="ticker"
        )

        if isinstance(data.columns, pd.MultiIndex):
            closes = data.loc[:, (slice(None), "Close")]
            closes.columns = closes.columns.get_level_values(0)
        else:
            closes = data["Close"]

        all_prices.append(closes)

        time.sleep(1)  # avoid rate limiting

    prices = pd.concat(all_prices, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    return prices


def main():
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/raw"
    raw_path.mkdir(parents=True, exist_ok=True)

    tickers = get_us_tickers(max_n=1200)
    prices = download_prices(tickers)
    
    prices.to_parquet(raw_path / "us_prices_daily.parquet")
    print("Saved raw prices.")


if __name__ == "__main__":
    main()