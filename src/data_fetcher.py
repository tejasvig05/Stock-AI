"""
data_fetcher.py
Fetches historical OHLCV (Open, High, Low, Close, Volume) data for stocks
using yfinance. This is the foundation of the short-term technical pipeline.
"""

import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta


def fetch_stock_data(symbol: str, period: str = "1y", interval: str = "1d",
                      max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch historical price data for a single stock.

    Retries with exponential backoff on rate-limit/transient errors --
    yfinance can intermittently throttle under heavy usage, and a single
    failed call shouldn't crash the whole dashboard.

    Parameters
    ----------
    symbol : str
        Stock ticker with exchange suffix, e.g. 'RELIANCE.NS' for NSE,
        'TCS.BO' for BSE.
    period : str
        How far back to fetch data. Options: '1mo','3mo','6mo','1y','2y','5y','max'
    interval : str
        Data granularity. Options: '1d','1wk','1mo' (intraday needs a paid/other source)
    max_retries : int
        Number of retry attempts on failure before giving up.

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}. Check the ticker symbol.")

            df = df.reset_index()
            df["symbol"] = symbol
            return df
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  Retry {attempt+1}/{max_retries} for {symbol} after {wait}s ({e})")
                time.sleep(wait)

    raise last_error


def fetch_multiple_stocks(symbols: list, period: str = "1y", interval: str = "1d") -> dict:
    """
    Fetch data for a list of stocks. Returns a dict of {symbol: DataFrame}.
    Skips and logs any symbol that fails, instead of crashing the whole batch.
    """
    data = {}
    for sym in symbols:
        try:
            print(f"Fetching {sym} ...")
            data[sym] = fetch_stock_data(sym, period=period, interval=interval)
        except Exception as e:
            print(f"  FAILED: {sym} -> {e}")
    return data


def get_stock_info(symbol: str) -> dict:
    """
    Fetch basic company info -- useful for market cap classification later.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "symbol": symbol,
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "currentPrice": info.get("currentPrice"),
        "trailingPE": info.get("trailingPE"),
    }

def get_live_price(symbol: str):
    """
    Fetch the latest available market price for a stock.
    """

    try:
        ticker = yf.Ticker(symbol)

        # fast_info is faster and more suitable for latest price
        price = ticker.fast_info.get("last_price")

        if price is not None:
            return float(price)

        # Fallback if fast_info fails
        df = ticker.history(period="1d", interval="1m")

        if not df.empty:
            return float(df["Close"].iloc[-1])

        return None

    except Exception as e:
        print(f"Could not fetch live price for {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Quick validation run -- start with ONE stock to confirm the pipeline works
    test_symbol = "RELIANCE.NS"

    print(f"--- Testing single stock fetch: {test_symbol} ---")
    df = fetch_stock_data(test_symbol, period="6mo")
    print(df.tail())
    print(f"\nRows fetched: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    print(f"\n--- Testing company info fetch ---")
    info = get_stock_info(test_symbol)
    for k, v in info.items():
        print(f"{k}: {v}")

    # Save a sample to check output format
    df.to_csv("data/raw/RELIANCE_sample.csv", index=False)
    print("\nSaved sample to data/raw/RELIANCE_sample.csv")