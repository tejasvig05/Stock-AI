"""
feature_engineering.py
Computes technical indicators on historical price data -- these become the
input features for the short-term prediction model.

Run from project root:
    python src/feature_engineering.py
"""

import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from data_fetcher import fetch_stock_data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicator columns to a price DataFrame.
    Expects columns: Open, High, Low, Close, Volume (yfinance format).
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- Moving averages ---
    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["EMA_20"] = EMAIndicator(close, window=20).ema_indicator()

    # --- RSI (momentum) ---
    df["RSI_14"] = RSIIndicator(close, window=14).rsi()

    # --- MACD (trend/momentum) ---
    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()  # histogram

    # --- Bollinger Bands (volatility) ---
    bb = BollingerBands(close, window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # --- Average True Range (volatility) ---
    df["ATR_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # --- Simple derived features ---
    df["daily_return"] = close.pct_change()
    df["volatility_20"] = df["daily_return"].rolling(window=20).std()
    df["price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]  # % above/below trend

    return df


def label_signal(df: pd.DataFrame, horizon_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    Creates a simple target label for short-term classification:
    looks `horizon_days` ahead and labels the move as Up / Down / Sideways
    based on `threshold` (e.g. 2% move).

    NOTE: this label uses FUTURE data (shifted backwards) purely for
    training purposes. Never use this column as an input feature -- it's
    the label the model is trying to predict.
    """
    df = df.copy()
    future_return = df["Close"].shift(-horizon_days) / df["Close"] - 1

    def classify(r):
        if pd.isna(r):
            return None
        if r > threshold:
            return "Up"
        elif r < -threshold:
            return "Down"
        else:
            return "Sideways"

    df["future_return"] = future_return
    df["signal_label"] = future_return.apply(classify)
    return df


if __name__ == "__main__":
    symbol = "RELIANCE.NS"

    print(f"Fetching price data for {symbol} ...")
    raw_df = fetch_stock_data(symbol, period="1y")

    print("Computing technical indicators ...")
    featured_df = add_technical_indicators(raw_df)

    print("Generating training labels ...")
    labeled_df = label_signal(featured_df, horizon_days=5, threshold=0.02)

    print(f"\nColumns generated:\n{list(labeled_df.columns)}")
    print(f"\nSample rows (most recent, indicators need warm-up so early rows have NaNs):")
    print(labeled_df[["Date", "Close", "RSI_14", "MACD", "SMA_20", "signal_label"]].tail(10))

    print(f"\nLabel distribution:")
    print(labeled_df["signal_label"].value_counts())

    # Save
    labeled_df.to_csv("data/processed/RELIANCE_features.csv", index=False)
    print("\nSaved to data/processed/RELIANCE_features.csv")