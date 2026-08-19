"""
build_training_dataset.py
Runs feature engineering across the ENTIRE watchlist and combines results
into a single training dataset for the short-term model.

Run from project root:
    python src/build_training_dataset.py
"""

import pandas as pd
import time

from data_fetcher import fetch_stock_data
from feature_engineering import add_technical_indicators, label_signal
from fetch_watchlist import WATCHLIST


def build_dataset_for_symbol(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch, engineer features, and label data for a single symbol."""
    raw_df = fetch_stock_data(symbol, period=period)
    featured_df = add_technical_indicators(raw_df)
    labeled_df = label_signal(featured_df, horizon_days=5, threshold=0.02)
    return labeled_df


def build_full_dataset(symbols: list, period: str = "1y") -> pd.DataFrame:
    """
    Loops through every symbol, builds features + labels, and concatenates
    into one big DataFrame. Skips failures instead of crashing the batch.
    """
    all_dfs = []
    failed = []

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Processing {sym} ...")
        try:
            df = build_dataset_for_symbol(sym, period=period)
            all_dfs.append(df)
        except Exception as e:
            print(f"  FAILED: {sym} -> {e}")
            failed.append(sym)
        time.sleep(0.3)  # avoid hammering the API

    if failed:
        print(f"\n{len(failed)} symbols failed: {failed}")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


if __name__ == "__main__":
    print(f"Building training dataset for {len(WATCHLIST)} stocks ...\n")

    dataset = build_full_dataset(WATCHLIST, period="1y")

    # Drop rows where the label is missing (last 5 days of each stock's
    # history -- can't know the future return for those yet)
    dataset_clean = dataset.dropna(subset=["signal_label"])

    print(f"\n=== DATASET SUMMARY ===")
    print(f"Total rows (all stocks, before dropping unlabeled): {len(dataset)}")
    print(f"Total labeled rows (usable for training): {len(dataset_clean)}")
    print(f"Stocks included: {dataset_clean['symbol'].nunique()}")

    print(f"\nLabel distribution (overall):")
    print(dataset_clean["signal_label"].value_counts())
    print(f"\nLabel distribution (%):")
    print((dataset_clean["signal_label"].value_counts(normalize=True) * 100).round(1))

    # Check for rows with too many NaN indicator values (early warm-up rows
    # that slipped through, or data issues)
    indicator_cols = ["RSI_14", "MACD", "SMA_20", "SMA_50", "ATR_14"]
    before = len(dataset_clean)
    dataset_final = dataset_clean.dropna(subset=indicator_cols)
    after = len(dataset_final)
    print(f"\nDropped {before - after} rows with incomplete indicators (warm-up period)")
    print(f"Final training dataset size: {after} rows")

    # Save
    dataset_final.to_csv("data/processed/training_dataset.csv", index=False)
    print("\nSaved to data/processed/training_dataset.csv")