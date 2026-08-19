"""
batch_fetch_universe.py
Fetches fundamentals for the FULL NSE universe (~2,400 stocks) -- not
feasible to do live on every dashboard load, so this runs as a separate
batch job (intended to run once daily, e.g. via Task Scheduler before
market open) and caches results to disk. The dashboard then reads from
this cache instantly instead of live-fetching thousands of stocks.

Uses a thread pool for concurrency (yfinance calls are I/O-bound, so
threads work well here) and checkpoints progress every N stocks so a
crash or interruption doesn't lose everything already fetched.

Run from project root:
    python src/batch_fetch_universe.py
"""

import pandas as pd
import time
import os
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

CHECKPOINT_FILE = "data/processed/universe_fundamentals.csv"
CHECKPOINT_EVERY = 50       # save progress every 50 stocks
MAX_WORKERS = 8             # concurrent threads -- keep modest to avoid rate limiting
REQUEST_DELAY = 0.15        # small delay per request even with threading


def load_universe() -> list:
    df = pd.read_csv("data/raw/nse_full_universe.csv")
    return df["yf_symbol"].tolist()


def load_existing_results() -> pd.DataFrame:
    """Resume from a previous run if a checkpoint file already exists."""
    if os.path.exists(CHECKPOINT_FILE):
        existing = pd.read_csv(CHECKPOINT_FILE)
        print(f"Found existing checkpoint with {len(existing)} stocks already fetched.")
        return existing
    return pd.DataFrame()


def fetch_one(symbol: str) -> dict:
    """Fetches the EXTENDED fundamentals set -- same fields the long-term
    scorer needs (ROE, debt/equity, growth, margins), so this one cache
    serves both the dashboard's basic info AND the long-term scoring."""
    time.sleep(REQUEST_DELAY)
    try:
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
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "earningsGrowth": info.get("earningsQuarterlyGrowth"),
            "revenueGrowth": info.get("revenueGrowth"),
            "profitMargins": info.get("profitMargins"),
            "dividendYield": info.get("dividendYield"),
            "status": "ok",
        }
    except Exception as e:
        return {"symbol": symbol, "status": "failed", "error": str(e)}


def batch_fetch(symbols: list, already_done: set) -> pd.DataFrame:
    remaining = [s for s in symbols if s not in already_done]
    print(f"Total symbols: {len(symbols)} | Already done: {len(already_done)} | Remaining: {len(remaining)}")

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in remaining}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if completed % 25 == 0:
                print(f"  Progress: {completed}/{len(remaining)}")

            # Checkpoint periodically
            if completed % CHECKPOINT_EVERY == 0:
                save_checkpoint(results)

    return pd.DataFrame(results)


def save_checkpoint(new_results: list):
    """Appends new results to the checkpoint file (creates it if absent)."""
    new_df = pd.DataFrame(new_results)
    if os.path.exists(CHECKPOINT_FILE):
        existing = pd.read_csv(CHECKPOINT_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="symbol", keep="last")
    else:
        combined = new_df
    combined.to_csv(CHECKPOINT_FILE, index=False)


if __name__ == "__main__":
    symbols = load_universe()
    existing_df = load_existing_results()
    already_done = set(existing_df["symbol"]) if not existing_df.empty else set()

    print(f"\nStarting batch fetch for full NSE universe ({len(symbols)} stocks) ...")
    print(f"Using {MAX_WORKERS} concurrent workers. This will take a while.\n")

    start = time.time()
    new_results_df = batch_fetch(symbols, already_done)
    elapsed = time.time() - start

    # Final save (catches any remainder not caught by periodic checkpoints)
    if not new_results_df.empty:
        save_checkpoint(new_results_df.to_dict("records"))

    final_df = pd.read_csv(CHECKPOINT_FILE)
    ok_count = (final_df["status"] == "ok").sum()
    failed_count = (final_df["status"] == "failed").sum()

    print(f"\n=== BATCH FETCH COMPLETE ===")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Successful: {ok_count}")
    print(f"Failed: {failed_count}")
    print(f"Total in checkpoint file: {len(final_df)}")
    print(f"\nSaved to {CHECKPOINT_FILE}")
    print("\nRe-run this script anytime to resume/refresh -- it skips symbols")
    print("already fetched and only retries failures or fetches new ones.")