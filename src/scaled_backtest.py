"""
scaled_backtest.py
Runs the same chronological, no-lookahead backtest as backtest.py, but
across the full 49-stock watchlist, then aggregates results by market
cap category (Large/Mid/Small) to check whether the "works better on
large/mid cap, weaker on small cap" pattern seen on 3 stocks actually
holds up statistically across a bigger sample.

Run from project root:
    python src/scaled_backtest.py
"""

import pandas as pd
import time

from fetch_watchlist import WATCHLIST, classify_market_cap
from backtest import backtest_symbol


def load_cap_mapping() -> dict:
    """Builds symbol -> cap_category using the cached universe fundamentals
    (already fetched for the full NSE universe, so the 49-stock watchlist
    is a subset of it)."""
    try:
        df = pd.read_csv("data/processed/universe_fundamentals.csv")
        df["cap_category"] = df["marketCap"].apply(classify_market_cap)
        return dict(zip(df["symbol"], df["cap_category"]))
    except FileNotFoundError:
        print("universe_fundamentals.csv not found -- cap categories will be 'Unknown'.")
        return {}


if __name__ == "__main__":
    cap_mapping = load_cap_mapping()

    all_rows = []
    failed = []

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"\n[{i}/{len(WATCHLIST)}] {symbol}")
        try:
            _, strat_metrics, bh_metrics = backtest_symbol(symbol, period="3y", test_fraction=0.25)
            cap_cat = cap_mapping.get(symbol, "Unknown")

            strat_metrics.update({"symbol": symbol, "type": "Strategy", "cap_category": cap_cat})
            bh_metrics.update({"symbol": symbol, "type": "Buy & Hold", "cap_category": cap_cat})
            all_rows.append(strat_metrics)
            all_rows.append(bh_metrics)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(symbol)
        time.sleep(0.3)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("data/processed/scaled_backtest_results.csv", index=False)

    print(f"\n\n{'='*70}")
    print(f"BACKTEST COMPLETE -- {len(WATCHLIST) - len(failed)}/{len(WATCHLIST)} stocks succeeded")
    if failed:
        print(f"Failed: {failed}")

    # --- Aggregate by cap category ---
    print(f"\n=== AVERAGE METRICS BY CAP CATEGORY ===\n")
    metrics_to_avg = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]

    summary = (
        results_df.groupby(["cap_category", "type"])[metrics_to_avg]
        .mean()
        .round(2)
        .reset_index()
    )
    print(summary.to_string(index=False))

    # --- Strategy vs Buy & Hold win count (how often did strategy beat B&H on Sharpe?) ---
    pivot = results_df.pivot_table(
        index=["symbol", "cap_category"], columns="type", values="sharpe_ratio"
    ).reset_index()
    pivot["strategy_wins"] = pivot["Strategy"] > pivot["Buy & Hold"]

    print(f"\n=== HOW OFTEN DID THE STRATEGY BEAT BUY & HOLD ON SHARPE RATIO? ===")
    win_summary = pivot.groupby("cap_category")["strategy_wins"].agg(["sum", "count"])
    win_summary["win_pct"] = (win_summary["sum"] / win_summary["count"] * 100).round(1)
    print(win_summary)

    summary.to_csv("data/processed/backtest_summary_by_cap.csv", index=False)
    print(f"\nSaved detailed results to data/processed/scaled_backtest_results.csv")
    print(f"Saved cap-category summary to data/processed/backtest_summary_by_cap.csv")