"""
fetch_watchlist.py
Fetches price + fundamental data for a watchlist of NSE stocks spanning
large, mid, and small cap segments. Validates data quality and applies
market-cap classification.

Run this from the project root:
    python src/fetch_watchlist.py
"""

import pandas as pd
import time
from data_fetcher import fetch_stock_data, get_stock_info


# Expanded watchlist -- mix of large/mid/small cap NSE stocks across sectors
# (banking, IT, FMCG, auto, pharma, energy, infra). ~50 stocks gives a
# realistic resume-scale universe while staying manageable to fetch.
WATCHLIST = [
    # --- Large cap ---
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "NESTLEIND.NS",

    # --- Mid cap ---
    "PERSISTENT.NS", "COFORGE.NS", "PAGEIND.NS", "AUBANK.NS", "PIIND.NS",
    "MPHASIS.NS", "ASTRAL.NS", "TRENT.NS", "POLYCAB.NS", "CUMMINSIND.NS",
    "SRF.NS", "APLAPOLLO.NS", "SUPREMEIND.NS", "COROMANDEL.NS", "GODREJPROP.NS",

    # --- Small cap ---
    "CAMS.NS", "ROUTE.NS", "HAPPSTMNDS.NS", "TANLA.NS", "LATENTVIEW.NS",
    "RATNAMANI.NS", "KFINTECH.NS", "JBCHEPHARM.NS", "GRAVITA.NS",
    "SIGNATURE.NS", "GALAXYSURF.NS", "ETHOSLTD.NS", "IEX.NS", "AAVAS.NS",
]


# Market cap thresholds (in INR). SEBI's official definitions rank companies
# by rank (top 100 = large cap, 101-250 = mid cap, rest = small cap), but
# for a resume project, fixed value bands are a reasonable simplification.
# These are rough 2026 approximations -- refine using an updated Nifty
# ranking list if you want to be more precise later.
LARGE_CAP_THRESHOLD = 1_000_000_000_000    # >= Rs 1,00,000 Cr (Rs 1 lakh crore)
MID_CAP_THRESHOLD = 250_000_000_000        # >= Rs 25,000 Cr


def classify_market_cap(market_cap):
    """Classify a stock into Large / Mid / Small cap based on market cap value."""
    if market_cap is None:
        return "Unknown"
    if market_cap >= LARGE_CAP_THRESHOLD:
        return "Large Cap"
    elif market_cap >= MID_CAP_THRESHOLD:
        return "Mid Cap"
    else:
        return "Small Cap"


def build_watchlist_summary(symbols: list) -> pd.DataFrame:
    """
    Fetches info for each stock in the watchlist and builds a summary
    DataFrame with market cap classification and key fundamentals.
    """
    rows = []
    for sym in symbols:
        print(f"Fetching info: {sym} ...")
        try:
            info = get_stock_info(sym)
            info["cap_category"] = classify_market_cap(info.get("marketCap"))
            rows.append(info)
        except Exception as e:
            print(f"  FAILED: {sym} -> {e}")
        time.sleep(0.5)  # be polite to the API, avoid rate limiting

    return pd.DataFrame(rows)


def validate_data_quality(df: pd.DataFrame):
    """Print a quick data quality report -- missing values, coverage, etc."""
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Total stocks fetched: {len(df)}")
    print(f"\nMissing values per column:\n{df.isna().sum()}")
    print(f"\nCap category breakdown:\n{df['cap_category'].value_counts()}")


if __name__ == "__main__":
    summary_df = build_watchlist_summary(WATCHLIST)

    print("\n=== WATCHLIST SUMMARY ===")
    print(summary_df[["symbol", "shortName", "sector", "marketCap",
                       "trailingPE", "cap_category"]].to_string(index=False))

    validate_data_quality(summary_df)

    # Save for later use in feature engineering
    summary_df.to_csv("data/raw/watchlist_summary.csv", index=False)
    print("\nSaved to data/raw/watchlist_summary.csv")