"""
fetch_nse_universe.py
Downloads the official, complete list of NSE-listed equities and prepares
it as the full stock universe for the project (replacing the hand-picked
49-stock watchlist).

NSE publishes this as a public CSV -- no scraping/auth needed.

Run from project root:
    python src/fetch_nse_universe.py
"""

import pandas as pd
import requests
import io

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def fetch_nse_universe() -> pd.DataFrame:
    """
    Downloads NSE's official equity list CSV. NSE requires browser-like
    headers AND session cookies from visiting the main site first --
    hitting the archive URL cold often gets rejected.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,*/*",
    }
    session = requests.Session()
    session.headers.update(headers)

    # Visit the main site first to pick up session cookies -- NSE often
    # rejects direct archive requests without them
    print("Establishing NSE session ...")
    session.get("https://www.nseindia.com", timeout=15)

    print(f"Fetching NSE equity list from {NSE_EQUITY_LIST_URL} ...")
    response = session.get(NSE_EQUITY_LIST_URL, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df.columns = [c.strip() for c in df.columns]
    return df


if __name__ == "__main__":
    try:
        nse_df = fetch_nse_universe()
    except Exception as e:
        print(f"Failed to fetch NSE equity list: {e}")
        print("\nIf this fails repeatedly, NSE occasionally blocks datacenter/")
        print("automated requests. Fallback: manually download the CSV from")
        print("https://www.nseindia.com/market-data/securities-available-for-trading")
        print("and place it at data/raw/EQUITY_L.csv, then re-run this script.")
        raise SystemExit(1)

    print(f"\nTotal NSE-listed equities found: {len(nse_df)}")
    print(f"Columns: {list(nse_df.columns)}")
    print(nse_df.head())

    # Build yfinance-compatible symbols (NSE symbol + .NS suffix)
    nse_df["yf_symbol"] = nse_df["SYMBOL"].str.strip() + ".NS"

    nse_df.to_csv("data/raw/nse_full_universe.csv", index=False)
    print(f"\nSaved full universe to data/raw/nse_full_universe.csv")
    print(f"({len(nse_df)} stocks -- this becomes the new full watchlist)")