"""
long_term_scorer.py
Builds a fundamentals-based "quality score" for long-term investment
suitability -- a different approach than the short-term technical model.

Unlike the short-term model, this is NOT a trained ML classifier predicting
a label. Long-term fundamental quality doesn't have a clean historical
"label" to train against (what counts as "good fundamentals" 3 years ago
that led to good 3-year returns is a much harder, noisier labeling problem
than 5-day price moves). Instead, we use a transparent, rule-based
WEIGHTED SCORING system -- similar to how real equity research screens
work (e.g. Piotroski F-Score, Graham-style screens).

This is a deliberate, defensible design choice worth explaining in
interviews: not everything needs a black-box model. A transparent scoring
system is often *more* appropriate and *more* trustworthy for fundamental
analysis, where investors want to understand exactly WHY a stock scored
well, not just get a number.

Run from project root:
    python src/long_term_scorer.py
"""

import pandas as pd
import numpy as np
import time

from data_fetcher import get_stock_info
from fetch_watchlist import WATCHLIST, classify_market_cap


def get_extended_fundamentals(symbol: str) -> dict:
    """Fetches a broader set of fundamental metrics than get_stock_info
    provides, needed for a proper long-term quality score."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "symbol": symbol,
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "priceToBook": info.get("priceToBook"),
        "returnOnEquity": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "earningsGrowth": info.get("earningsQuarterlyGrowth"),
        "revenueGrowth": info.get("revenueGrowth"),
        "profitMargins": info.get("profitMargins"),
        "dividendYield": info.get("dividendYield"),
        "currentPrice": info.get("currentPrice"),
    }


def score_stock(row: pd.Series, sector_medians: pd.DataFrame) -> dict:
    """
    Scores a single stock 0-100 across weighted fundamental factors,
    each compared against its SECTOR median (comparing a bank's P/E to
    an IT company's P/E is meaningless -- sector-relative comparison is
    the standard, correct approach in equity research).

    Weights (must sum to 100):
      - Valuation (P/E vs sector):        25
      - Profitability (ROE):              25
      - Financial health (Debt/Equity):   20
      - Growth (earnings growth):         20
      - Profit margins:                   10
    """
    sector = row.get("sector")
    sector_row = sector_medians.loc[sector] if sector in sector_medians.index else None

    score = 0
    breakdown = {}

    # --- Valuation: lower P/E relative to sector = better (cheaper) ---
    pe = row.get("trailingPE")
    if pd.notna(pe) and pe > 0 and sector_row is not None and pd.notna(sector_row["trailingPE"]):
        sector_pe = sector_row["trailingPE"]
        pe_score = np.clip(25 * (sector_pe / pe), 0, 25) if pe > 0 else 0
        pe_score = min(pe_score, 25)
    else:
        pe_score = 12.5  # neutral score if data missing
    score += pe_score
    breakdown["valuation_score"] = round(pe_score, 1)

    # --- Profitability: higher ROE = better ---
    roe = row.get("returnOnEquity")
    if pd.notna(roe):
        # ROE typically 0-0.30+ range; scale to 0-25
        roe_score = np.clip(roe / 0.30 * 25, 0, 25)
    else:
        roe_score = 12.5
    score += roe_score
    breakdown["profitability_score"] = round(roe_score, 1)

    # --- Financial health: lower Debt/Equity = better ---
    de = row.get("debtToEquity")
    if pd.notna(de):
        # D/E of 0 = full marks, D/E of 200+ = zero marks (yfinance reports as %, e.g. 45.2 = 0.452)
        de_score = np.clip(20 * (1 - de / 200), 0, 20)
    else:
        de_score = 10
    score += de_score
    breakdown["financial_health_score"] = round(de_score, 1)

    # --- Growth: higher earnings growth = better ---
    growth = row.get("earningsGrowth")
    if pd.notna(growth):
        growth_score = np.clip(20 * (growth / 0.25), 0, 20)  # 25%+ growth = full marks
    else:
        growth_score = 10
    score += growth_score
    breakdown["growth_score"] = round(growth_score, 1)

    # --- Margins: higher profit margin = better ---
    margin = row.get("profitMargins")
    if pd.notna(margin):
        margin_score = np.clip(10 * (margin / 0.20), 0, 10)  # 20%+ margin = full marks
    else:
        margin_score = 5
    score += margin_score
    breakdown["margin_score"] = round(margin_score, 1)

    breakdown["total_score"] = round(score, 1)
    return breakdown


def classify_recommendation(score: float, percentile: float) -> str:
    """
    Uses PERCENTILE rank within the watchlist rather than fixed absolute
    thresholds. Fixed thresholds (e.g. '70+ = Strong Buy') are arbitrary
    guesses that can make nearly every stock look like a buy if the whole
    universe scores high on average. Percentile ranking guarantees the
    labels are actually discriminating -- meaningful relative to peers,
    which is how real equity screens are typically built.
    """
    if percentile >= 80:
        return "Strong Buy (Long-Term)"
    elif percentile >= 55:
        return "Accumulate"
    elif percentile >= 25:
        return "Hold / Neutral"
    else:
        return "Avoid (Weak Fundamentals, Relative to Peers)"


def build_long_term_scores(symbols: list) -> pd.DataFrame:
    print(f"Fetching extended fundamentals for {len(symbols)} stocks ...")
    rows = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym} ...")
        try:
            rows.append(get_extended_fundamentals(sym))
        except Exception as e:
            print(f"  FAILED: {sym} -> {e}")
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    df["cap_category"] = df["marketCap"].apply(classify_market_cap)

    # Compute sector medians for relative scoring
    sector_medians = df.groupby("sector")[["trailingPE", "returnOnEquity", "debtToEquity"]].median()

    print("\nScoring stocks against sector medians ...")
    score_results = df.apply(lambda row: score_stock(row, sector_medians), axis=1)
    score_df = pd.DataFrame(list(score_results))

    final_df = pd.concat([df, score_df], axis=1)
    final_df["score_percentile"] = final_df["total_score"].rank(pct=True) * 100
    final_df["recommendation"] = final_df.apply(
        lambda row: classify_recommendation(row["total_score"], row["score_percentile"]), axis=1
    )

    return final_df.sort_values("total_score", ascending=False)


def build_long_term_scores_from_cache() -> pd.DataFrame:
    """Reads the pre-fetched universe cache instead of live-fetching --
    makes it feasible to score the full ~2,400 stock universe instantly,
    since the batch_fetch_universe.py job already did the slow part."""
    df = pd.read_csv("data/processed/universe_fundamentals.csv")
    df = df[df["status"] == "ok"].copy()
    df["cap_category"] = df["marketCap"].apply(classify_market_cap)

    sector_medians = df.groupby("sector")[["trailingPE", "returnOnEquity", "debtToEquity"]].median()

    print(f"Scoring {len(df)} stocks against sector medians ...")
    score_results = df.apply(lambda row: score_stock(row, sector_medians), axis=1)
    score_df = pd.DataFrame(list(score_results))

    final_df = pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
    final_df["score_percentile"] = final_df["total_score"].rank(pct=True) * 100
    final_df["recommendation"] = final_df.apply(
        lambda row: classify_recommendation(row["total_score"], row["score_percentile"]), axis=1
    )

    return final_df.sort_values("total_score", ascending=False)


if __name__ == "__main__":
    results = build_long_term_scores_from_cache()

    print("\n=== LONG-TERM FUNDAMENTAL SCORES (sorted best to worst) ===")
    display_cols = ["symbol", "sector", "cap_category", "total_score", "score_percentile", "recommendation"]
    print(results[display_cols].to_string(index=False))

    print("\n=== RECOMMENDATION DISTRIBUTION ===")
    print(results["recommendation"].value_counts())

    print("\n=== SCORE BREAKDOWN (top 5) ===")
    breakdown_cols = ["symbol", "valuation_score", "profitability_score",
                       "financial_health_score", "growth_score", "margin_score", "total_score"]
    print(results[breakdown_cols].head(5).to_string(index=False))

    results.to_csv("data/processed/long_term_scores.csv", index=False)
    print("\nSaved to data/processed/long_term_scores.csv")