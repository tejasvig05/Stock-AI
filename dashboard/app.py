"""
app.py
Main Streamlit dashboard -- the user-facing deliverable of the project.

Run from project root:
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_fetcher import fetch_stock_data, get_stock_info
from feature_engineering import add_technical_indicators


st.set_page_config(page_title="AI Stock Analysis Dashboard", layout="wide")

FEATURE_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_upper", "BB_lower", "BB_width", "ATR_14",
    "daily_return", "volatility_20", "price_vs_SMA20",
]

@st.cache_data(ttl=3600)
def load_universe_list():
    """Loads the full stock universe from the batch-fetched cache.
    Falls back to a small default list if the cache doesn't exist yet."""
    try:
        df = pd.read_csv("data/processed/universe_fundamentals.csv")
        df = df[df["status"] == "ok"]
        # Sort alphabetically for a usable dropdown; Streamlit's selectbox
        # is searchable by typing, so order mainly matters for browsing
        return sorted(df["symbol"].dropna().unique().tolist())
    except FileNotFoundError:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]


WATCHLIST = load_universe_list()

LARGE_CAP_THRESHOLD = 1_000_000_000_000
MID_CAP_THRESHOLD = 250_000_000_000


def classify_market_cap(market_cap):
    if market_cap is None:
        return "Unknown"
    if market_cap >= LARGE_CAP_THRESHOLD:
        return "Large Cap"
    elif market_cap >= MID_CAP_THRESHOLD:
        return "Mid Cap"
    else:
        return "Small Cap"


@st.cache_data(ttl=3600)
def load_long_term_scores():
    try:
        return pd.read_csv("data/processed/long_term_scores.csv")
    except FileNotFoundError:
        return None


def get_long_term_score(scores_df, symbol):
    if scores_df is None:
        return None
    row = scores_df[scores_df["symbol"] == symbol]
    if row.empty:
        return None
    return row.iloc[0]


@st.cache_resource
def load_model():
    model = joblib.load("data/processed/rf_model.pkl")
    scaler = joblib.load("data/processed/scaler.pkl")
    reg_model = joblib.load("data/processed/rf_regressor.pkl")
    reg_scaler = joblib.load("data/processed/scaler_reg.pkl")
    return model, scaler, reg_model, reg_scaler


@st.cache_data(ttl=900)
def load_stock_data(symbol: str):
    """Fetches PRICE data live (needed fresh for charts/signals), but
    pulls company info from the cached universe file instead of making
    a second live yfinance call -- halves live API calls per interaction,
    which matters a lot given yfinance's rate limits at this scale."""
    raw_df = fetch_stock_data(symbol, period="1y")
    featured_df = add_technical_indicators(raw_df)
    info = get_cached_info(symbol)
    return featured_df, info


@st.cache_data(ttl=3600)
def get_cached_info(symbol: str) -> dict:
    """Looks up company info from the batch-fetched cache. Falls back to
    a live call only if the symbol isn't in the cache for some reason."""
    try:
        cache_df = pd.read_csv("data/processed/universe_fundamentals.csv")
        row = cache_df[cache_df["symbol"] == symbol]
        if not row.empty:
            return row.iloc[0].to_dict()
    except FileNotFoundError:
        pass
    # Fallback: live fetch (only hit if not in cache)
    return get_stock_info(symbol)


def get_recommendation(featured_df: pd.DataFrame, model, scaler):
    """Runs the latest row of features through the model to get a signal."""
    latest = featured_df.iloc[-1:][FEATURE_COLUMNS]
    if latest.isna().any(axis=1).values[0]:
        return None, None
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    proba = model.predict_proba(latest_scaled)[0]
    proba_dict = dict(zip(model.classes_, proba))
    return pred, proba_dict


def get_expected_return(featured_df: pd.DataFrame, reg_model, reg_scaler):
    """Predicts expected % return over the 5-day horizon using the
    regression model. Returns None if features aren't ready yet."""
    latest = featured_df.iloc[-1:][FEATURE_COLUMNS]
    if latest.isna().any(axis=1).values[0]:
        return None
    latest_scaled = reg_scaler.transform(latest)
    predicted_return = reg_model.predict(latest_scaled)[0]
    return predicted_return


def plot_price_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol} Price & Moving Averages", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"], name="Signal", line=dict(color="orange")), row=3, col=1)
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_diff"], name="Histogram"), row=3, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True)
    return fig


# ============ SIDEBAR ============
st.sidebar.title("Stock AI Dashboard")
selected_symbol = st.sidebar.selectbox("Select a stock", WATCHLIST)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:** This tool is for educational and informational "
    "purposes only and does not constitute financial advice. Stock market "
    "investments are subject to market risk. Please consult a certified "
    "financial advisor before making investment decisions. Past performance "
    "and model outputs do not guarantee future returns."
)

# ============ MAIN ============
st.title("AI-Powered Stock Analysis")

with st.spinner(f"Loading data for {selected_symbol} ..."):
    featured_df, info = load_stock_data(selected_symbol)
    model, scaler, reg_model, reg_scaler = load_model()

cap_category = classify_market_cap(info.get("marketCap"))

# --- Top info row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"₹{info.get('currentPrice', 'N/A')}")
col2.metric("Market Cap Category", cap_category)
col3.metric("Sector", info.get("sector", "N/A"))
col4.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else "N/A")

st.markdown("---")

# --- Recommendation card ---
pred, proba = get_recommendation(featured_df, model, scaler)
expected_return = get_expected_return(featured_df, reg_model, reg_scaler)

st.subheader("Short-Term Signal (Next 5 Trading Days)")
if pred is None:
    st.warning("Not enough recent data to generate a signal for this stock right now.")
else:
    signal_colors = {"Up": "🟢", "Down": "🔴", "Sideways": "🟡"}
    rec_col1, rec_col2 = st.columns([1, 2])
    with rec_col1:
        st.markdown(f"### {signal_colors.get(pred, '')} {pred}")
        st.caption(f"Model confidence: {proba[pred]*100:.1f}%")

        if expected_return is not None:
            NEGLIGIBLE_THRESHOLD = 0.005  # +/- 0.5% -- below this, don't claim a confident direction
            if abs(expected_return) < NEGLIGIBLE_THRESHOLD:
                verdict = "Uncertain / Negligible Move"
                verdict_color = "orange"
            elif expected_return > 0:
                verdict = "Likely Profit"
                verdict_color = "green"
            else:
                verdict = "Likely Loss"
                verdict_color = "red"
            st.markdown(
                f"**Expected return (5 days): "
                f":{verdict_color}[{expected_return*100:+.2f}%]**"
            )
            st.markdown(f"**Verdict: :{verdict_color}[{verdict}]**")
    with rec_col2:
        proba_df = pd.DataFrame(list(proba.items()), columns=["Signal", "Probability"])
        proba_df["Probability"] = proba_df["Probability"] * 100
        st.bar_chart(proba_df.set_index("Signal"))

st.caption(
    "The signal is generated by a Random Forest classifier; the expected "
    "return is a separate regression model, both trained on historical "
    "technical indicators. These reflect statistical patterns in past data, "
    "not guarantees -- stock return prediction has inherently low precision "
    "(see model evaluation notes), so treat this as one input among many, "
    "not a definitive answer."
)

st.markdown("---")

# --- Long-term fundamentals section ---
st.subheader("Long-Term Signal (Fundamentals-Based)")
long_term_scores = load_long_term_scores()
lt_row = get_long_term_score(long_term_scores, selected_symbol)

if lt_row is None:
    st.info(
        "Long-term score not available for this stock yet. Run "
        "`python src/long_term_scorer.py` to generate scores for the full watchlist."
    )
else:
    lt_col1, lt_col2 = st.columns([1, 2])
    with lt_col1:
        rec = lt_row["recommendation"]
        rec_colors = {
            "Strong Buy (Long-Term)": "green",
            "Accumulate": "blue",
            "Hold / Neutral": "orange",
            "Avoid (Weak Fundamentals, Relative to Peers)": "red",
        }
        color = rec_colors.get(rec, "gray")
        st.markdown(f"### :{color}[{rec}]")
        st.metric("Fundamental Score", f"{lt_row['total_score']:.1f} / 100")
        st.caption(f"Ranks in the {lt_row['score_percentile']:.0f}th percentile of the watchlist")
    with lt_col2:
        breakdown = pd.DataFrame({
            "Factor": ["Valuation", "Profitability", "Financial Health", "Growth", "Margins"],
            "Score": [
                lt_row["valuation_score"], lt_row["profitability_score"],
                lt_row["financial_health_score"], lt_row["growth_score"], lt_row["margin_score"],
            ],
            "Max": [25, 25, 20, 20, 10],
        })
        st.bar_chart(breakdown.set_index("Factor")[["Score"]])

    st.caption(
        "This score compares the stock's valuation, profitability, financial health, "
        "growth, and margins against sector peers and the rest of the watchlist. "
        "It's a transparent rule-based score, not a machine-learned prediction -- "
        "useful for understanding *why* a stock ranks where it does, but still not "
        "a substitute for full due diligence."
    )

st.markdown("---")

# --- Backtest / historical performance section ---
st.subheader("Historical Strategy Performance (Backtest)")

backtest_file = f"data/processed/backtest_{selected_symbol.replace('.NS','')}.csv"
try:
    bt_df = pd.read_csv(backtest_file)
    bt_df["Date"] = pd.to_datetime(bt_df["Date"])

    bt_col1, bt_col2 = st.columns([2, 1])
    with bt_col1:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=bt_df["Date"], y=(bt_df["strategy_equity"] - 1) * 100,
            name="Model Strategy", line=dict(color="#2ca02c"),
        ))
        equity_fig.add_trace(go.Scatter(
            x=bt_df["Date"], y=(bt_df["buyhold_equity"] - 1) * 100,
            name="Buy & Hold", line=dict(color="gray", dash="dash"),
        ))
        equity_fig.update_layout(
            title="Cumulative Return: Strategy vs. Buy & Hold (Out-of-Sample Test Period)",
            yaxis_title="Return (%)", height=350,
        )
        st.plotly_chart(equity_fig, use_container_width=True)
    with bt_col2:
        final_strat = (bt_df["strategy_equity"].iloc[-1] - 1) * 100
        final_bh = (bt_df["buyhold_equity"].iloc[-1] - 1) * 100
        st.metric("Strategy Return", f"{final_strat:+.2f}%")
        st.metric("Buy & Hold Return", f"{final_bh:+.2f}%",
                   delta=f"{final_strat - final_bh:+.2f}% vs strategy" if final_strat != final_bh else None)
        active_days = (bt_df["position"] == 1).sum()
        st.caption(f"Strategy was active (holding a position) {active_days} of {len(bt_df)} test days.")

    st.caption(
        "This shows a backtest on a HISTORICAL out-of-sample period the model was not "
        "trained on -- not a live track record. Across the full 49-stock watchlist, this "
        "strategy beat buy-and-hold on risk-adjusted return (Sharpe ratio) roughly 35-52% "
        "of the time depending on cap category, and typically reduced maximum drawdown "
        "substantially. It has not shown a consistent return advantage. Full methodology "
        "and results: see project documentation."
    )
except FileNotFoundError:
    st.info(
        f"No backtest on record for {selected_symbol} yet. Run "
        f"`python src/backtest.py` (or add it to `scaled_backtest.py`'s list) to generate one."
    )

st.markdown("---")
st.subheader("Price Chart & Technical Indicators")
st.plotly_chart(plot_price_chart(featured_df, selected_symbol), use_container_width=True)

# --- Fundamentals table ---
st.subheader("Fundamentals")
fund_df = pd.DataFrame({
    "Metric": ["Sector", "Industry", "Market Cap", "P/E Ratio", "Cap Category"],
    "Value": [
        info.get("sector", "N/A"),
        info.get("industry", "N/A"),
        f"₹{info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A",
        f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else "N/A",
        cap_category,
    ]
})
st.table(fund_df)

# --- Recent data table ---
with st.expander("View recent price data"):
    st.dataframe(
        featured_df[["Date", "Open", "High", "Low", "Close", "Volume", "RSI_14", "MACD"]].tail(20),
        use_container_width=True,
    )