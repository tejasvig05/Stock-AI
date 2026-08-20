"""
app.py
AI-Powered Stock Analysis Dashboard -- redesigned UI/UX pass.

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
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_fetcher import fetch_stock_data, get_stock_info
from feature_engineering import add_technical_indicators


st.set_page_config(
    page_title="Stock AI | Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_upper", "BB_lower", "BB_width", "ATR_14",
    "daily_return", "volatility_20", "price_vs_SMA20",
]

FEATURE_DISPLAY_NAMES = {
    "SMA_20": "20-day Moving Avg", "SMA_50": "50-day Moving Avg", "EMA_20": "20-day EMA",
    "RSI_14": "RSI (14)", "MACD": "MACD", "MACD_signal": "MACD Signal Line",
    "MACD_diff": "MACD Histogram", "BB_upper": "Bollinger Upper Band",
    "BB_lower": "Bollinger Lower Band", "BB_width": "Bollinger Band Width",
    "ATR_14": "Avg True Range (14)", "daily_return": "Daily Return",
    "volatility_20": "20-day Volatility", "price_vs_SMA20": "Price vs 20-day Avg (%)",
}

LARGE_CAP_THRESHOLD = 1_000_000_000_000
MID_CAP_THRESHOLD = 250_000_000_000

# ============================================================
# THEME / CSS
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #131a2b 0%, #0b0f1a 45%, #090c14 100%);
}

section[data-testid="stSidebar"] {
    background: #0e1420;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7dd3fc 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #8b93a7;
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}

.metric-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(6px);
}
.metric-label {
    color: #8b93a7;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin-bottom: 4px;
}
.metric-value {
    color: #f1f4fa;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
}
.badge-up { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.35); }
.badge-down { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.35); }
.badge-sideways { background: rgba(234,179,8,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.35); }
.badge-strongbuy { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.35); }
.badge-accumulate { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.35); }
.badge-hold { background: rgba(234,179,8,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.35); }
.badge-avoid { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.35); }

.signal-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 8px;
}

.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e5e9f2;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

button[data-baseweb="tab"] {
    font-weight: 600;
}

.disclaimer-box {
    background: rgba(250, 204, 21, 0.06);
    border: 1px solid rgba(250, 204, 21, 0.2);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.78rem;
    color: #c9cddb;
    line-height: 1.5;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def classify_market_cap(market_cap):
    if market_cap is None or pd.isna(market_cap):
        return "Unknown"
    if market_cap >= LARGE_CAP_THRESHOLD:
        return "Large Cap"
    elif market_cap >= MID_CAP_THRESHOLD:
        return "Mid Cap"
    else:
        return "Small Cap"


@st.cache_data(ttl=3600)
def load_universe_list():
    try:
        df = pd.read_csv("data/processed/universe_fundamentals.csv")
        df = df[df["status"] == "ok"]
        return sorted(df["symbol"].dropna().unique().tolist())
    except FileNotFoundError:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]


WATCHLIST = load_universe_list()


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


@st.cache_data(ttl=3600)
def get_cached_info(symbol: str) -> dict:
    try:
        cache_df = pd.read_csv("data/processed/universe_fundamentals.csv")
        row = cache_df[cache_df["symbol"] == symbol]
        if not row.empty:
            return row.iloc[0].to_dict()
    except FileNotFoundError:
        pass
    return get_stock_info(symbol)


@st.cache_data(ttl=900)
def load_stock_data(symbol: str):
    raw_df = fetch_stock_data(symbol, period="1y")
    featured_df = add_technical_indicators(raw_df)
    info = get_cached_info(symbol)
    return featured_df, info


@st.cache_resource
def load_model():
    model = joblib.load("data/processed/rf_model.pkl")
    scaler = joblib.load("data/processed/scaler.pkl")
    reg_model = joblib.load("data/processed/rf_regressor.pkl")
    reg_scaler = joblib.load("data/processed/scaler_reg.pkl")
    return model, scaler, reg_model, reg_scaler


def get_recommendation(featured_df, model, scaler):
    latest = featured_df.iloc[-1:][FEATURE_COLUMNS]
    if latest.isna().any(axis=1).values[0]:
        return None, None
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    proba = model.predict_proba(latest_scaled)[0]
    return pred, dict(zip(model.classes_, proba))


def get_expected_return(featured_df, reg_model, reg_scaler):
    latest = featured_df.iloc[-1:][FEATURE_COLUMNS]
    if latest.isna().any(axis=1).values[0]:
        return None
    latest_scaled = reg_scaler.transform(latest)
    return reg_model.predict(latest_scaled)[0]


@st.cache_resource
def get_shap_explainer(_model):
    """TreeExplainer is exact and fast for Random Forest -- no sampling
    needed, unlike model-agnostic SHAP methods. Cached as a resource since
    building the explainer has some one-time cost."""
    return shap.TreeExplainer(_model)


def get_shap_breakdown(explainer, model, latest_scaled, predicted_class, feature_names):
    """
    Returns a DataFrame of each feature's SHAP contribution toward the
    PREDICTED class specifically -- i.e. "how much did this feature push
    the model toward saying Down/Up/Sideways", not toward some other class.
    """
    shap_values = explainer.shap_values(latest_scaled)
    class_idx = list(model.classes_).index(predicted_class)

    # TreeExplainer for multiclass RF can return either a list of per-class
    # arrays or a single 3D array depending on shap/sklearn version -- handle both
    if isinstance(shap_values, list):
        values = shap_values[class_idx][0]
    else:
        values = np.asarray(shap_values)[0, :, class_idx]

    df = pd.DataFrame({
        "feature": feature_names,
        "display_name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
        "shap_value": values,
    })
    df["direction"] = np.where(df["shap_value"] >= 0, "Toward " + predicted_class, "Against " + predicted_class)
    df["abs_value"] = df["shap_value"].abs()
    return df.sort_values("abs_value", ascending=False)


def badge_html(text, css_class):
    return f'<span class="badge {css_class}">{text}</span>'


def metric_card(label, value):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def plot_price_chart(df, symbol):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol} Price & Moving Averages", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20",
                              line=dict(width=1.3, color="#38bdf8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50",
                              line=dict(width=1.3, color="#a78bfa")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI",
                              line=dict(color="#f472b6")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f87171", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#4ade80", row=2, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD",
                              line=dict(color="#38bdf8")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"], name="Signal",
                              line=dict(color="#facc15")), row=3, col=1)
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_diff"], name="Histogram",
                          marker_color="rgba(167,139,250,0.5)"), row=3, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False, showlegend=True,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem;">📈 Stock AI</div>', unsafe_allow_html=True)
    st.caption(f"Live coverage of {len(WATCHLIST):,} NSE-listed stocks")
    st.markdown("")

    selected_symbol = st.selectbox("Select a stock", WATCHLIST)

    st.markdown("")
    with st.expander("⚠️ Disclaimer", expanded=False):
        st.markdown(
            """<div class="disclaimer-box">
            This tool is for educational and informational purposes only and does not
            constitute financial advice. Stock market investments are subject to market risk.
            Please consult a certified financial advisor before making investment decisions.
            Past performance and model outputs do not guarantee future returns.
            </div>""",
            unsafe_allow_html=True,
        )

# ============================================================
# MAIN
# ============================================================
with st.spinner(f"Loading {selected_symbol}..."):
    featured_df, info = load_stock_data(selected_symbol)
    model, scaler, reg_model, reg_scaler = load_model()

cap_category = classify_market_cap(info.get("marketCap"))
company_name = info.get("shortName", selected_symbol)

st.markdown(f'<div class="hero-title">{company_name}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">{selected_symbol} &nbsp;•&nbsp; {info.get("sector", "N/A")} &nbsp;•&nbsp; {cap_category}</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(metric_card("Current Price", f"₹{info.get('currentPrice', 'N/A')}"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Market Cap", cap_category), unsafe_allow_html=True)
with m3:
    pe = info.get("trailingPE")
    st.markdown(metric_card("P/E Ratio", f"{pe:.2f}" if pe else "N/A"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card("Sector", info.get("sector", "N/A")), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

pred, proba = get_recommendation(featured_df, model, scaler)
expected_return = get_expected_return(featured_df, reg_model, reg_scaler)
long_term_scores = load_long_term_scores()
lt_row = get_long_term_score(long_term_scores, selected_symbol)

tab_overview, tab_charts, tab_longterm, tab_data = st.tabs(
    ["🎯 Signals", "📊 Charts & Indicators", "💰 Long-Term Score", "📋 Raw Data"]
)

# ============================================================
# TAB: SIGNALS
# ============================================================
with tab_overview:
    col_short, col_long = st.columns(2)

    with col_short:
        st.markdown('<div class="section-header">⚡ Short-Term Signal (5 days)</div>', unsafe_allow_html=True)
        if pred is None:
            st.warning("Not enough recent data to generate a signal.")
        else:
            badge_map = {"Up": "badge-up", "Down": "badge-down", "Sideways": "badge-sideways"}
            card_html = f'<div class="signal-card">{badge_html(pred, badge_map.get(pred))}'
            card_html += (
                f'<div style="margin-top:10px; color:#8b93a7; font-size:0.85rem;">'
                f'Model confidence: {proba[pred]*100:.1f}%</div>'
            )
            if expected_return is not None:
                NEG_THRESH = 0.005
                if abs(expected_return) < NEG_THRESH:
                    verdict, vcolor = "Uncertain / Negligible", "#facc15"
                elif expected_return > 0:
                    verdict, vcolor = "Likely Profit", "#4ade80"
                else:
                    verdict, vcolor = "Likely Loss", "#f87171"
                card_html += (
                    f'<div style="margin-top:14px; font-size:1.1rem;">'
                    f'Expected return: <b style="color:{vcolor}">{expected_return*100:+.2f}%</b></div>'
                    f'<div style="color:{vcolor}; font-weight:600;">{verdict}</div>'
                )
            card_html += '</div>'
            st.markdown(card_html, unsafe_allow_html=True)

            proba_df = pd.DataFrame(list(proba.items()), columns=["Signal", "Probability"])
            proba_df["Probability"] = proba_df["Probability"] * 100
            st.bar_chart(proba_df.set_index("Signal"))

            # --- SHAP explainability ---
            with st.expander("🔍 Why this signal? (Model explainability)"):
                try:
                    latest = featured_df.iloc[-1:][FEATURE_COLUMNS]
                    latest_scaled = scaler.transform(latest)
                    explainer = get_shap_explainer(model)
                    shap_df = get_shap_breakdown(explainer, model, latest_scaled, pred, FEATURE_COLUMNS)
                    top_features = shap_df.head(8)

                    shap_fig = go.Figure()
                    colors = ["#4ade80" if v >= 0 else "#f87171" for v in top_features["shap_value"]]
                    shap_fig.add_trace(go.Bar(
                        x=top_features["shap_value"],
                        y=top_features["display_name"],
                        orientation="h",
                        marker_color=colors,
                    ))
                    shap_fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=340, margin=dict(t=10, b=10, l=10, r=10),
                        xaxis_title=f"Contribution toward '{pred}'",
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(shap_fig, use_container_width=True)
                    st.caption(
                        f"Green bars pushed the model TOWARD predicting '{pred}'; red bars pushed AGAINST it. "
                        "Values are SHAP contributions from a TreeExplainer on the Random Forest -- this shows "
                        "which indicators drove today's specific prediction, not general feature importance."
                    )
                except Exception as e:
                    st.caption(f"Explainability unavailable for this prediction ({e}).")

        st.caption(
            "Random Forest classifier + regressor trained on technical indicators. "
            "Statistical pattern, not a guarantee -- one input among many."
        )

    with col_long:
        st.markdown('<div class="section-header">🏛️ Long-Term Signal (Fundamentals)</div>', unsafe_allow_html=True)
        if lt_row is None:
            st.info("Long-term score not available for this stock yet.")
        else:
            rec = lt_row["recommendation"]
            badge_map_lt = {
                "Strong Buy (Long-Term)": "badge-strongbuy",
                "Accumulate": "badge-accumulate",
                "Hold / Neutral": "badge-hold",
                "Avoid (Weak Fundamentals, Relative to Peers)": "badge-avoid",
            }
            lt_card = (
                f'<div class="signal-card">{badge_html(rec, badge_map_lt.get(rec, "badge-hold"))}'
                f'<div style="margin-top:14px; font-size:1.8rem; font-weight:800; font-family:JetBrains Mono, monospace;">'
                f'{lt_row["total_score"]:.1f}<span style="font-size:1rem; color:#8b93a7;">/100</span></div>'
                f'<div style="color:#8b93a7; font-size:0.85rem;">'
                f'{lt_row["score_percentile"]:.0f}th percentile of the watchlist</div></div>'
            )
            st.markdown(lt_card, unsafe_allow_html=True)

            breakdown = pd.DataFrame({
                "Factor": ["Valuation", "Profitability", "Fin. Health", "Growth", "Margins"],
                "Score": [lt_row["valuation_score"], lt_row["profitability_score"],
                          lt_row["financial_health_score"], lt_row["growth_score"], lt_row["margin_score"]],
            })
            st.bar_chart(breakdown.set_index("Factor"))

        st.caption(
            "Transparent, sector-relative rule-based score -- not machine-learned. "
            "Useful for understanding *why* a stock ranks where it does."
        )

# ============================================================
# TAB: CHARTS
# ============================================================
with tab_charts:
    st.plotly_chart(plot_price_chart(featured_df, selected_symbol), use_container_width=True)

# ============================================================
# TAB: LONG TERM DETAIL
# ============================================================
with tab_longterm:
    if long_term_scores is not None:
        st.markdown('<div class="section-header">🏆 Watchlist Leaderboard</div>', unsafe_allow_html=True)
        display_cols = ["symbol", "sector", "cap_category", "total_score", "score_percentile", "recommendation"]
        available_cols = [c for c in display_cols if c in long_term_scores.columns]
        st.dataframe(
            long_term_scores[available_cols].sort_values("total_score", ascending=False),
            use_container_width=True, height=500,
        )
    else:
        st.info("Run `python src/long_term_scorer.py` to generate the leaderboard.")

# ============================================================
# TAB: RAW DATA
# ============================================================
with tab_data:
    st.markdown('<div class="section-header">📋 Recent Price Data</div>', unsafe_allow_html=True)
    st.dataframe(
        featured_df[["Date", "Open", "High", "Low", "Close", "Volume", "RSI_14", "MACD"]].tail(30),
        use_container_width=True,
    )

    st.markdown('<div class="section-header" style="margin-top:20px;">🏢 Fundamentals</div>', unsafe_allow_html=True)
    fund_df = pd.DataFrame({
        "Metric": ["Sector", "Industry", "Market Cap", "P/E Ratio", "Cap Category"],
        "Value": [
            info.get("sector", "N/A"), info.get("industry", "N/A"),
            f"₹{info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A",
            f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else "N/A",
            cap_category,
        ]
    })
    st.table(fund_df)