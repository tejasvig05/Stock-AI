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
from scipy.optimize import minimize
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

/* ---- Hover polish ---- */
.metric-card, .signal-card {
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.metric-card:hover {
    border-color: rgba(255,255,255,0.18);
}

/* ---- Footer ---- */
.app-footer {
    color: #5b6377;
    font-size: 0.75rem;
    text-align: center;
    padding: 24px 0 8px 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 32px;
}
.app-footer a { color: #7dd3fc; text-decoration: none; }

/* ---- Mobile responsiveness ---- */
@media (max-width: 768px) {
    .hero-title { font-size: 1.5rem; }
    .hero-sub { font-size: 0.82rem; margin-bottom: 1rem; }
    .metric-value { font-size: 1.15rem; }
    .metric-card { padding: 12px 14px; }
    .signal-card { padding: 16px 18px; }
    .section-header { font-size: 0.95rem; }
    .badge { font-size: 0.85rem; padding: 5px 12px; }
    .mobile-hint { display: block !important; }
}
.mobile-hint {
    display: none;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #7dd3fc;
    margin-bottom: 14px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Only visible on narrow (mobile) viewports via CSS media query above --
# points users to the sidebar toggle, since Streamlit collapses it by
# default on mobile and the stock selector lives there
st.markdown(
    '<div class="mobile-hint">👆 Tap the arrow (top-left) to open the stock selector</div>',
    unsafe_allow_html=True,
)


def format_inr_compact(value):
    """Formats large rupee values in Indian convention (Crore/Lakh) instead
    of raw digits -- e.g. 1723000000000 -> '17.2L Cr'. Much more readable
    and matches how Indian financial sites actually display market cap."""
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    crore = 1_00_00_000
    lakh = 1_00_000
    if value >= 100 * crore:
        return f"₹{value / crore:,.0f} Cr"
    elif value >= crore:
        return f"₹{value / crore:,.1f} Cr"
    elif value >= lakh:
        return f"₹{value / lakh:,.1f} L"
    else:
        return f"₹{value:,.0f}"


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


@st.cache_data(ttl=3600)
def build_returns_matrix(symbols: tuple, period: str = "1y") -> pd.DataFrame:
    """
    Builds a DataFrame of daily returns for multiple stocks, aligned by
    date. This is the core input to Markowitz mean-variance optimization --
    everything downstream (expected returns, covariance matrix) is derived
    from this.
    """
    returns = {}
    for sym in symbols:
        try:
            df, _ = load_stock_data(sym)
            df = df.set_index("Date")
            returns[sym] = df["daily_return"]
        except Exception:
            continue
    returns_df = pd.DataFrame(returns).dropna()
    return returns_df


def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
    """Annualized expected return and volatility for a given weight vector."""
    port_return = np.sum(mean_returns * weights) * trading_days
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))
    return port_return, port_vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_vol if p_vol > 0 else 0


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.065, objective="max_sharpe", max_weight=1.0):
    """
    Solves for optimal weights using scipy's SLSQP optimizer -- the
    standard approach for constrained mean-variance optimization.
    Constraints: weights sum to 1 (fully invested), no shorting,
    and optionally a max weight per stock -- unconstrained mean-variance
    optimization is well known to produce unstable, over-concentrated
    "corner" portfolios when using sample covariance from limited history
    (the "Markowitz optimization enigma," Michaud 1989); capping max
    position size is the standard practical fix.
    """
    n = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate) if objective == "max_sharpe" else (mean_returns, cov_matrix)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, max_weight) for _ in range(n))
    initial_guess = np.array([1 / n] * n)

    objective_fn = negative_sharpe if objective == "max_sharpe" else portfolio_volatility
    result = minimize(objective_fn, initial_guess, args=args, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    return result.x


def compute_efficient_frontier(mean_returns, cov_matrix, n_points=40):
    """Traces the efficient frontier: for a range of target returns, finds
    the MINIMUM volatility portfolio achieving that return. This is the
    classic Markowitz frontier curve."""
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    target_returns = np.linspace(min_ret, max_ret, n_points)

    n = len(mean_returns)
    frontier_vols = []
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_performance(w, mean_returns, cov_matrix)[0] - t},
        )
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(portfolio_volatility, [1 / n] * n, args=(mean_returns, cov_matrix),
                           method="SLSQP", bounds=bounds, constraints=constraints)
        frontier_vols.append(result.fun if result.success else np.nan)

    return target_returns, np.array(frontier_vols)


def get_full_signal_for_symbol(symbol, model, scaler, reg_model, reg_scaler, long_term_scores_df):
    """Bundles everything the portfolio tab needs for one holding: current
    price, short-term signal, expected return, and long-term score. Reuses
    the same cached functions as the main dashboard view."""
    try:
        f_df, s_info = load_stock_data(symbol)
        s_pred, s_proba = get_recommendation(f_df, model, scaler)
        s_exp_ret = get_expected_return(f_df, reg_model, reg_scaler)
        s_lt = get_long_term_score(long_term_scores_df, symbol)
        return {
            "current_price": s_info.get("currentPrice"),
            "signal": s_pred,
            "confidence": s_proba[s_pred] if s_pred else None,
            "expected_return": s_exp_ret,
            "lt_score": s_lt["total_score"] if s_lt is not None else None,
            "lt_recommendation": s_lt["recommendation"] if s_lt is not None else None,
        }
    except Exception:
        return {
            "current_price": None, "signal": None, "confidence": None,
            "expected_return": None, "lt_score": None, "lt_recommendation": None,
        }


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

tab_overview, tab_charts, tab_longterm, tab_portfolio, tab_optimizer, tab_data = st.tabs(
    ["🎯 Signals", "📊 Charts & Indicators", "💰 Long-Term Score", "💼 Portfolio",
     "⚖️ Portfolio Optimizer", "📋 Raw Data"]
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
    # --- Sector peer comparison for the selected stock ---
    st.markdown('<div class="section-header">🔬 Sector Peer Comparison</div>', unsafe_allow_html=True)

    if long_term_scores is not None and lt_row is not None:
        stock_sector = lt_row.get("sector")
        peers_df = long_term_scores[
            (long_term_scores["sector"] == stock_sector) &
            (long_term_scores["symbol"] != selected_symbol)
        ].sort_values("total_score", ascending=False).head(6)

        if peers_df.empty:
            st.info(f"No other stocks found in the '{stock_sector}' sector in the current watchlist.")
        else:
            pc1, pc2 = st.columns([1, 1])

            with pc1:
                st.caption(f"Top peers in **{stock_sector}**, ranked by fundamental score")
                compare_cols = ["symbol", "cap_category", "trailingPE", "total_score", "recommendation"]
                available = [c for c in compare_cols if c in peers_df.columns]
                peer_display = peers_df[available].rename(columns={
                    "symbol": "Symbol", "cap_category": "Cap", "trailingPE": "P/E",
                    "total_score": "Score", "recommendation": "Recommendation",
                })
                st.dataframe(peer_display, use_container_width=True, hide_index=True)

            with pc2:
                st.caption(f"Fundamental score: {selected_symbol} vs. top peers")
                compare_scores = pd.concat([
                    pd.DataFrame([{"symbol": selected_symbol, "total_score": lt_row["total_score"]}]),
                    peers_df[["symbol", "total_score"]],
                ])
                bar_fig = go.Figure()
                colors = ["#a78bfa" if s == selected_symbol else "#38bdf8" for s in compare_scores["symbol"]]
                bar_fig.add_trace(go.Bar(
                    x=compare_scores["symbol"], y=compare_scores["total_score"],
                    marker_color=colors,
                ))
                bar_fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=320, margin=dict(t=10, b=10, l=10, r=10),
                    yaxis_title="Fundamental Score",
                )
                st.plotly_chart(bar_fig, use_container_width=True)
                st.caption(f"{selected_symbol} highlighted in purple")

            # --- Normalized price performance vs top 3 peers ---
            st.caption("Price performance (normalized to 0% at start of period) -- last 1 year")
            top_peer_symbols = peers_df["symbol"].head(3).tolist()
            perf_fig = go.Figure()

            for sym, color in zip([selected_symbol] + top_peer_symbols,
                                   ["#f472b6", "#38bdf8", "#4ade80", "#facc15"]):
                try:
                    if sym == selected_symbol:
                        p_df = featured_df
                    else:
                        p_df, _ = load_stock_data(sym)
                    normalized = (p_df["Close"] / p_df["Close"].iloc[0] - 1) * 100
                    perf_fig.add_trace(go.Scatter(
                        x=p_df["Date"], y=normalized, name=sym,
                        line=dict(color=color, width=2.5 if sym == selected_symbol else 1.5),
                    ))
                except Exception:
                    continue

            perf_fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(t=10, b=10, l=10, r=10),
                yaxis_title="Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(perf_fig, use_container_width=True)
    else:
        st.info("Long-term scores not available yet -- run `python src/long_term_scorer.py`.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Full watchlist leaderboard ---
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
# TAB: PORTFOLIO
# ============================================================
with tab_portfolio:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    st.markdown('<div class="section-header">💼 Your Portfolio</div>', unsafe_allow_html=True)
    st.caption(
        "Holdings are stored for this browser session only -- nothing is saved to a "
        "server or shared with other visitors. Refreshing the page clears this."
    )

    with st.form("add_holding_form", clear_on_submit=True):
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
        with f1:
            new_symbol = st.selectbox("Stock", WATCHLIST, key="new_holding_symbol")
        with f2:
            new_qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        with f3:
            new_buy_price = st.number_input("Buy Price (₹)", min_value=0.01, value=100.0, step=1.0)
        with f4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("➕ Add Holding")

        if submitted:
            st.session_state.portfolio.append({
                "symbol": new_symbol, "quantity": new_qty, "buy_price": new_buy_price,
            })
            st.rerun()

    if not st.session_state.portfolio:
        st.info("No holdings added yet. Use the form above to add your first position.")
    else:
        rows = []
        total_invested = 0.0
        total_current = 0.0

        with st.spinner("Fetching live prices and signals for your holdings..."):
            for i, holding in enumerate(st.session_state.portfolio):
                sig = get_full_signal_for_symbol(
                    holding["symbol"], model, scaler, reg_model, reg_scaler, long_term_scores
                )
                invested = holding["quantity"] * holding["buy_price"]
                current_price = sig["current_price"]
                current_value = holding["quantity"] * current_price if current_price else None
                pnl = (current_value - invested) if current_value is not None else None
                pnl_pct = (pnl / invested * 100) if pnl is not None and invested > 0 else None

                total_invested += invested
                if current_value is not None:
                    total_current += current_value

                rows.append({
                    "idx": i,
                    "Symbol": holding["symbol"],
                    "Qty": holding["quantity"],
                    "Buy Price": f"₹{holding['buy_price']:.2f}",
                    "Current Price": f"₹{current_price:.2f}" if current_price else "N/A",
                    "Invested": f"₹{invested:,.0f}",
                    "Current Value": f"₹{current_value:,.0f}" if current_value else "N/A",
                    "P&L": f"₹{pnl:+,.0f}" if pnl is not None else "N/A",
                    "P&L %": f"{pnl_pct:+.2f}%" if pnl_pct is not None else "N/A",
                    "Short-Term Signal": sig["signal"] or "N/A",
                    "Long-Term Score": f"{sig['lt_score']:.0f}/100" if sig["lt_score"] is not None else "N/A",
                })

        # --- Summary metrics ---
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(metric_card("Total Invested", f"₹{total_invested:,.0f}"), unsafe_allow_html=True)
        with s2:
            st.markdown(metric_card("Current Value", f"₹{total_current:,.0f}"), unsafe_allow_html=True)
        with s3:
            pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Total P&L</div>'
                f'<div class="metric-value" style="color:{pnl_color}">'
                f'₹{total_pnl:+,.0f} ({total_pnl_pct:+.2f}%)</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        portfolio_df = pd.DataFrame(rows).drop(columns=["idx"])
        st.dataframe(portfolio_df, use_container_width=True)

        st.markdown("**Remove a holding:**")
        remove_cols = st.columns(min(len(rows), 6))
        for i, holding in enumerate(st.session_state.portfolio):
            with remove_cols[i % len(remove_cols)]:
                if st.button(f"❌ {holding['symbol']}", key=f"remove_{i}"):
                    st.session_state.portfolio.pop(i)
                    st.rerun()

        st.caption(
            "Signals shown are the same short-term (5-day) and long-term (fundamentals) "
            "model outputs used elsewhere in this dashboard -- see the Signals tab for "
            "each stock's full detail and explainability."
        )

# ============================================================
# TAB: PORTFOLIO OPTIMIZER
# ============================================================
with tab_optimizer:
    st.markdown('<div class="section-header">⚖️ Modern Portfolio Theory Optimizer</div>', unsafe_allow_html=True)
    st.caption(
        "Given a set of stocks, this computes the mathematically optimal allocation "
        "using Markowitz mean-variance optimization -- the same core technique used "
        "in real institutional asset allocation. Not a prediction; a risk/return "
        "trade-off calculation based on historical volatility and correlation."
    )

    default_selection = WATCHLIST[:6] if len(WATCHLIST) >= 6 else WATCHLIST
    opt_symbols = st.multiselect(
        "Select stocks to optimize across (3-12 recommended)",
        WATCHLIST, default=default_selection, max_selections=15,
    )

    risk_free = st.slider("Risk-free rate (annual %, e.g. Indian G-Sec yield)", 0.0, 12.0, 6.5, 0.1) / 100
    max_weight_pct = st.slider(
        "Max allocation per stock (%)", 20, 100, 35, 5,
        help="Unconstrained optimization tends to concentrate heavily in 1-2 stocks "
             "(a known instability in mean-variance optimization). Capping single-stock "
             "exposure produces more realistic, diversified portfolios.",
    )
    max_weight = max_weight_pct / 100

    if len(opt_symbols) < 3:
        st.warning("Select at least 3 stocks to run optimization.")
    elif max_weight * len(opt_symbols) < 1.0:
        st.error(
            f"Infeasible: {len(opt_symbols)} stocks capped at {max_weight_pct}% each can only "
            f"reach {max_weight * len(opt_symbols) * 100:.0f}% total. Raise the cap or add more stocks."
        )
    elif st.button("🔮 Run Optimization", type="primary"):
        with st.spinner("Fetching historical returns and solving for optimal weights..."):
            returns_df = build_returns_matrix(tuple(sorted(opt_symbols)))

            if returns_df.shape[1] < 3 or len(returns_df) < 30:
                st.error("Not enough overlapping historical data for these stocks. Try different symbols.")
            else:
                mean_returns = returns_df.mean()
                cov_matrix = returns_df.cov()

                max_sharpe_w = optimize_portfolio(mean_returns, cov_matrix, risk_free, "max_sharpe", max_weight)
                min_vol_w = optimize_portfolio(mean_returns, cov_matrix, risk_free, "min_vol", max_weight)
                equal_w = np.array([1 / len(returns_df.columns)] * len(returns_df.columns))

                ms_ret, ms_vol = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)
                mv_ret, mv_vol = portfolio_performance(min_vol_w, mean_returns, cov_matrix)
                eq_ret, eq_vol = portfolio_performance(equal_w, mean_returns, cov_matrix)

                ms_sharpe = (ms_ret - risk_free) / ms_vol
                mv_sharpe = (mv_ret - risk_free) / mv_vol
                eq_sharpe = (eq_ret - risk_free) / eq_vol

                # --- Results cards ---
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(
                        f'<div class="signal-card"><b style="color:#4ade80;">🏆 Max Sharpe Portfolio</b>'
                        f'<div style="margin-top:10px; font-size:1.6rem; font-weight:800; font-family:JetBrains Mono, monospace;">'
                        f'{ms_ret*100:.1f}%<span style="font-size:0.9rem; color:#8b93a7;"> return</span></div>'
                        f'<div style="color:#8b93a7;">Volatility: {ms_vol*100:.1f}% &nbsp;|&nbsp; Sharpe: {ms_sharpe:.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f'<div class="signal-card"><b style="color:#38bdf8;">🛡️ Min Volatility Portfolio</b>'
                        f'<div style="margin-top:10px; font-size:1.6rem; font-weight:800; font-family:JetBrains Mono, monospace;">'
                        f'{mv_ret*100:.1f}%<span style="font-size:0.9rem; color:#8b93a7;"> return</span></div>'
                        f'<div style="color:#8b93a7;">Volatility: {mv_vol*100:.1f}% &nbsp;|&nbsp; Sharpe: {mv_sharpe:.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        f'<div class="signal-card"><b style="color:#facc15;">⚪ Equal Weight (Baseline)</b>'
                        f'<div style="margin-top:10px; font-size:1.6rem; font-weight:800; font-family:JetBrains Mono, monospace;">'
                        f'{eq_ret*100:.1f}%<span style="font-size:0.9rem; color:#8b93a7;"> return</span></div>'
                        f'<div style="color:#8b93a7;">Volatility: {eq_vol*100:.1f}% &nbsp;|&nbsp; Sharpe: {eq_sharpe:.2f}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # --- Efficient frontier chart ---
                oc1, oc2 = st.columns([3, 2])
                with oc1:
                    with st.spinner("Tracing efficient frontier..."):
                        frontier_returns, frontier_vols = compute_efficient_frontier(mean_returns, cov_matrix)

                    frontier_fig = go.Figure()
                    frontier_fig.add_trace(go.Scatter(
                        x=frontier_vols * 100, y=frontier_returns * 100, mode="lines",
                        name="Efficient Frontier", line=dict(color="#a78bfa", width=2),
                    ))
                    frontier_fig.add_trace(go.Scatter(
                        x=[ms_vol * 100], y=[ms_ret * 100], mode="markers", name="Max Sharpe",
                        marker=dict(color="#4ade80", size=14, symbol="star"),
                    ))
                    frontier_fig.add_trace(go.Scatter(
                        x=[mv_vol * 100], y=[mv_ret * 100], mode="markers", name="Min Volatility",
                        marker=dict(color="#38bdf8", size=14, symbol="diamond"),
                    ))
                    frontier_fig.add_trace(go.Scatter(
                        x=[eq_vol * 100], y=[eq_ret * 100], mode="markers", name="Equal Weight",
                        marker=dict(color="#facc15", size=12, symbol="circle"),
                    ))
                    frontier_fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=420, margin=dict(t=20, b=10, l=10, r=10),
                        xaxis_title="Annualized Volatility (Risk) %",
                        yaxis_title="Annualized Expected Return %",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(frontier_fig, use_container_width=True)
                    st.caption(
                        "Every point on the purple curve is the lowest-risk portfolio achievable for "
                        "that return level. Portfolios below/right of the curve are sub-optimal -- "
                        "you could get more return for the same risk, or less risk for the same return."
                    )

                with oc2:
                    st.markdown("**Recommended Allocation (Max Sharpe)**")
                    alloc_df = pd.DataFrame({
                        "Stock": returns_df.columns,
                        "Weight": (max_sharpe_w * 100).round(1),
                    }).sort_values("Weight", ascending=False)
                    alloc_df = alloc_df[alloc_df["Weight"] > 0.1]

                    alloc_fig = go.Figure(data=[go.Pie(
                        labels=alloc_df["Stock"], values=alloc_df["Weight"], hole=0.5,
                        marker=dict(colors=["#4ade80", "#38bdf8", "#a78bfa", "#f472b6", "#facc15", "#fb923c"]),
                    )])
                    alloc_fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                    )
                    st.plotly_chart(alloc_fig, use_container_width=True)
                    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Based on trailing 1-year historical returns and volatility. Past correlation "
        "and volatility are not guaranteed to persist -- this is a risk/return optimization "
        "tool, not a return forecast."
    )

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
            format_inr_compact(info.get("marketCap")),
            f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else "N/A",
            cap_category,
        ]
    })
    st.table(fund_df)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """<div class="app-footer">
    Built by Tejasvi Gupta &nbsp;•&nbsp;
    <a href="https://github.com/tejasvig05/Stock-AI" target="_blank">GitHub</a> &nbsp;•&nbsp;
    Educational project, not financial advice &nbsp;•&nbsp;
    Data via yfinance, refreshed daily
    </div>""",
    unsafe_allow_html=True,
)