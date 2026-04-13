"""
Real-Time NSE Stock Price Predictor with Buy/Sell Signal Dashboard
Run: streamlit run app.py
"""

import time
import warnings
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports from src ──────────────────────────────────────────────────────────
from src.data     import fetch_ohlcv, fetch_market_context, get_stock_info
from src.features import build_feature_matrix, get_feature_cols, SECTOR_INDEX_MAP
from src.models   import (
    train_arima, arima_rolling_forecast, arima_future_forecast, arima_metrics,
    train_lstm, lstm_predict_latest, lstm_metrics,
    train_xgboost, xgb_predict_latest, xgb_metrics,
    walk_forward_validate,
)
from src.signals  import generate_signal, ensemble_prob, trend_label

if "sidebar_hidden" not in st.session_state:
    st.session_state.sidebar_hidden = False

# ── MODEL CACHING ─────────────────────────────────────────────

@st.cache_resource
def get_arima_model(close):
    return train_arima(close)


@st.cache_resource
def get_lstm_model(feat_df, feature_cols, ticker):
    model_path = os.path.join("cache", f"{ticker.replace('.','_')}_lstm.keras")

    return train_lstm(
        feat_df, feature_cols,
        window=60,
        train_ratio=0.8,
        epochs=12,          # faster default for dashboard responsiveness
        batch_size=32,
        patience=3,
        model_path=model_path,
    )

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bb-bg: #0a0d12;
    --bb-panel: #111723;
    --bb-panel-2: #0f141e;
    --bb-border: #2a3345;
    --bb-accent: #f2a900;
    --bb-text: #d9e1ee;
    --bb-muted: #96a3ba;
}

.stApp {
    background: radial-gradient(circle at top right, #151b28 0%, var(--bb-bg) 42%);
    color: var(--bb-text);
    font-family: 'IBM Plex Sans', sans-serif;
}

.block-container {
    padding-top: 1.05rem;
    padding-bottom: 1.6rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121925 0%, #0f141d 100%);
    border-right: 1px solid var(--bb-border);
}

[data-testid="stSidebar"] * {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Remove blinking text cursor from selectbox controls (e.g., stock list) */
div[data-baseweb="select"] input {
    caret-color: transparent !important;
    cursor: pointer !important;
}

div[data-baseweb="select"] input:focus {
    outline: none !important;
}

.site-title {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-size: 2.05rem;
    margin: 0.1rem 0 0.25rem 0;
    color: var(--bb-accent);
}

.terminal-strip {
    border: 1px solid var(--bb-border);
    background: linear-gradient(90deg, #151d2b 0%, #111825 100%);
    border-radius: 8px;
    padding: 0.5rem 0.85rem;
    margin: 0.4rem 0 0.9rem 0;
    font-family: 'JetBrains Mono', monospace;
    color: var(--bb-text);
    font-size: 0.85rem;
}

.terminal-strip .label {
    color: var(--bb-muted);
}

.terminal-strip .value {
    color: var(--bb-accent);
    font-weight: 700;
    margin-right: 1rem;
}

.terminal-section {
    margin-top: 0.85rem;
    margin-bottom: 0.8rem;
    padding: 0.5rem 0.75rem;
    border: 1px solid #8a6200;
    border-radius: 6px;
    background: linear-gradient(90deg, rgba(242,169,0,0.22) 0%, rgba(242,169,0,0.08) 100%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.18rem;
    font-weight: 700;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    text-align: center;
    color: #f6b93b;
}

[data-testid="stMetricLabel"] {
    color: var(--bb-muted);
    font-size: 0.75rem;
    letter-spacing: 0.3px;
    line-height: 1.2;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

[data-testid="stMetricValue"] {
    color: var(--bb-text);
    font-size: 1.4rem;
    font-weight: 500;
    line-height: 1.2;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

.stDataFrame, div[data-testid="stTable"] {
    border: 1px solid var(--bb-border);
    border-radius: 8px;
    overflow: hidden;
}

div[data-testid="stPlotlyChart"],
div[data-testid="stMetric"],
div[data-testid="stDataFrame"],
div[data-testid="stInfo"],
div[data-testid="stWarning"] {
    margin-top: 0.4rem;
    margin-bottom: 0.95rem;
}

div[data-baseweb="tab-list"] {
    gap: 0.7rem;
    margin-top: 0.35rem;
    margin-bottom: 0.95rem;
}

div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    background: transparent !important;
    height: 0 !important;
    border: 0 !important;
    display: none !important;
}

button[data-baseweb="tab"] {
    border: 1px solid var(--bb-border) !important;
    border-radius: 12px !important;
    background: #0f1520 !important;
    color: var(--bb-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    padding: 0.52rem 0.9rem !important;
    transition: all 0.2s ease;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #141414 !important;
    background: var(--bb-accent) !important;
    border-color: #ffcc53 !important;
}

button[data-testid="baseButton-secondary"] {
    background: #0f1520 !important;
    color: #d9e1ee !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='site-title'>NSE Stock Predictor</h1>", unsafe_allow_html=True)

ctrl_col1, _ = st.columns([0.75, 7.25])
with ctrl_col1:
    toggle_label = "❯❯" if st.session_state.sidebar_hidden else "❮❮"
    if st.button(toggle_label, key="sidebar_toggle_arrow", help="Toggle sidebar", use_container_width=True):
        st.session_state.sidebar_hidden = not st.session_state.sidebar_hidden
        st.rerun()

if st.session_state.sidebar_hidden:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── stock universe ────────────────────────────────────────────────────────────
NSE_STOCKS = {
    "IT": {
        "TCS":         "TCS.NS",
        "Infosys":     "INFY.NS",
        "Wipro":       "WIPRO.NS",
        "HCL Tech":    "HCLTECH.NS",
        "Tech Mahindra": "TECHM.NS",
    },
    "Banking & Finance": {
        "HDFC Bank":   "HDFCBANK.NS",
        "ICICI Bank":  "ICICIBANK.NS",
        "SBI":         "SBIN.NS",
        "Kotak Bank":  "KOTAKBANK.NS",
        "Axis Bank":   "AXISBANK.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
    },
    "Energy & Oil": {
        "Reliance":    "RELIANCE.NS",
        "ONGC":        "ONGC.NS",
        "NTPC":        "NTPC.NS",
        "Power Grid":  "POWERGRID.NS",
        "Adani Green": "ADANIGREEN.NS",
    },
    "Consumer & FMCG": {
        "Hindustan Unilever": "HINDUNILVR.NS",
        "ITC":         "ITC.NS",
        "Nestle India": "NESTLEIND.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Titan":       "TITAN.NS",
    },
    "Auto": {
        "Maruti Suzuki": "MARUTI.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Bajaj Auto":  "BAJAJ-AUTO.NS",
        "M&M":         "M&M.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS",
    },
    "Pharma": {
        "Sun Pharma":  "SUNPHARMA.NS",
        "Dr. Reddy's": "DRREDDY.NS",
        "Cipla":       "CIPLA.NS",
        "Divi's Labs": "DIVISLAB.NS",
    },
    "Metals & Mining": {
        "Tata Steel":  "TATASTEEL.NS",
        "JSW Steel":   "JSWSTEEL.NS",
        "Hindalco":    "HINDALCO.NS",
        "Coal India":  "COALINDIA.NS",
    },
}

# Ticker -> primary company website domain (used with Clearbit logo endpoint)
STOCK_LOGO_DOMAINS = {
    "TCS.NS": "tcs.com",
    "INFY.NS": "infosys.com",
    "WIPRO.NS": "wipro.com",
    "HCLTECH.NS": "hcltech.com",
    "TECHM.NS": "techmahindra.com",
    "HDFCBANK.NS": "hdfcbank.com",
    "ICICIBANK.NS": "icicibank.com",
    "SBIN.NS": "sbi.co.in",
    "KOTAKBANK.NS": "kotak.com",
    "AXISBANK.NS": "axisbank.com",
    "BAJFINANCE.NS": "bajajfinserv.in",
    "RELIANCE.NS": "ril.com",
    "ONGC.NS": "ongcindia.com",
    "NTPC.NS": "ntpc.co.in",
    "POWERGRID.NS": "powergrid.in",
    "ADANIGREEN.NS": "adanigreenenergy.com",
    "HINDUNILVR.NS": "hul.co.in",
    "ITC.NS": "itcportal.com",
    "NESTLEIND.NS": "nestle.in",
    "ASIANPAINT.NS": "asianpaints.com",
    "TITAN.NS": "titancompany.in",
    "MARUTI.NS": "marutisuzuki.com",
    "TATAMOTORS.NS": "tatamotors.com",
    "BAJAJ-AUTO.NS": "bajajauto.com",
    "M&M.NS": "mahindra.com",
    "HEROMOTOCO.NS": "heromotocorp.com",
    "SUNPHARMA.NS": "sunpharma.com",
    "DRREDDY.NS": "drreddys.com",
    "CIPLA.NS": "cipla.com",
    "DIVISLAB.NS": "divislabs.com",
    "TATASTEEL.NS": "tatasteel.com",
    "JSWSTEEL.NS": "jsw.in",
    "HINDALCO.NS": "hindalco.com",
    "COALINDIA.NS": "coalindia.in",
}

# Flat label → ticker map and label → sector map for lookup
_LABEL_TO_TICKER = {
    name: tick
    for sector, stocks in NSE_STOCKS.items()
    for name, tick in stocks.items()
}
_LABEL_TO_SECTOR = {
    name: sector
    for sector, stocks in NSE_STOCKS.items()
    for name in stocks
}
_LABELS = list(_LABEL_TO_TICKER.keys())

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sector_choice = st.selectbox("Sector", ["All"] + list(NSE_STOCKS.keys()))
    if sector_choice == "All":
        filtered_labels = _LABELS
    else:
        filtered_labels = list(NSE_STOCKS[sector_choice].keys())

    selected_label  = st.selectbox("Stock", filtered_labels)
    ticker          = _LABEL_TO_TICKER[selected_label]
    active_sector   = _LABEL_TO_SECTOR[selected_label]
    st.caption(f"Ticker: `{ticker}`")

    years         = st.slider("Historical data (years)", 2, 7, 5)
    forecast_days = st.slider("Forecast horizon (days)", 5, 60, 30)
    run_btn       = st.button("Run Prediction", type="primary", use_container_width=True)
    auto_refresh  = st.toggle("Auto-refresh every 5 min", value=False)
    st.caption("Data via yfinance · Models trained fresh per stock")

# ── session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None
if "auto_refresh_due" not in st.session_state:
    st.session_state.auto_refresh_due = False

# ── auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    # st_autorefresh injects a JS timer — actually triggers a rerun every 5 min
    count = st_autorefresh(interval=300_000, limit=None, key="autorefresh")
    if count > 0:
        st.session_state.auto_refresh_due = True
    st.sidebar.info("Auto-refresh ON — every 5 min")


# ── cached data fetchers ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, years: int, sector: str):
    df            = fetch_ohlcv(ticker, years)
    nifty, usdinr = fetch_market_context(years)
    info          = get_stock_info(ticker)
    # Fetch sector index if available; silently skip on failure
    sector_df = None
    sector_ticker = SECTOR_INDEX_MAP.get(sector)
    if sector_ticker:
        try:
            sector_df = fetch_ohlcv(sector_ticker, years)
        except Exception:
            sector_df = None
    feat_df      = build_feature_matrix(df, nifty, usdinr, sector_df, ticker=ticker)
    feature_cols = get_feature_cols(feat_df)
    return df, feat_df, feature_cols, info


# ── helpers ───────────────────────────────────────────────────────────────────
def fmt_inr(val: float) -> str:
    if val >= 1e12:
        return f"₹{val/1e12:.2f}T"
    if val >= 1e9:
        return f"₹{val/1e9:.2f}B"
    if val >= 1e7:
        return f"₹{val/1e7:.2f}Cr"
    return f"₹{val:,.0f}"


def safe_last(series: pd.Series, default=0):
    try:
        v = series.dropna().iloc[-1]
        return float(v)
    except Exception:
        return default


def get_stock_logo_url(ticker: str, stock_name: str) -> str:
    domain = STOCK_LOGO_DOMAINS.get(ticker)
    if domain:
        return f"https://logo.clearbit.com/{domain}"
    fallback = stock_name.replace(" ", "+")
    return (
        "https://ui-avatars.com/api/"
        f"?name={fallback}&background=0f141e&color=f2a900&size=96&bold=true"
    )


def terminal_section(title: str):
    st.markdown(f"<div class='terminal-section'>{title}</div>", unsafe_allow_html=True)


# ── main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(ticker: str, years: int, forecast_days: int, sector: str):
    results = {}

    with st.spinner("Fetching market data + sector index..."):
        df, feat_df, feature_cols, info = load_data(ticker, years, sector)
        # Use recent window to keep retraining fast on stock switches.
        feat_df = feat_df.tail(500)
    results["df"]           = df
    results["feat_df"]      = feat_df
    results["feature_cols"] = feature_cols
    results["info"]         = info

    # ── ARIMA ──
    with st.spinner("Training ARIMA model (~30 seconds)..."):
        close = feat_df["Close"]
        arima_model, arima_split = get_arima_model(close)
        arima_preds, arima_model = arima_rolling_forecast(close, arima_model, arima_split)
        arima_actual = close.iloc[arima_split:].values
        arima_met    = arima_metrics(arima_actual, arima_preds)
        fc, lo, hi   = arima_future_forecast(arima_model, forecast_days)
    results["arima"] = dict(
        model=arima_model, split=arima_split,
        preds=arima_preds, actual=arima_actual,
        metrics=arima_met, fc=fc, lo=lo, hi=hi,
        dates=feat_df.index,
    )

    # ── XGBoost (fast — trains in seconds) ──
    with st.spinner("Training XGBoost model..."):
        xgb_model, xgb_X_test, xgb_y_test, xgb_importances = train_xgboost(
            feat_df, feature_cols
        )
        xgb_prob = xgb_predict_latest(xgb_model, feat_df, feature_cols)
        xgb_met  = xgb_metrics(xgb_model, xgb_X_test, xgb_y_test)
    results["xgb"] = dict(
        model=xgb_model, prob=xgb_prob, metrics=xgb_met,
        importances=xgb_importances,
        X_test=xgb_X_test, y_test=xgb_y_test,
    )

    # ── LSTM ──
    model_path = os.path.join("cache", f"{ticker.replace('.','_')}_lstm.keras")
    with st.spinner("Training LSTM model (60–120 seconds)..."):
        lstm_model, scaler, history, split_idx, X_test, y_test = get_lstm_model(
            feat_df, feature_cols, ticker
        )
        lstm_prob = lstm_predict_latest(lstm_model, scaler, feat_df, feature_cols)
        lstm_met  = lstm_metrics(lstm_model, X_test, y_test)
    results["lstm"] = dict(
        model=lstm_model, scaler=scaler, history=history,
        prob=lstm_prob, metrics=lstm_met,
        X_test=X_test, y_test=y_test,
    )

    return results


@st.cache_resource(ttl=1800)
def get_pipeline_results(ticker: str, years: int, forecast_days: int, sector: str):
    """Cache full pipeline per stock/settings to speed up revisits."""
    return run_pipeline(ticker, years, forecast_days, sector)


# ── trigger pipeline ──────────────────────────────────────────────────────────
should_auto_refresh = (
    auto_refresh
    and st.session_state.auto_refresh_due
    and st.session_state.last_ticker == ticker
)

if run_btn or should_auto_refresh:
    st.session_state.results = get_pipeline_results(ticker, years, forecast_days, active_sector)
    st.session_state.last_ticker = ticker
    st.session_state.auto_refresh_due = False

results = st.session_state.results

if results is None:
    st.info("Configure settings in the sidebar and click **Run Prediction** to begin.")
    st.stop()

display_ticker = st.session_state.last_ticker or ticker
display_sector = results["info"].get("sector", active_sector)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Live Stock Info
# ─────────────────────────────────────────────────────────────────────────────
info   = results["info"]
feat_df = results["feat_df"]
df      = results["df"]

display_name = info.get("name") or display_ticker
st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.15rem; margin-top:1rem;">
            <div style="font-size:1.85rem; font-weight:700; line-height:1.2; color:#ffffff;">
                {display_name} 
                <span style="background-color:rgba(38,166,154,0.15); color:#26a69a; padding: 2px 8px; border-radius: 4px; font-family:'JetBrains Mono', monospace; font-size:1.1rem; margin-left:8px;">{display_ticker}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)
st.caption(f"Sector: {info['sector']}")
st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
current_price = info["current_price"] or safe_last(feat_df["Close"])
change_pct    = info["change_pct"]

c1.metric("Current Price",  f"₹{current_price:,.2f}")
c2.metric("Change Today",   f"{change_pct:+.2f}%",
          delta_color="normal" if change_pct >= 0 else "inverse")
c3.metric("52-Week High",   f"₹{info['week52_high']:,.2f}")
c4.metric("52-Week Low",    f"₹{info['week52_low']:,.2f}")
c5.metric("Volume",         f"{info['volume']:,}" if info['volume'] else "N/A")
c6.metric("Market Cap",     fmt_inr(info['market_cap']) if info['market_cap'] else "N/A")

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

arima_r = results["arima"]
lstm_r  = results["lstm"]
xgb_r   = results["xgb"]

lstm_prob     = lstm_r["prob"]
xgb_prob      = xgb_r["prob"]
arima_next    = float(arima_r["fc"][0])
rsi_val       = safe_last(feat_df["rsi"])
macd_hist_val = safe_last(feat_df["macd_hist"])

# Weighted ensemble: 40% LSTM + 40% XGBoost + 20% ARIMA direction
ens_prob_val  = ensemble_prob(lstm_prob, xgb_prob, arima_next, current_price)

signal, sig_color = generate_signal(lstm_prob, xgb_prob, rsi_val,
                                     macd_hist_val, arima_next, current_price)
trend_txt, trend_color = trend_label(ens_prob_val)

bg_map = {
    "green": "#388e3c", "lightgreen": "#4caf50",
    "orange": "#f57c00", "red": "#d32f2f", "gray": "#455a64",
}
bg = bg_map.get(sig_color, "#455a64")
trend_icon = '↗' if trend_color=='green' else '↘' if trend_color=='red' else '→'

st.markdown(f"""
<div style="background-color: {bg}; border-radius: 12px; padding: 28px; text-align: center; margin-bottom: 2.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="color: white; font-size: 2.8rem; font-weight: 800; letter-spacing: 1.5px; margin-bottom: 8px; text-transform: uppercase;">
    {signal}
  </div>
  <div style="color: rgba(255, 255, 255, 0.9); font-size: 0.95rem; font-weight: 500; margin-bottom: 4px;">
    LSTM Confidence: <b>{lstm_prob*100:.1f}%</b> &nbsp;|&nbsp; Trend: {trend_txt} {trend_icon}
  </div>
  <div style="color: rgba(255, 255, 255, 0.75); font-size: 0.85rem;">
    ARIMA next-day price forecast: ₹{arima_next:,.2f} &nbsp;|&nbsp; RSI: {rsi_val:.1f}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Price Chart — Last 6 Months")
chart_df = feat_df.iloc[-130:]  # ~6 months

fig_price = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.8, 0.2], vertical_spacing=0.03,
)

# Candlestick
fig_price.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df["Open"], high=chart_df["High"],
    low=chart_df["Low"],   close=chart_df["Close"],
    name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

# MAs
fig_price.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df["sma20"],
    name="SMA 20", line=dict(color="orange", dash="dash", width=1.5),
), row=1, col=1)
fig_price.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df["sma50"],
    name="SMA 50", line=dict(color="red", dash="dash", width=1.5),
), row=1, col=1)

# Bollinger Bands
fig_price.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df["bb_upper"],
    name="BB Upper", line=dict(color="rgba(173,216,230,0.6)", width=1),
    showlegend=False,
), row=1, col=1)
fig_price.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df["bb_lower"],
    name="BB", fill="tonexty",
    fillcolor="rgba(173,216,230,0.12)",
    line=dict(color="rgba(173,216,230,0.6)", width=1),
), row=1, col=1)

# Volume
up_mask   = chart_df["Close"] >= chart_df["Open"]
vol_colors = ["#26a69a" if u else "#ef5350" for u in up_mask]
fig_price.add_trace(go.Bar(
    x=chart_df.index, y=chart_df["Volume"],
    marker_color=vol_colors, name="Volume", showlegend=False,
), row=2, col=1)

fig_price.update_layout(
    height=450, xaxis_rangeslider_visible=False,
    template="plotly_dark", margin=dict(l=0, r=0, t=20, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_price, use_container_width=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("### Technical Indicators")

ind_df = feat_df.iloc[-260:]  # ~1 year for indicators

ti_c1, ti_c2, ti_c3 = st.columns(3)

with ti_c1:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=ind_df.index, y=ind_df["rsi"],
                                  line=dict(color="#7c4dff"), name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title=dict(text="RSI (14)", font=dict(size=16), x=0.01, xanchor="left"), height=250,
                           template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                           showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rsi, use_container_width=True)

with ti_c2:
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=ind_df.index, y=ind_df["macd"],
                                   line=dict(color="#2196f3"), name="MACD"))
    fig_macd.add_trace(go.Scatter(x=ind_df.index, y=ind_df["macd_signal"],
                                   line=dict(color="orange"), name="Signal"))
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                   for v in ind_df["macd_hist"].fillna(0)]
    fig_macd.add_trace(go.Bar(x=ind_df.index, y=ind_df["macd_hist"],
                               marker_color=hist_colors, name="Histogram"))
    fig_macd.update_layout(title=dict(text="MACD", font=dict(size=16), x=0.01, xanchor="left"), height=250,
                            template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font_size=10,
                            ), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_macd, use_container_width=True)

with ti_c3:
    fig_bbw = go.Figure()
    fig_bbw.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_width"],
                                  fill="tozeroy", line=dict(color="#ff6f00"),
                                  name="BB Width"))
    fig_bbw.update_layout(title=dict(text="Bollinger Band Width (Squeeze Indicator)", font=dict(size=16), x=0.01, xanchor="left"),
                           height=250, template="plotly_dark",
                           margin=dict(l=10, r=10, t=40, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bbw, use_container_width=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

st.markdown("### Model Performance & Forecast")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["ARIMA Evaluation", "LSTM Training", "Future Forecast", "Metrics Comparison", "Walk-Forward Validation"]
)

# ── Tab 1: ARIMA actual vs predicted ─────────────────────────────────────────
with tab1:
    test_dates  = arima_r["dates"][arima_r["split"]:]
    fig_arima   = go.Figure()
    fig_arima.add_trace(go.Scatter(
        x=test_dates, y=arima_r["actual"],
        name="Actual", line=dict(color="#2196f3"),
    ))
    fig_arima.add_trace(go.Scatter(
        x=test_dates, y=arima_r["preds"],
        name="ARIMA Predicted", line=dict(color="#ff5722", dash="dash"),
    ))
    fig_arima.update_layout(title="ARIMA: Actual vs Predicted (Test Set)",
                             height=400, template="plotly_dark",
                             margin=dict(l=0, r=0, t=70, b=10),
                             legend=dict(
                                 orientation="h",
                                 yanchor="bottom",
                                 y=1.02,
                                 xanchor="left",
                                 x=0,
                             ))
    st.plotly_chart(fig_arima, use_container_width=True)

    m = arima_r["metrics"]
    ca, cb, cc = st.columns(3)
    ca.metric("MAE",  f"₹{m['MAE']:.2f}")
    cb.metric("RMSE", f"₹{m['RMSE']:.2f}")
    cc.metric("MAPE", f"{m['MAPE']:.2f}%")

# ── Tab 2: LSTM training curves ─────────────────────────────────────────────
with tab2:
    hist = lstm_r["history"].history
    epochs_range = list(range(1, len(hist["accuracy"]) + 1))

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=epochs_range, y=hist["accuracy"],
                                   name="Train Accuracy", line=dict(color="#26a69a")))
    fig_hist.add_trace(go.Scatter(x=epochs_range, y=hist["val_accuracy"],
                                   name="Val Accuracy",   line=dict(color="#ef5350")))
    fig_hist.add_trace(go.Scatter(x=epochs_range, y=hist["loss"],
                                   name="Train Loss", line=dict(color="#42a5f5", dash="dot")))
    fig_hist.add_trace(go.Scatter(x=epochs_range, y=hist["val_loss"],
                                   name="Val Loss", line=dict(color="#ffa726", dash="dot")))
    fig_hist.update_layout(title="LSTM Training Curves",
                            height=380, template="plotly_dark",
                            margin=dict(l=0, r=0, t=80, b=0),
                            legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── LSTM Actual vs Predicted ──
    st.markdown("#### Actual vs Predicted (Test Set)")

    train_split = int(len(feat_df) * 0.8)
    window       = 60
    test_dates   = feat_df.index[train_split + window:]
    test_close   = feat_df["Close"].iloc[train_split + window:].values

    X_test_lstm  = lstm_r["X_test"]
    y_test_lstm  = lstm_r["y_test"]
    probs        = lstm_r["model"].predict(X_test_lstm, verbose=0).flatten()
    preds        = (probs > 0.5).astype(int)

    n = min(len(test_dates), len(probs))
    test_dates  = test_dates[:n]
    test_close  = test_close[:n]
    probs       = probs[:n]
    preds       = preds[:n]
    y_test_lstm = y_test_lstm[:n]

    correct = preds == y_test_lstm.astype(int)

    fig_lstm_pred = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Close Price + Direction Calls", "LSTM Up-Probability"),
    )

    fig_lstm_pred.add_trace(go.Scatter(
        x=test_dates, y=test_close,
        name="Close", line=dict(color="#90caf9", width=1.5),
    ), row=1, col=1)

    fig_lstm_pred.add_trace(go.Scatter(
        x=test_dates[correct], y=test_close[correct],
        mode="markers", name="Correct",
        marker=dict(color="#26a69a", size=5, symbol="circle"),
    ), row=1, col=1)

    fig_lstm_pred.add_trace(go.Scatter(
        x=test_dates[~correct], y=test_close[~correct],
        mode="markers", name="Wrong",
        marker=dict(color="#ef5350", size=5, symbol="x"),
    ), row=1, col=1)

    fig_lstm_pred.add_trace(go.Scatter(
        x=test_dates, y=probs,
        name="Up-Prob", line=dict(color="#ffa726", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,167,38,0.08)",
    ), row=2, col=1)
    fig_lstm_pred.add_hline(y=0.55, line_dash="dash", line_color="#26a69a",
                             annotation_text="Buy zone", row=2, col=1)
    fig_lstm_pred.add_hline(y=0.45, line_dash="dash", line_color="#ef5350",
                             annotation_text="Sell zone", row=2, col=1)
    fig_lstm_pred.add_hline(y=0.50, line_dash="dot",  line_color="gray", row=2, col=1)

    acc_pct = correct.mean() * 100
    fig_lstm_pred.update_layout(
        title=f"LSTM Test Set — Accuracy: {acc_pct:.1f}%",
        height=520, template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_lstm_pred, use_container_width=True)

# ── Tab 3: Future forecast ────────────────────────────────────────────────────
with tab3:
    st.markdown("<span style='margin-top:-1.2rem; margin-bottom:0.5rem; display:block; font-size:1.02rem; color:#f2a900; font-weight:600;'>ARIMA Forecast (Actual, Forecast, 95% CI)</span>", unsafe_allow_html=True)
    
    last_actual = feat_df["Close"].iloc[-60:]
    last_date   = feat_df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                   periods=forecast_days)
    fc   = arima_r["fc"]
    lo   = arima_r["lo"]
    hi   = arima_r["hi"]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=last_actual.index, y=last_actual,
        name="Actual (last 60 days)", line=dict(color="#2196f3"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=fc,
        name="ARIMA Forecast", line=dict(color="#ef5350", dash="dash"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor="rgba(239,83,80,0.12)",
        line=dict(color="rgba(239,83,80,0)"),
        name="95% CI",
    ))
    fig_fc.update_layout(
        height=420, template="plotly_dark",
        margin=dict(l=0, r=0, t=70, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# ── Tab 4: Side-by-side metrics + feature importance ──────────────────────────
with tab4:
    am = arima_r["metrics"]
    lm = lstm_r["metrics"]
    xm = xgb_r["metrics"]

    comparison = pd.DataFrame({
        "ARIMA": {
            "MAE (₹)":   f"{am['MAE']:.2f}",
            "RMSE (₹)":  f"{am['RMSE']:.2f}",
            "MAPE (%)":  f"{am['MAPE']:.2f}",
            "Accuracy":  "—",
            "Precision": "—",
            "Recall":    "—",
            "F1 Score":  "—",
        },
        "XGBoost": {
            "MAE (₹)":   "—",
            "RMSE (₹)":  "—",
            "MAPE (%)":  "—",
            "Accuracy":  f"{xm['Accuracy']*100:.1f}%",
            "Precision": f"{xm['Precision']*100:.1f}%",
            "Recall":    f"{xm['Recall']*100:.1f}%",
            "F1 Score":  f"{xm['F1']*100:.1f}%",
        },
        "LSTM": {
            "MAE (₹)":   "—",
            "RMSE (₹)":  "—",
            "MAPE (%)":  "—",
            "Accuracy":  f"{lm['Accuracy']*100:.1f}%",
            "Precision": f"{lm['Precision']*100:.1f}%",
            "Recall":    f"{lm['Recall']*100:.1f}%",
            "F1 Score":  f"{lm['F1']*100:.1f}%",
        },
    })
    
    # Center-align all columns in the dataframe
    styled_comparison = comparison.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
    st.dataframe(styled_comparison, use_container_width=True)

    # XGBoost feature importance chart — top 15
    st.subheader("Top 15 Most Important Features (XGBoost)")
    imp     = xgb_r["importances"]
    top15   = dict(list(imp.items())[:15])
    fig_imp = go.Figure(go.Bar(
        x=list(top15.values()),
        y=list(top15.keys()),
        orientation="h",
        marker_color="#42a5f5",
    ))
    fig_imp.update_layout(
        height=400, template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ── Tab 5: Walk-Forward Validation ───────────────────────────────────────────
with tab5:
    if "wfv_cache" not in st.session_state:
        st.session_state.wfv_cache = {}

    wfv_key = f"{display_ticker}|{years}|{forecast_days}|{display_sector}"

    st.markdown("""
    **Walk-Forward Validation** trains on all data up to each fold boundary and tests
    on the next unseen window — a more honest estimate of real-world accuracy than a
    single train/test split.
    """)

    if wfv_key not in st.session_state.wfv_cache:
        if st.button("Run Walk-Forward Validation", use_container_width=True):
            with st.spinner("Running walk-forward validation (5 folds)..."):
                st.session_state.wfv_cache[wfv_key] = walk_forward_validate(
                    feat_df, results["feature_cols"], feat_df["Close"],
                    lstm_r["model"], lstm_r["scaler"],
                    n_splits=5, initial_train_ratio=0.6, window=60,
                )
            st.rerun()
        else:
            st.info("Walk-forward validation is on-demand now. Click the button above to run it.")
    if wfv_key in st.session_state.wfv_cache:
        wfv_data = st.session_state.wfv_cache[wfv_key]
        folds_df = wfv_data["folds"]

        if folds_df.empty:
            st.warning("Not enough data to run walk-forward validation.")
        else:
            # Accuracy chart per fold
            fig_wfv = go.Figure()
            fig_wfv.add_trace(go.Bar(
                x=folds_df["Fold"].astype(str),
                y=folds_df["XGB Accuracy"],
                name="XGBoost (retrained each fold)",
                marker_color="#42a5f5",
            ))
            lstm_vals = folds_df["LSTM Accuracy"].dropna()
            if not lstm_vals.empty:
                fig_wfv.add_trace(go.Bar(
                    x=folds_df.loc[folds_df["LSTM Accuracy"].notna(), "Fold"].astype(str),
                    y=folds_df["LSTM Accuracy"].dropna(),
                    name="LSTM (fixed model, rolling eval)",
                    marker_color="#ef5350",
                ))
            fig_wfv.add_hline(y=50, line_dash="dash", line_color="gray",
                               annotation_text="Random baseline (50%)")
            fig_wfv.update_layout(
                title="Accuracy per Walk-Forward Fold",
                xaxis_title="Fold", yaxis_title="Accuracy (%)",
                height=380, template="plotly_dark", barmode="group",
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[40, 100]),
            )
            st.plotly_chart(fig_wfv, use_container_width=True)

            # Summary stats
            st.markdown("**Per-fold detail**")
            st.dataframe(folds_df.set_index("Fold"), use_container_width=True)

            xgb_mean = folds_df["XGB Accuracy"].mean()
            xgb_std  = folds_df["XGB Accuracy"].std()
            st.info(
                f"XGBoost walk-forward mean accuracy: **{xgb_mean:.1f}%** ± {xgb_std:.1f}%  "
                f"(vs single-split: {xgb_r['metrics']['Accuracy']*100:.1f}%)"
            )

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Disclaimer: This tool is for educational purposes only. Not financial advice.")
