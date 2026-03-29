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
from src.features import build_feature_matrix, get_feature_cols
from src.models   import (
    train_arima, arima_rolling_forecast, arima_future_forecast, arima_metrics,
    train_lstm, lstm_predict_latest, lstm_metrics,
)
from src.signals  import generate_signal, trend_label

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.signal-card {
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin: 12px 0;
}
.signal-text { font-size: 3rem; font-weight: 900; letter-spacing: 2px; }
.metric-label { font-size: 0.78rem; color: #aaa; text-transform: uppercase; }
.metric-val   { font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

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

# Flat label → ticker map for lookup
_LABEL_TO_TICKER = {
    f"{name} ({sector})": tick
    for sector, stocks in NSE_STOCKS.items()
    for name, tick in stocks.items()
}
_LABELS = list(_LABEL_TO_TICKER.keys())

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("NSE Predictor")

    sector_choice = st.selectbox("Sector", ["All"] + list(NSE_STOCKS.keys()))
    if sector_choice == "All":
        filtered_labels = _LABELS
    else:
        filtered_labels = [
            f"{name} ({sector_choice})"
            for name in NSE_STOCKS[sector_choice]
        ]

    selected_label = st.selectbox("Stock", filtered_labels)
    ticker         = _LABEL_TO_TICKER[selected_label]
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

# ── auto-refresh countdown ────────────────────────────────────────────────────
if auto_refresh:
    refresh_placeholder = st.sidebar.empty()
    if "next_refresh" not in st.session_state:
        st.session_state.next_refresh = time.time() + 300
    remaining = int(st.session_state.next_refresh - time.time())
    if remaining <= 0:
        st.session_state.next_refresh = time.time() + 300
        st.rerun()
    refresh_placeholder.info(f"Next refresh in {remaining}s")


# ── cached data fetchers ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(ticker: str, years: int):
    df              = fetch_ohlcv(ticker, years)
    nifty, usdinr   = fetch_market_context(years)
    info            = get_stock_info(ticker)
    feat_df         = build_feature_matrix(df, nifty, usdinr)
    feature_cols    = get_feature_cols(feat_df)
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


# ── main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(ticker: str, years: int, forecast_days: int):
    results = {}

    with st.spinner("Fetching market data..."):
        df, feat_df, feature_cols, info = load_data(ticker, years)
    results["df"]           = df
    results["feat_df"]      = feat_df
    results["feature_cols"] = feature_cols
    results["info"]         = info

    # ── ARIMA ──
    with st.spinner("Training ARIMA model (this may take ~30 seconds)..."):
        close = feat_df["Close"]
        arima_model, arima_split = train_arima(close)
        arima_preds  = arima_rolling_forecast(close, arima_model, arima_split)
        arima_actual = close.iloc[arima_split:].values
        arima_met    = arima_metrics(arima_actual, arima_preds)
        fc, lo, hi   = arima_future_forecast(arima_model, forecast_days)
    results["arima"] = dict(
        model=arima_model, split=arima_split,
        preds=arima_preds, actual=arima_actual,
        metrics=arima_met, fc=fc, lo=lo, hi=hi,
        dates=feat_df.index,
    )

    # ── LSTM ──
    model_path = os.path.join("cache", f"{ticker.replace('.','_')}_lstm.keras")
    with st.spinner("Training LSTM model (60–120 seconds depending on data size)..."):
        lstm_model, scaler, history, split_idx, X_test, y_test = train_lstm(
            feat_df, feature_cols,
            window=60, train_ratio=0.8,
            epochs=100, batch_size=32, patience=10,
            model_path=model_path,
        )
        lstm_prob  = lstm_predict_latest(lstm_model, scaler, feat_df, feature_cols)
        lstm_met   = lstm_metrics(lstm_model, X_test, y_test)
    results["lstm"] = dict(
        model=lstm_model, scaler=scaler, history=history,
        prob=lstm_prob, metrics=lstm_met,
        X_test=X_test, y_test=y_test,
    )

    return results


# ── trigger pipeline ──────────────────────────────────────────────────────────
if run_btn or (auto_refresh and st.session_state.results is None):
    st.session_state.results     = run_pipeline(ticker, years, forecast_days)
    st.session_state.last_ticker = ticker

results = st.session_state.results

if results is None:
    st.info("Configure settings in the sidebar and click **Run Prediction** to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Live Stock Info
# ─────────────────────────────────────────────────────────────────────────────
info   = results["info"]
feat_df = results["feat_df"]
df      = results["df"]

st.markdown(f"## {info['name']}  `{ticker}`")
st.caption(f"Sector: {info['sector']}")

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

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Signal Card
# ─────────────────────────────────────────────────────────────────────────────
arima_r   = results["arima"]
lstm_r    = results["lstm"]

lstm_prob    = lstm_r["prob"]
arima_next   = float(arima_r["fc"][0])
rsi_val      = safe_last(feat_df["rsi"])
macd_hist_val = safe_last(feat_df["macd_hist"])

signal, sig_color = generate_signal(lstm_prob, rsi_val, macd_hist_val,
                                     arima_next, current_price)
trend_txt, trend_color = trend_label(lstm_prob)

bg_map = {
    "green": "#1b5e20", "lightgreen": "#2e7d32",
    "orange": "#e65100", "red": "#b71c1c", "gray": "#37474f",
}
bg = bg_map.get(sig_color, "#37474f")

st.markdown(f"""
<div class="signal-card" style="background:{bg};">
  <div class="signal-text" style="color:white;">{signal}</div>
  <div style="color:rgba(255,255,255,0.85); font-size:1.1rem; margin-top:8px;">
    LSTM Confidence: <b>{lstm_prob*100:.1f}%</b> &nbsp;|&nbsp;
    Trend: <span style="color:{trend_color};font-weight:700;">{trend_txt}</span>
  </div>
  <div style="color:rgba(255,255,255,0.75); font-size:1rem; margin-top:6px;">
    ARIMA next-day price forecast: <b>₹{arima_next:,.2f}</b> &nbsp;|&nbsp;
    RSI: <b>{rsi_val:.1f}</b>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Interactive Price Chart (Candlestick + Volume)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Price Chart — Last 6 Months")
chart_df = feat_df.iloc[-130:]  # ~6 months

fig_price = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.75, 0.25], vertical_spacing=0.03,
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
    height=520, xaxis_rangeslider_visible=False,
    template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig_price, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Technical Indicators")
ind_df = feat_df.iloc[-260:]  # ~1 year for indicators

col_rsi, col_macd, col_bbw = st.columns(3)

with col_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=ind_df.index, y=ind_df["rsi"],
                                  line=dict(color="#7c4dff"), name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title="RSI (14)", height=280,
                           template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0),
                           showlegend=False)
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_macd:
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=ind_df.index, y=ind_df["macd"],
                                   line=dict(color="#2196f3"), name="MACD"))
    fig_macd.add_trace(go.Scatter(x=ind_df.index, y=ind_df["macd_signal"],
                                   line=dict(color="orange"), name="Signal"))
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                   for v in ind_df["macd_hist"].fillna(0)]
    fig_macd.add_trace(go.Bar(x=ind_df.index, y=ind_df["macd_hist"],
                               marker_color=hist_colors, name="Histogram"))
    fig_macd.update_layout(title="MACD", height=280,
                            template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", y=1.15, font_size=10))
    st.plotly_chart(fig_macd, use_container_width=True)

with col_bbw:
    fig_bbw = go.Figure()
    fig_bbw.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_width"],
                                  fill="tozeroy", line=dict(color="#ff6f00"),
                                  name="BB Width"))
    fig_bbw.update_layout(title="Bollinger Band Width (Squeeze Indicator)",
                           height=280, template="plotly_dark",
                           margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig_bbw, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Model Performance & Forecast
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Model Performance & Forecast")

tab1, tab2, tab3, tab4 = st.tabs(
    ["ARIMA Evaluation", "LSTM Training", "Future Forecast", "Metrics Comparison"]
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
                             height=380, template="plotly_dark",
                             margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_arima, use_container_width=True)

    m = arima_r["metrics"]
    ca, cb, cc = st.columns(3)
    ca.metric("MAE",  f"₹{m['MAE']:.2f}")
    cb.metric("RMSE", f"₹{m['RMSE']:.2f}")
    cc.metric("MAPE", f"{m['MAPE']:.2f}%")

# ── Tab 2: LSTM training curves ───────────────────────────────────────────────
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
                            margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 3: Future forecast ────────────────────────────────────────────────────
with tab3:
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
    fig_fc.update_layout(title=f"ARIMA {forecast_days}-Day Forecast with 95% CI",
                          height=420, template="plotly_dark",
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_fc, use_container_width=True)

# ── Tab 4: Side-by-side metrics ───────────────────────────────────────────────
with tab4:
    am = arima_r["metrics"]
    lm = lstm_r["metrics"]

    comparison = pd.DataFrame({
        "ARIMA": {
            "MAE (₹)":    f"{am['MAE']:.2f}",
            "RMSE (₹)":   f"{am['RMSE']:.2f}",
            "MAPE (%)":   f"{am['MAPE']:.2f}",
            "Accuracy":   "—",
            "Precision":  "—",
            "Recall":     "—",
            "F1 Score":   "—",
        },
        "LSTM": {
            "MAE (₹)":    "—",
            "RMSE (₹)":   "—",
            "MAPE (%)":   "—",
            "Accuracy":   f"{lm['Accuracy']*100:.1f}%",
            "Precision":  f"{lm['Precision']*100:.1f}%",
            "Recall":     f"{lm['Recall']*100:.1f}%",
            "F1 Score":   f"{lm['F1']*100:.1f}%",
        },
    })
    st.dataframe(comparison, use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Disclaimer: This tool is for educational purposes only. Not financial advice.")
