"""Feature engineering using the `ta` library."""

import numpy as np
import pandas as pd
import ta

# Sector index tickers for each NSE sector
SECTOR_INDEX_MAP = {
    "IT":                 "^CNXIT",
    "Banking & Finance":  "^NSEBANK",
    "Energy & Oil":       "^CNXENERGY",
    "Consumer & FMCG":    "^CNXFMCG",
    "Auto":               "^CNXAUTO",
    "Pharma":             "^CNXPHARMA",
    "Metals & Mining":    "^CNXMETAL",
}


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to OHLCV dataframe."""
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # MACD
    macd_ind = ta.trend.MACD(close=close)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close, window=20)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()

    # Moving Averages
    df["sma20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df["sma50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

    # ATR — measures daily volatility, helps LSTM know when to be uncertain
    df["atr"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # OBV — volume-price momentum, strong leading indicator
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=vol
    ).on_balance_volume()

    # Stochastic Oscillator — momentum indicator like RSI but uses H/L range
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Lagged closes — explicit price memory the LSTM can use directly
    df["close_lag1"] = close.shift(1)
    df["close_lag2"] = close.shift(2)
    df["close_lag5"] = close.shift(5)

    # Day-of-week one-hot — avoids false ordinal relationship (Mon≠0 < Fri≠4)
    dow = pd.to_datetime(df.index).dayofweek
    for i, name in enumerate(["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]):
        df[name] = (dow == i).astype(float)

    # Returns
    df["daily_return"]  = close.pct_change() * 100
    df["volume_change"] = vol.pct_change() * 100

    return df


def add_earnings_proximity(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add earnings_proximity flag: 1 if the trading day is within 5 days
    before or 2 days after a quarterly earnings release, else 0.

    Tries yfinance earnings dates first; falls back to a calendar-quarter
    heuristic (last 5 trading days of each quarter) if unavailable.
    """
    import yfinance as yf

    df = df.copy()
    dates = pd.to_datetime(df.index, utc=True).tz_localize(None)
    proximity = pd.Series(0, index=df.index, dtype=float)

    earnings_dates = []
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            ed_idx = pd.to_datetime(ed.index)
            if ed_idx.tz is not None:
                ed_idx = ed_idx.tz_localize(None)
            earnings_dates = ed_idx.normalize().tolist()
    except Exception:
        pass

    if earnings_dates:
        for ed in earnings_dates:
            ed_ts = pd.to_datetime(ed, utc=True).tz_localize(None)
            mask = (dates >= ed_ts - pd.Timedelta(days=7)) & \
                   (dates <= ed_ts + pd.Timedelta(days=3))
            proximity[mask] = 1.0
    else:
        # Heuristic fallback: last 5 trading days of Mar/Jun/Sep/Dec
        for date in dates:
            if date.month in (3, 6, 9, 12):
                month_end = date + pd.offsets.BMonthEnd(0)
                days_to_end = (month_end - date).days
                if days_to_end <= 7:
                    proximity[date] = 1.0

    df["earnings_proximity"] = proximity
    return df


def merge_market_context(df: pd.DataFrame,
                          nifty: pd.DataFrame,
                          usdinr: pd.DataFrame,
                          sector_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge NIFTY, USD/INR, and optional sector index returns."""
    df = df.copy()

    nifty_ret  = nifty["Close"].pct_change() * 100
    usdinr_ret = usdinr["Close"].pct_change() * 100

    nifty_ret.name  = "nifty_return"
    usdinr_ret.name = "usdinr_return"

    df = df.join(nifty_ret,  how="left")
    df = df.join(usdinr_ret, how="left")

    if sector_df is not None and not sector_df.empty:
        sector_ret = sector_df["Close"].pct_change() * 100
        sector_ret.name = "sector_return"
        df = df.join(sector_ret, how="left")

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: 1 if next-day close > today, else 0."""
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame,
                          nifty: pd.DataFrame,
                          usdinr: pd.DataFrame,
                          sector_df: pd.DataFrame | None = None,
                          ticker: str | None = None) -> pd.DataFrame:
    """Full pipeline: indicators → context → earnings → target → dropna."""
    df = add_technical_indicators(df)
    df = merge_market_context(df, nifty, usdinr, sector_df)
    if ticker:
        df = add_earnings_proximity(df, ticker)
    df = add_target(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    # Original indicators
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "sma20", "sma50",
    # New indicators
    "atr", "obv", "stoch_k", "stoch_d",
    # Lagged prices
    "close_lag1", "close_lag2", "close_lag5",
    # Calendar — one-hot day of week (no false ordinal relationship)
    "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri",
    # Earnings proximity flag
    "earnings_proximity",
    # Returns
    "daily_return", "volume_change",
    # Market context
    "nifty_return", "usdinr_return", "sector_return",
    # Raw OHLCV
    "Open", "High", "Low", "Close", "Volume",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]
