"""Feature engineering using the `ta` library."""

import numpy as np
import pandas as pd
import ta


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

    # Returns
    df["daily_return"]    = close.pct_change() * 100
    df["volume_change"]   = vol.pct_change() * 100

    return df


def merge_market_context(df: pd.DataFrame,
                          nifty: pd.DataFrame,
                          usdinr: pd.DataFrame) -> pd.DataFrame:
    """Merge NIFTY and USD/INR daily returns into the main dataframe."""
    df = df.copy()

    nifty_ret  = nifty["Close"].pct_change() * 100
    usdinr_ret = usdinr["Close"].pct_change() * 100

    nifty_ret.name  = "nifty_return"
    usdinr_ret.name = "usdinr_return"

    df = df.join(nifty_ret,  how="left")
    df = df.join(usdinr_ret, how="left")

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: 1 if next-day close > today, else 0."""
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame,
                          nifty: pd.DataFrame,
                          usdinr: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: indicators → context → target → dropna."""
    df = add_technical_indicators(df)
    df = merge_market_context(df, nifty, usdinr)
    df = add_target(df)
    # Replace infinities introduced by pct_change on zero values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "sma20", "sma50",
    "daily_return", "volume_change",
    "nifty_return", "usdinr_return",
    "Open", "High", "Low", "Close", "Volume",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]
