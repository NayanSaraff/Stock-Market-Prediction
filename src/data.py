"""Data collection with yfinance and CSV cache fallback."""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str, suffix: str = "") -> str:
    safe = ticker.replace("/", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"{safe}{suffix}.csv")


def fetch_ohlcv(ticker: str, years: int = 5) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. Falls back to cached CSV on failure."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    path = _cache_path(ticker)
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.to_csv(path)
        return df
    except Exception:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        raise


def fetch_market_context(years: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch NIFTY 50 and USD/INR data."""
    nifty = fetch_ohlcv("^NSEI", years)
    usdinr = fetch_ohlcv("INR=X", years)
    return nifty, usdinr


def get_stock_info(ticker: str) -> dict:
    """Return key info fields for the ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "change_pct": info.get("regularMarketChangePercent", 0),
            "week52_high": info.get("fiftyTwoWeekHigh", 0),
            "week52_low": info.get("fiftyTwoWeekLow", 0),
            "volume": info.get("regularMarketVolume", 0),
            "market_cap": info.get("marketCap", 0),
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
        }
    except Exception:
        return {
            "current_price": 0, "change_pct": 0, "week52_high": 0,
            "week52_low": 0, "volume": 0, "market_cap": 0,
            "name": ticker, "sector": "Unknown",
        }
