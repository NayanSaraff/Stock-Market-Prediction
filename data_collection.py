# ============================================================
# PHASE 1 — DATA COLLECTION
# ============================================================
import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- Step 1: Choose ticker (change this for any NSE stock) ---
TICKER = "TCS.NS"
START  = "2018-01-01"
END    = "2024-12-31"

# --- Step 2: Download data ---
df = yf.download(TICKER, start=START, end=END, auto_adjust=True)

# --- Step 3: Keep only needed columns ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.sort_index(inplace=True)

# --- Step 4: Verify quality ---
print("=== Data Summary ===")
print(f"Ticker       : {TICKER}")
print(f"Trading days : {len(df)}")
print(f"Date range   : {df.index[0].date()} → {df.index[-1].date()}")
print(f"Missing vals : {df.isnull().sum().sum()}")
print(f"Close range  : ₹{df['Close'].min():.2f} → ₹{df['Close'].max():.2f}")
print(df.tail(3))

# --- Step 5: Save raw data ---
os.makedirs("data", exist_ok=True)
df.to_csv(f"data/{TICKER.replace('.','_')}_raw.csv")
print("Raw data saved.")
