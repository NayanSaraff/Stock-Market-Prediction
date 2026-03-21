Phase 1 — Data Collection
1 Install required libraries
yfinance, pandas, numpy, matplotlib, scikit-learn, tensorflow, statsmodels, pmdarima, streamlit
2 Accept ticker symbol as input (e.g. TCS.NS)
This makes the system work for any NSE stock
3 Download 5–7 years of daily OHLCV data using yfinance
start="2018-01-01", end="2024-12-31", auto_adjust=True
4 Verify data quality — shape, date range, missing values
Print summary statistics before proceeding
5 Save raw CSV to /data folder
Keeps a backup; useful for Kaggle dataset alternative

Phase 2 — Preprocessing
1 Extract Close price column only
Target variable for both ARIMA and LSTM
2 Handle missing values using forward fill
Accounts for market holidays and weekends
3 Chronological train/test split — 80% / 20%
Never shuffle time series data
4 Normalize using MinMaxScaler → [0, 1] range
Fit scaler on train only, transform both train and test
5 Create sliding window sequences (window = 60 days)
60 days input → predict day 61. Required for LSTM input shape
6 Reshape X to 3D array for LSTM
(samples, timesteps, features) — e.g. (1400, 60, 1)


data_collection.py
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


data_preprocessing.py
# ============================================================
# PHASE 2 — PREPROCESSING
# ============================================================
from sklearn.preprocessing import MinMaxScaler

# --- Step 1: Extract Close price ---
close = df[['Close']].copy()

# --- Step 2: Forward fill missing values ---
close.ffill(inplace=True)

# --- Step 3: Train / Test split (80/20, chronological) ---
split = int(len(close) * 0.80)
train_data = close[:split]
test_data  = close[split:]

print(f"Train size : {len(train_data)} days")
print(f"Test size  : {len(test_data)} days")

# --- Step 4: Normalize (fit on train only!) ---
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# --- Step 5: Create sliding window sequences ---
WINDOW = 60   # use past 60 days to predict next day

def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, WINDOW)
X_test,  y_test  = make_sequences(test_scaled,  WINDOW)

# --- Step 6: Reshape for LSTM input (samples, timesteps, features) ---
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

print(f"\nX_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print("Preprocessing complete. Ready for ARIMA and LSTM.")