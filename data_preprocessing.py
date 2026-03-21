# ============================================================
# PHASE 2 — PREPROCESSING
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data since df is undefined in this script
TICKER = "TCS.NS"
df = pd.read_csv(f"data/{TICKER.replace('.','_')}_raw.csv", index_col="Date", parse_dates=True)

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
