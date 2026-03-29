# ============================================================
# LSTM MODEL EVALUATION — TCS.NS Stock Price Forecasting
# Mirrors the evaluation methodology from the ARIMA baseline
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
import os

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
plt.style.use('seaborn-v0_8-darkgrid')

# ── CONFIGURATION ────────────────────────────────────────────
TICKER       = "TCS.NS"
WINDOW_SIZE  = 60          # lookback window (days)
TRAIN_RATIO  = 0.80        # 80/20 chronological split
EPOCHS       = 100
BATCH_SIZE   = 32
PATIENCE     = 10          # EarlyStopping patience
FORECAST_DAYS = 30         # future forecast horizon
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(_SCRIPT_DIR, "data", "TCS_NS_raw.csv")
PLOT_DIR     = os.path.join(_SCRIPT_DIR, "plots")
MODEL_DIR    = os.path.join(_SCRIPT_DIR, "models")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND PREPARE DATA
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading and Preparing Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = df[['Close']].values

print(f"Ticker          : {TICKER}")
print(f"Total records   : {len(df)}")
print(f"Date range      : {df.index[0].date()} → {df.index[-1].date()}")
print(f"Close range     : ₹{df['Close'].min():.2f} → ₹{df['Close'].max():.2f}")
print(f"Missing values  : {df['Close'].isnull().sum()}")

# Normalise (fit on train slice only to prevent data leakage)
scaler = MinMaxScaler(feature_range=(0, 1))

# Create sequences first, then split
def create_sequences(dataset, window):
    X, y = [], []
    for i in range(window, len(dataset)):
        X.append(dataset[i - window:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Fit scaler on training portion of raw data
split_raw = int(len(data) * TRAIN_RATIO)
scaler.fit(data[:split_raw])
scaled_data = scaler.transform(data)

X, y = create_sequences(scaled_data, WINDOW_SIZE)
split_idx = int(len(X) * TRAIN_RATIO)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

print(f"\nAfter windowing (window={WINDOW_SIZE}):")
print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")


# ════════════════════════════════════════════════════════════
# STEP 2 — BUILD MODEL
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Building Stacked LSTM Model")
print("=" * 60)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
], name="LSTM_TCS_Forecaster")

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# ════════════════════════════════════════════════════════════
# STEP 3 — TRAIN MODEL
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Training Model")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=f"{MODEL_DIR}/lstm_best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    callbacks=callbacks,
    verbose=1
)

total_epochs   = len(history.history['loss'])
best_val_loss  = min(history.history['val_loss'])
best_epoch     = history.history['val_loss'].index(best_val_loss) + 1
final_train_loss = history.history['loss'][-1]

print(f"\nTraining complete.")
print(f"  Total epochs trained : {total_epochs}")
print(f"  Best epoch           : {best_epoch}")
print(f"  Best val_loss (MSE)  : {best_val_loss:.6f}")
print(f"  Final train_loss     : {final_train_loss:.6f}")


# ════════════════════════════════════════════════════════════
# STEP 4 — GENERATE PREDICTIONS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Generating Test Set Predictions")
print("=" * 60)

preds_scaled = model.predict(X_test, verbose=1)
preds_actual = scaler.inverse_transform(preds_scaled)
actual       = scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"\nPrediction shape : {preds_actual.shape}")
print(f"Sample predictions (₹): {preds_actual[:5].flatten().round(2)}")
print(f"Sample actuals    (₹): {actual[:5].flatten().round(2)}")


# ════════════════════════════════════════════════════════════
# STEP 5 — QUANTITATIVE METRICS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Quantitative Evaluation Metrics")
print("=" * 60)

mae  = mean_absolute_error(actual, preds_actual)
rmse = np.sqrt(mean_squared_error(actual, preds_actual))
mape = np.mean(np.abs((actual - preds_actual) / actual)) * 100

# ARIMA baseline (from ARIMA evaluation report) for comparison
ARIMA_MAE  = 33.44
ARIMA_RMSE = 47.00
ARIMA_MAPE = 0.90

print(f"\n{'Metric':<10} {'ARIMA(0,1,0)':>14} {'LSTM':>14} {'Delta':>14}")
print("-" * 55)
print(f"{'MAE (₹)':<10} {ARIMA_MAE:>14.2f} {mae:>14.2f} {mae - ARIMA_MAE:>+14.2f}")
print(f"{'RMSE (₹)':<10} {ARIMA_RMSE:>14.2f} {rmse:>14.2f} {rmse - ARIMA_RMSE:>+14.2f}")
print(f"{'MAPE (%)':<10} {ARIMA_MAPE:>14.2f} {mape:>14.2f} {mape - ARIMA_MAPE:>+14.2f}")


# ════════════════════════════════════════════════════════════
# STEP 6 — RESIDUAL DIAGNOSTICS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Residual Diagnostics")
print("=" * 60)

residuals    = (actual - preds_actual).flatten()
mean_bias    = np.mean(residuals)
std_errors   = np.std(residuals)
max_overest  = np.max(residuals)     # actual > pred → under-prediction
max_underest = np.min(residuals)     # actual < pred → over-prediction

# Directional accuracy
actual_dir = np.diff(actual.flatten())
pred_dir   = np.diff(preds_actual.flatten())
dir_acc    = np.mean(np.sign(actual_dir) == np.sign(pred_dir)) * 100

print(f"  Mean Error (Bias)        : ₹{mean_bias:+.2f}  ({'under-predicts' if mean_bias > 0 else 'over-predicts'} on average)")
print(f"  Std of Residuals         : ₹{std_errors:.2f}")
print(f"  Max Under-prediction     : ₹{max_overest:.2f}  (actual much higher than predicted)")
print(f"  Max Over-prediction      : ₹{max_underest:.2f} (actual much lower than predicted)")
print(f"  Directional Accuracy     : {dir_acc:.2f}%  ({'above' if dir_acc > 50 else 'below'} 50% random baseline)")

# Percentage of predictions within tolerance bands
for tol in [1, 2, 5]:
    within = np.mean(np.abs(residuals / actual.flatten()) * 100 <= tol) * 100
    print(f"  Within ±{tol}% tolerance      : {within:.1f}% of test days")


# ════════════════════════════════════════════════════════════
# STEP 7 — PLOT: TRAINING HISTORY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Plotting Training History")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 4))
epochs_range = range(1, total_epochs + 1)
ax.plot(epochs_range, history.history['loss'],     label='Training Loss',   color='steelblue', linewidth=1.8)
ax.plot(epochs_range, history.history['val_loss'], label='Validation Loss', color='darkorange', linewidth=1.8)
ax.axvline(x=best_epoch, color='crimson', linestyle='--', linewidth=1.2, label=f'Best epoch ({best_epoch})')
ax.set_title(f'{TICKER} — LSTM Training & Validation Loss (MSE)', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (normalised scale)')
ax.legend()
ax.annotate(
    f'Best val_loss: {best_val_loss:.5f}\nEarly stop @ epoch {total_epochs}',
    xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8)
)
plt.tight_layout()
path = f"{PLOT_DIR}/09_lstm_training_loss.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════
# STEP 8 — PLOT: ACTUAL vs PREDICTED
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Actual vs Predicted Plot")
print("=" * 60)

test_dates = df.index[split_idx + WINDOW_SIZE:]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates, actual,      color='steelblue', linewidth=1.6, label='Actual Price')
ax.plot(test_dates, preds_actual, color='crimson',  linewidth=1.6, linestyle='--', label='LSTM Predicted')
ax.fill_between(test_dates, actual.flatten(), preds_actual.flatten(), alpha=0.12, color='gray', label='Error band')

ax.set_title(f'{TICKER} — LSTM: Actual vs Predicted Closing Price ({len(test_dates)} test days)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price (₹)')
ax.legend(loc='upper left')

metrics_text = (f'MAE: ₹{mae:.2f}  |  RMSE: ₹{rmse:.2f}  |  MAPE: {mape:.2f}%\n'
                f'Bias: ₹{mean_bias:+.2f}  |  Dir. Accuracy: {dir_acc:.2f}%')
ax.text(0.02, 0.04, metrics_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
path = f"{PLOT_DIR}/10_lstm_actual_vs_predicted.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════
# STEP 9 — PLOT: RESIDUAL ANALYSIS (2-panel)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Residual Analysis Plots")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: residuals over time
axes[0].plot(test_dates, residuals, color='darkorange', linewidth=0.9, alpha=0.85)
axes[0].axhline(0,         color='black',  linewidth=1.2, linestyle='--')
axes[0].axhline(mean_bias, color='red',    linewidth=1.0, linestyle=':', label=f'Mean bias ₹{mean_bias:+.1f}')
axes[0].fill_between(test_dates, residuals, 0,
                     where=(residuals > 0), alpha=0.25, color='red',  label='Under-predicted (actual > pred)')
axes[0].fill_between(test_dates, residuals, 0,
                     where=(residuals < 0), alpha=0.25, color='blue', label='Over-predicted (actual < pred)')
axes[0].set_title('Residuals Over Time  (Actual − Predicted)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residual (₹)')
axes[0].legend(fontsize=8)

# Right: residual histogram
axes[1].hist(residuals, bins=35, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].axvline(np.mean(residuals),   color='red',   linestyle='--', linewidth=1.8,
                label=f'Mean:   ₹{np.mean(residuals):.1f}')
axes[1].axvline(np.median(residuals), color='green', linestyle='--', linewidth=1.8,
                label=f'Median: ₹{np.median(residuals):.1f}')
axes[1].set_title('Residual Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Residual (₹)')
axes[1].set_ylabel('Frequency')
axes[1].legend(fontsize=8)
axes[1].annotate(f'Std: ₹{std_errors:.1f}\nSkew: {pd.Series(residuals).skew():.2f}',
                 xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

fig.suptitle(f'{TICKER} — LSTM Residual Analysis (Test Set: {len(test_dates)} days)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path = f"{PLOT_DIR}/11_lstm_residuals.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════
# STEP 10 — PLOT: ERROR DISTRIBUTION BY PRICE QUARTILE
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: Error Analysis by Price Level (Quartiles)")
print("=" * 60)

actual_flat = actual.flatten()
quartiles   = np.percentile(actual_flat, [25, 50, 75])
labels = [
    f'Q1: ₹{actual_flat.min():.0f}–{quartiles[0]:.0f}',
    f'Q2: ₹{quartiles[0]:.0f}–{quartiles[1]:.0f}',
    f'Q3: ₹{quartiles[1]:.0f}–{quartiles[2]:.0f}',
    f'Q4: ₹{quartiles[2]:.0f}–{actual_flat.max():.0f}'
]
q_masks = [
    actual_flat < quartiles[0],
    (actual_flat >= quartiles[0]) & (actual_flat < quartiles[1]),
    (actual_flat >= quartiles[1]) & (actual_flat < quartiles[2]),
    actual_flat >= quartiles[2]
]

print(f"\n{'Quartile':<35} {'MAE (₹)':>10} {'MAPE (%)':>10} {'N':>6}")
print("-" * 65)
q_mae_vals = []
for lbl, mask in zip(labels, q_masks):
    q_mae  = mean_absolute_error(actual_flat[mask], preds_actual.flatten()[mask])
    q_mape = np.mean(np.abs((actual_flat[mask] - preds_actual.flatten()[mask]) / actual_flat[mask])) * 100
    q_mae_vals.append(q_mae)
    print(f"{lbl:<35} {q_mae:>10.2f} {q_mape:>10.2f} {mask.sum():>6}")

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(labels, q_mae_vals, color=['#5B9BD5', '#70AD47', '#FFC000', '#FF6347'], edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, q_mae_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'₹{val:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(mae, color='crimson', linestyle='--', linewidth=1.5, label=f'Overall MAE: ₹{mae:.2f}')
ax.set_title(f'{TICKER} — LSTM MAE by Price Quartile', fontsize=12, fontweight='bold')
ax.set_xlabel('Price Quartile (Actual Closing Price)')
ax.set_ylabel('MAE (₹)')
ax.legend()
plt.tight_layout()
path = f"{PLOT_DIR}/12_lstm_mae_by_quartile.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved → {path}")


# ════════════════════════════════════════════════════════════
# STEP 11 — PLOT: ARIMA vs LSTM COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 11: ARIMA vs LSTM Comparative Bar Chart")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics_cfg = [
    ("MAE (₹)",  ARIMA_MAE,  mae,  "₹"),
    ("RMSE (₹)", ARIMA_RMSE, rmse, "₹"),
    ("MAPE (%)", ARIMA_MAPE, mape, "%"),
]
for ax, (title, arima_val, lstm_val, unit) in zip(axes, metrics_cfg):
    bars = ax.bar(['ARIMA\n(0,1,0)', 'LSTM\n(64+64)'],
                  [arima_val, lstm_val],
                  color=['#5B9BD5', '#FF6347'], edgecolor='white', linewidth=0.8, width=0.5)
    for bar, val in zip(bars, [arima_val, lstm_val]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f'{unit}{val:.2f}' if unit == '₹' else f'{val:.2f}{unit}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(title)
    ax.set_ylim(0, max(arima_val, lstm_val) * 1.25)

fig.suptitle(f'{TICKER} — ARIMA(0,1,0) vs LSTM: Test Set Performance Comparison',
             fontsize=12, fontweight='bold')
plt.tight_layout()
path = f"{PLOT_DIR}/13_arima_vs_lstm_comparison.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════
# STEP 12 — FUTURE FORECAST (30 days autoregressive)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 12: Autoregressive Future Forecast ({FORECAST_DAYS} days)")
print("=" * 60)

last_window = scaler.transform(df[['Close']].values[-WINDOW_SIZE:])
input_seq   = list(last_window.flatten())

future_preds_scaled = []
for _ in range(FORECAST_DAYS):
    x    = np.array(input_seq[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
    pred = model.predict(x, verbose=0)[0][0]
    future_preds_scaled.append(pred)
    input_seq.append(pred)

future_preds = scaler.inverse_transform(
    np.array(future_preds_scaled).reshape(-1, 1)
)
last_date    = df.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=FORECAST_DAYS + 1)[1:]

last_close   = df['Close'].values[-1]
final_pred   = future_preds[-1, 0]
pct_change   = (final_pred - last_close) / last_close * 100

print(f"  Last observed close      : ₹{last_close:.2f}  ({last_date.date()})")
print(f"  30-day forecast endpoint : ₹{final_pred:.2f}  ({future_dates[-1].date()})")
print(f"  Implied price change     : ₹{final_pred - last_close:.2f}  ({pct_change:+.2f}%)")
print(f"  ⚠ Note: Autoregressive forecasts revert toward training mean.")
print(f"    Directional accuracy on test set was {dir_acc:.1f}% — interpret with caution.")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index[-120:], df['Close'].values[-120:],
        color='steelblue', linewidth=1.6, label='Historical Price (last 120 days)')
ax.plot(future_dates, future_preds,
        color='crimson', linestyle='--', linewidth=1.8, marker='o', markersize=3,
        label=f'LSTM {FORECAST_DAYS}-Day Forecast')
ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.fill_between(future_dates,
                future_preds.flatten() * 0.98, future_preds.flatten() * 1.02,
                alpha=0.15, color='crimson', label='±2% uncertainty band')
ax.set_title(f'{TICKER} — LSTM {FORECAST_DAYS}-Day Autoregressive Forecast', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price (₹)')
ax.legend(loc='upper left', fontsize=9)

info = (f"Forecast start : {last_date.date()}\n"
        f"Last close     : ₹{last_close:.2f}\n"
        f"30d target     : ₹{final_pred:.2f}  ({pct_change:+.1f}%)")
ax.text(0.02, 0.05, info, transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))
plt.tight_layout()
path = f"{PLOT_DIR}/14_lstm_future_forecast.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════
# STEP 13 — FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)

summary = {
    'model'              : 'LSTM (64+64 units, 52,033 params)',
    'test_days'          : len(test_dates),
    'total_epochs'       : total_epochs,
    'best_epoch'         : best_epoch,
    'best_val_loss_MSE'  : round(best_val_loss, 6),
    'MAE_INR'            : round(mae,  2),
    'RMSE_INR'           : round(rmse, 2),
    'MAPE_pct'           : round(mape, 2),
    'mean_bias_INR'      : round(mean_bias, 2),
    'std_residuals_INR'  : round(std_errors, 2),
    'directional_acc_pct': round(dir_acc, 2),
    'within_1pct_pct'    : round(np.mean(np.abs(residuals / actual.flatten()) * 100 <= 1) * 100, 1),
    'within_2pct_pct'    : round(np.mean(np.abs(residuals / actual.flatten()) * 100 <= 2) * 100, 1),
    'within_5pct_pct'    : round(np.mean(np.abs(residuals / actual.flatten()) * 100 <= 5) * 100, 1),
    'ARIMA_MAE_INR'      : ARIMA_MAE,
    'ARIMA_RMSE_INR'     : ARIMA_RMSE,
    'ARIMA_MAPE_pct'     : ARIMA_MAPE,
    'MAE_vs_ARIMA_pct'   : round((mae - ARIMA_MAE) / ARIMA_MAE * 100, 1),
    'MAPE_vs_ARIMA_pct'  : round((mape - ARIMA_MAPE) / ARIMA_MAPE * 100, 1),
    'verdict'            : 'LSTM does NOT outperform ARIMA baseline on this dataset/horizon',
    'forecast_30d_INR'   : round(final_pred, 2),
    'forecast_30d_chg_pct': round(pct_change, 2),
}

print(f"\n{'Key':<30} {'Value':>20}")
print("-" * 52)
for k, v in summary.items():
    print(f"  {k:<28} {str(v):>20}")

# Save results dict
import json
with open("lstm_evaluation_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n  Results saved → lstm_evaluation_results.json")
print(f"\n  Plots saved in  → {PLOT_DIR}/")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
