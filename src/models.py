"""ARIMA and LSTM model training, evaluation, and forecasting."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def train_arima(close_series: pd.Series, train_ratio: float = 0.8):
    """Fit auto_arima on training portion, return model + split index."""
    import pmdarima as pm
    split = int(len(close_series) * train_ratio)
    train = close_series.iloc[:split]
    model = pm.auto_arima(
        train, d=1, seasonal=False,
        stepwise=True, suppress_warnings=True,
        error_action="ignore",
    )
    return model, split


def arima_rolling_forecast(close_series: pd.Series,
                            model,
                            split: int) -> np.ndarray:
    """One-step-ahead rolling forecast on the test set."""
    history = list(close_series.iloc[:split])
    preds = []
    for val in close_series.iloc[split:]:
        fc = float(np.asarray(model.predict(n_periods=1)).flat[0])
        preds.append(fc)
        model.update([val])
        history.append(val)
    return np.array(preds)


def arima_future_forecast(model, n_periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast n_periods ahead with 95% confidence interval."""
    fc, conf = model.predict(n_periods=n_periods, return_conf_int=True)
    return fc, conf[:, 0], conf[:, 1]


def arima_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm_sequences(scaled: np.ndarray,
                          targets: np.ndarray,
                          window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    n = len(scaled) - window
    n_features = scaled.shape[1]
    X = np.zeros((n, window, n_features), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        X[i] = scaled[i: i + window]
        y[i] = targets[i + window]
    return X, y


def build_lstm_model(input_shape: tuple):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(df: pd.DataFrame,
               feature_cols: list[str],
               window: int = 60,
               train_ratio: float = 0.8,
               epochs: int = 100,
               batch_size: int = 32,
               patience: int = 10,
               model_path: str = "lstm_best.keras"):
    """
    Train LSTM classifier. Returns (model, scaler, history, split_idx, X_test, y_test).
    Scaler is fit on training data only.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    features = df[feature_cols].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    features.bfill(inplace=True)
    features = features.values
    targets  = df["target"].values

    split = int(len(features) * train_ratio)
    train_feat = features[:split]
    test_feat  = features[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled  = scaler.transform(test_feat)

    X_train, y_train = build_lstm_sequences(train_scaled, targets[:split], window)
    X_test,  y_test  = build_lstm_sequences(test_scaled,  targets[split:], window)

    model = build_lstm_model((window, len(feature_cols)))

    callbacks = [
        EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )

    return model, scaler, history, split, X_test, y_test


def lstm_predict_latest(model, scaler, df: pd.DataFrame,
                         feature_cols: list[str], window: int = 60) -> float:
    """Return probability (0-1) for the next trading day being up."""
    features = df[feature_cols].values[-window:]
    scaled   = scaler.transform(features)
    X        = scaled.reshape(1, window, len(feature_cols))
    prob     = float(model.predict(X, verbose=0)[0][0])
    return prob


def lstm_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    return {
        "Accuracy":  accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall":    recall_score(y_test, preds, zero_division=0),
        "F1":        f1_score(y_test, preds, zero_division=0),
    }
