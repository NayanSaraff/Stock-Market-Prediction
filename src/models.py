"""ARIMA, LSTM, and XGBoost model training, evaluation, and forecasting."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")



# ── ARIMA ─────────────────────────────────────────────────────────────────────

def train_arima(close_series: pd.Series, train_ratio: float = 0.8):
    """Fast ARIMA (no heavy auto search)."""
    from statsmodels.tsa.arima.model import ARIMA

    split = int(len(close_series) * train_ratio)
    train = close_series.iloc[:split]

    # 🔥 Fixed order (VERY fast)
    model = ARIMA(train, order=(1, 1, 1)).fit()

    return model, split


def arima_rolling_forecast(close_series: pd.Series,
                            model,
                            split: int) -> tuple[np.ndarray, object]:
    """One-step-ahead rolling forecast on the test set."""
    working_model = model
    preds = []
    for val in close_series.iloc[split:]:
        fc = float(np.asarray(working_model.forecast(steps=1)).flat[0])
        preds.append(fc)
        # Append each observed point without refitting full ARIMA params.
        working_model = working_model.append([float(val)], refit=False)
    return np.array(preds), working_model


def arima_future_forecast(model, n_periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast n_periods ahead with 95% confidence interval."""
    forecast_res = model.get_forecast(steps=n_periods)
    fc = np.asarray(forecast_res.predicted_mean)
    conf = np.asarray(forecast_res.conf_int())
    return fc, conf[:, 0], conf[:, 1]


def arima_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(df: pd.DataFrame,
                  feature_cols: list[str],
                  train_ratio: float = 0.8):
    """
    Train XGBoost classifier on flat (non-sequential) features.
    Uses class weights to correct for the natural ~53% up-day bias.
    Returns (model, X_test, y_test, feature_importances_dict).
    """
    from xgboost import XGBClassifier

    features = df[feature_cols].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    features.bfill(inplace=True)
    features = features.values
    targets  = df["target"].values

    split    = int(len(features) * train_ratio)
    X_train, X_test   = features[:split], features[split:]
    y_train, y_test   = targets[:split],  targets[split:]

    # Compute class weights to handle up-day bias
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    scale_pos_weight = weights[1] / weights[0]  # XGBoost uses ratio

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    importances = dict(zip(feature_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    return model, X_test, y_test, importances


def xgb_predict_latest(model, df: pd.DataFrame, feature_cols: list[str]) -> float:
    """Return probability (0-1) for the next trading day being up."""
    features = df[feature_cols].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    row = features.values[-1].reshape(1, -1)
    return float(model.predict_proba(row)[0][1])


def xgb_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    preds = model.predict(X_test)
    return {
        "Accuracy":  accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall":    recall_score(y_test, preds, zero_division=0),
        "F1":        f1_score(y_test, preds, zero_division=0),
    }


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_validate(df: pd.DataFrame,
                           feature_cols: list[str],
                           close_series: pd.Series,
                           lstm_model,
                           lstm_scaler,
                           n_splits: int = 5,
                           initial_train_ratio: float = 0.6,
                           window: int = 60) -> dict:
    """
    Walk-forward validation across n_splits folds.

    Each fold expands the training set by one step:
        Fold 1: Train [0 .. split0],      Test [split0 .. split1]
        Fold 2: Train [0 .. split1],      Test [split1 .. split2]
        ...

    - XGBoost: retrained from scratch on each fold's expanding train set.
    - LSTM:    already-trained model evaluated on each fold's test window
               (retraining LSTM per fold would take too long in a dashboard).
    - ARIMA:   fold metrics derived from the rolling one-step forecast already
               stored in close_series.

    Returns a dict with per-fold DataFrames for charting.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    features = df[feature_cols].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    features.bfill(inplace=True)
    X_all = features.values
    y_all = df["target"].values
    n     = len(X_all)

    init_split = int(n * initial_train_ratio)
    remaining  = n - init_split
    fold_size  = remaining // n_splits

    fold_records = []

    for fold in range(n_splits):
        train_end = init_split + fold * fold_size
        test_start = train_end
        test_end   = test_start + fold_size if fold < n_splits - 1 else n

        if test_end <= test_start or train_end < window + 1:
            continue

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        X_test  = X_all[test_start:test_end]
        y_test  = y_all[test_start:test_end]

        # ── XGBoost fold retrain ──
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        spw     = weights[1] / weights[0]

        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        xgb.fit(X_train, y_train, verbose=False)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

        # ── LSTM fold evaluation (no retraining) ──
        # Scale test window using the already-fit scaler
        X_test_scaled = lstm_scaler.transform(X_test)
        lstm_acc = None
        if len(X_test_scaled) > window:
            X_seq, y_seq = build_lstm_sequences(X_test_scaled, y_test, window)
            if len(X_seq) > 0:
                probs = lstm_model.predict(X_seq, verbose=0).flatten()
                preds = (probs > 0.5).astype(int)
                lstm_acc = accuracy_score(y_seq, preds)

        fold_records.append({
            "Fold":         fold + 1,
            "Train size":   train_end,
            "Test size":    test_end - test_start,
            "XGB Accuracy": round(xgb_acc * 100, 1),
            "LSTM Accuracy": round(lstm_acc * 100, 1) if lstm_acc is not None else None,
        })

    return {"folds": pd.DataFrame(fold_records)}


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
    """
    Bidirectional LSTM + Attention architecture.

    Bidirectional: processes the 60-day window forward AND backward,
    giving the model context from both directions within the window.

    Attention: learns a weight for each of the 60 timesteps so the model
    can focus on the most relevant days (e.g. a sudden volume spike 5 days
    ago) rather than treating all days equally.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Bidirectional, LSTM, Dense, Dropout,
        BatchNormalization, Attention, GlobalAveragePooling1D,
    )

    inputs = Input(shape=input_shape)

    # First Bidirectional LSTM — keeps all timestep outputs for attention
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)

    # Second Bidirectional LSTM — also returns sequences so attention can score them
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Attention: scores each timestep, produces a context vector
    attn_out = Attention()([lstm_out, lstm_out])

    # Pool across timesteps → fixed-size vector
    x = GlobalAveragePooling1D()(attn_out)

    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    Train LSTM classifier with class-weight correction and RobustScaler.
    Returns (model, scaler, history, split_idx, X_test, y_test).
    Scaler is fit on training data only.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    features = df[feature_cols].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    features.bfill(inplace=True)
    features = features.values
    targets  = df["target"].values

    split      = int(len(features) * train_ratio)
    train_feat = features[:split]
    test_feat  = features[split:]

    # RobustScaler handles volume outliers better than MinMaxScaler
    scaler       = RobustScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled  = scaler.transform(test_feat)

    X_train, y_train = build_lstm_sequences(train_scaled, targets[:split], window)
    X_test,  y_test  = build_lstm_sequences(test_scaled,  targets[split:], window)

    # Class weights — corrects for the natural up-day bias (~53% up days)
    classes      = np.unique(y_train)
    weights      = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

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
        class_weight=class_weight,
        verbose=0,
        shuffle=False,
    )

    return model, scaler, history, split, X_test, y_test


def lstm_predict_latest(model, scaler, df: pd.DataFrame,
                         feature_cols: list[str], window: int = 60) -> float:
    """Return probability (0-1) for the next trading day being up."""
    features = df[feature_cols].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    arr    = features.values[-window:]
    scaled = scaler.transform(arr)
    X      = scaled.reshape(1, window, len(feature_cols))
    return float(model.predict(X, verbose=0)[0][0])


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
