"""
Signal generation using a weighted probability ensemble + confidence suppression.

Ensemble:
    final_prob = 0.4 * lstm_prob + 0.4 * xgb_prob + 0.2 * arima_direction
    where arima_direction = 1.0 if arima_price > current_price else 0.0

Confidence suppression (applied before everything else):
    If either model is within 0.08 of 0.5 it is uncertain.
    If BOTH models are uncertain  → force HOLD.
    If ONE model is uncertain     → downgrade STRONG signals to regular ones.

RSI and MACD act as secondary tiebreakers only when final_prob is in the
neutral zone (0.45 – 0.55), preventing them from overriding a clear model signal.
"""

_UNCERTAIN = 0.08   # radius around 0.5 that counts as "uncertain"


def _arima_direction(arima_price: float, current_price: float) -> float:
    """Convert ARIMA forecast into a 0/1 probability-like value."""
    return 1.0 if arima_price > current_price else 0.0


def compute_final_prob(lstm_prob: float,
                       xgb_prob: float,
                       arima_price: float,
                       current_price: float) -> float:
    """Weighted ensemble probability (0–1)."""
    arima_dir = _arima_direction(arima_price, current_price)
    return 0.4 * lstm_prob + 0.4 * xgb_prob + 0.2 * arima_dir


def generate_signal(lstm_prob: float,
                    xgb_prob: float,
                    rsi: float,
                    macd_hist: float,
                    arima_price: float,
                    current_price: float) -> tuple[str, str]:

    lstm_uncertain = abs(lstm_prob - 0.5) < _UNCERTAIN
    xgb_uncertain  = abs(xgb_prob  - 0.5) < _UNCERTAIN

    # Both uncertain → no trade
    if lstm_uncertain and xgb_uncertain:
        return "HOLD", "gray"

    final_prob = compute_final_prob(lstm_prob, xgb_prob, arima_price, current_price)

    # RSI / MACD tiebreaker — only nudges when final_prob is in neutral zone
    tiebreak = 0
    if 0.45 <= final_prob <= 0.55:
        if rsi < 30:
            tiebreak += 1
        elif rsi > 70:
            tiebreak -= 1
        if macd_hist > 0:
            tiebreak += 1
        elif macd_hist < 0:
            tiebreak -= 1

    # Base signal from ensemble probability
    if final_prob > 0.65:
        base = "STRONG BUY"
    elif final_prob > 0.55:
        base = "BUY"
    elif final_prob < 0.35:
        base = "STRONG SELL"
    elif final_prob < 0.45:
        base = "SELL"
    else:
        # Neutral zone — let tiebreaker decide
        if tiebreak >= 2:
            base = "BUY"
        elif tiebreak <= -2:
            base = "SELL"
        else:
            base = "HOLD"

    # One model uncertain → downgrade STRONG signals
    if lstm_uncertain or xgb_uncertain:
        if base == "STRONG BUY":
            base = "BUY"
        elif base == "STRONG SELL":
            base = "SELL"

    color_map = {
        "STRONG BUY":  "green",
        "BUY":         "lightgreen",
        "HOLD":        "gray",
        "SELL":        "orange",
        "STRONG SELL": "red",
    }
    return base, color_map[base]


def ensemble_prob(lstm_prob: float,
                  xgb_prob: float,
                  arima_price: float = 0.0,
                  current_price: float = 0.0) -> float:
    """Full weighted ensemble probability for display."""
    return compute_final_prob(lstm_prob, xgb_prob, arima_price, current_price)


def trend_label(prob: float) -> tuple[str, str]:
    if prob > 0.55:
        return "Bullish ↑", "#00c853"
    elif prob < 0.45:
        return "Bearish ↓", "#d50000"
    return "Neutral →", "#9e9e9e"
