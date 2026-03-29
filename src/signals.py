"""Signal generation combining ARIMA price + LSTM probability + technicals."""


def generate_signal(lstm_prob: float, rsi: float, macd_hist: float,
                    arima_price: float, current_price: float) -> tuple[str, str]:
    score = 0

    if lstm_prob > 0.55:
        score += 2
    elif lstm_prob < 0.45:
        score -= 2

    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1

    if macd_hist > 0:
        score += 1
    elif macd_hist < 0:
        score -= 1

    if arima_price > current_price:
        score += 1
    elif arima_price < current_price:
        score -= 1

    if score >= 3:
        return "STRONG BUY", "green"
    elif score == 2:
        return "BUY", "lightgreen"
    elif score == -2:
        return "SELL", "orange"
    elif score <= -3:
        return "STRONG SELL", "red"
    else:
        return "HOLD", "gray"


def trend_label(lstm_prob: float) -> tuple[str, str]:
    if lstm_prob > 0.55:
        return "Bullish ↑", "#00c853"
    elif lstm_prob < 0.45:
        return "Bearish ↓", "#d50000"
    return "Neutral →", "#9e9e9e"
