import pandas as pd
import numpy as np
from .moving_averages import ema


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI). Warm-up: first (2 * period - 1) rows are NaN."""
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}.")

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series = rsi_series.where(avg_loss != 0, other=100.0)

    warmup = 2 * period - 1
    rsi_series.iloc[:warmup] = float("nan")
    return rsi_series


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""
    if k_period < 1:
        raise ValueError(f"Stochastic k_period must be >= 1, got {k_period}.")
    if d_period < 1:
        raise ValueError(f"Stochastic d_period must be >= 1, got {d_period}.")

    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    range_ = (highest_high - lowest_low).replace(0, float("nan"))
    pct_k = 100 * (close - lowest_low) / range_
    pct_d = pct_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"%K": pct_k, "%D": pct_d}, index=close.index)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (ROC)."""
    if period < 1:
        raise ValueError(f"ROC period must be >= 1, got {period}.")

    return (close / close.shift(period) - 1) * 100


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)."""
    if fast_period >= slow_period:
        raise ValueError(
            f"MACD fast_period ({fast_period}) must be < slow_period ({slow_period})."
        )
    if signal_period < 1:
        raise ValueError(f"MACD signal_period must be >= 1, got {signal_period}.")

    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()
    signal_line.iloc[: slow_period + signal_period - 2] = float("nan")

    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )
