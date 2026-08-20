"""
SignalForge - Oscillator Indicators

Pure functions for:
  - RSI  (Relative Strength Index)
  - Stochastic Oscillator (%K, %D)
  - ROC  (Rate of Change)
  - MACD (Moving Average Convergence Divergence)

All functions:
- Are pure: no side effects, no hidden state, no I/O.
- Return a pd.Series or pd.DataFrame aligned to the input index.
- Document their warm-up period (leading NaN rows) in the docstring.
- Compute using only past bars at each row (right-aligned windows / lagged shifts).
"""

import pandas as pd
import numpy as np
from .moving_averages import ema


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) — Wilder's smoothed method.

    RSI = 100 - (100 / (1 + RS))
    RS  = Average Gain / Average Loss over `period` bars.
    Uses Wilder's exponential smoothing (alpha = 1/period) for gain/loss averages.

    Warm-up: first (2 * period - 1) rows are NaN.
      - period rows needed for initial average gain/loss seed.
      - Wilder's smoothing requires one additional period of prior history to stabilise.
      - The first `period` rows of delta are NaN (no prior close for diff).

    :param close: pd.Series of closing prices.
    :param period: RSI lookback window (default 14).
    :return: pd.Series of RSI values [0, 100], same index.
    :raises ValueError: If period < 2.
    """
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}.")

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing: alpha = 1/period, equivalent to ewm(alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    # When avg_loss is 0, RS is undefined but RSI is definitionally 100 (no losses at all)
    rsi_series = rsi_series.where(avg_loss != 0, other=100.0)

    # Enforce warm-up: first (period) rows of delta are NaN; +1 for diff offset
    rsi_series.iloc[: period] = float("nan")

    return rsi_series


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    %K = 100 * (Close - Lowest Low[k_period]) / (Highest High[k_period] - Lowest Low[k_period])
    %D = SMA(%K, d_period)

    Warm-up:
      - %K: first (k_period - 1) rows are NaN.
      - %D: first (k_period + d_period - 2) rows are NaN.

    :param high:     pd.Series of High prices.
    :param low:      pd.Series of Low prices.
    :param close:    pd.Series of Close prices.
    :param k_period: Lookback window for %K (default 14).
    :param d_period: Smoothing period for %D (default 3).
    :return: pd.DataFrame with columns '%K' and '%D', same index as inputs.
    :raises ValueError: If k_period < 1 or d_period < 1.
    """
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
    """
    Rate of Change (ROC) — Momentum indicator.

    ROC = (Close[t] / Close[t - period] - 1) * 100

    Warm-up: first `period` rows are NaN (insufficient prior bars).

    :param close:  pd.Series of closing prices.
    :param period: Lookback window for ROC (default 10).
    :return: pd.Series of ROC values (%), same index.
    :raises ValueError: If period < 1.
    """
    if period < 1:
        raise ValueError(f"ROC period must be >= 1, got {period}.")

    return (close / close.shift(period) - 1) * 100


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    MACD Line   = EMA(fast_period) - EMA(slow_period)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram   = MACD Line - Signal Line

    Warm-up:
      - MACD Line:   first (slow_period - 1) rows are NaN.
      - Signal Line: first (slow_period + signal_period - 2) rows are NaN.
      - Histogram:   same as Signal Line.

    :param close:         pd.Series of closing prices.
    :param fast_period:   EMA period for fast line (default 12).
    :param slow_period:   EMA period for slow line (default 26).
    :param signal_period: EMA period for signal line (default 9).
    :return: pd.DataFrame with columns 'macd', 'signal', 'histogram', same index.
    :raises ValueError: If fast_period >= slow_period, or any period < 1.
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"MACD fast_period ({fast_period}) must be < slow_period ({slow_period})."
        )
    if signal_period < 1:
        raise ValueError(f"MACD signal_period must be >= 1, got {signal_period}.")

    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)

    macd_line = fast_ema - slow_ema

    # EMA of macd_line for signal; use ewm directly to allow NaN-propagation from warm-up
    signal_line = macd_line.ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()
    # Enforce MACD warm-up: signal should also be NaN for the full slow warm-up rows
    signal_line.iloc[: slow_period + signal_period - 2] = float("nan")

    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )
