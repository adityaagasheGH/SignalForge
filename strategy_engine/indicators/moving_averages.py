"""
SignalForge - Moving Average Indicators

Pure functions for Simple Moving Average (SMA) and Exponential Moving Average (EMA).

All functions:
- Are pure: no side effects, no hidden state, no I/O.
- Return a pd.Series aligned to the input index.
- Warm-up NaN rows: the first (period - 1) rows will be NaN.
"""

import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Computes the arithmetic mean of the closing price over a rolling window.
    Uses a right-aligned window so only past data informs each bar — no look-ahead bias.

    Warm-up: first (period - 1) rows are NaN.

    :param close: pd.Series of closing prices, DatetimeIndex.
    :param period: Rolling lookback window size (integer >= 1).
    :return: pd.Series of SMA values, same index as input.
    :raises ValueError: If period < 1.
    """
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, got {period}.")
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    Computes the EMA using the standard smoothing factor alpha = 2 / (period + 1).
    With adjust=False (default), computes recursive EMA as used in TA (e.g. MACD internals).

    Warm-up: the first (period - 1) rows will be NaN.
    NaN enforcement: leading NaNs from SMA are propagated to enforce the warm-up window
    consistently with SMA-based indicators — the recursive EMA naturally has a warm-up
    period equal to `period`, which is enforced explicitly here.

    :param close: pd.Series of closing prices, DatetimeIndex.
    :param period: EMA span (smoothing window) >= 1.
    :param adjust: If True, uses pandas weighted adjustment. Default False (recursive).
    :return: pd.Series of EMA values, same index as input.
    :raises ValueError: If period < 1.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}.")

    result = close.ewm(span=period, adjust=adjust, min_periods=period).mean()
    # Enforce explicit warm-up: first (period - 1) values set to NaN
    result.iloc[: period - 1] = float("nan")
    return result
