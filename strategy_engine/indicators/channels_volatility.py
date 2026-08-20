"""
SignalForge - Channel & Volatility Indicators

Pure functions for:
  - Bollinger Bands (Upper, Middle, Lower)
  - ATR  (Average True Range)
  - Keltner Channels (Upper, Middle, Lower)
  - ADX  (Average Directional Index with +DI, -DI)

All functions:
- Are pure: no side effects, no hidden state, no I/O.
- Return a pd.Series or pd.DataFrame aligned to the input index.
- Document their warm-up period (leading NaN rows) in the docstring.
- Compute using only past bars at each row — zero look-ahead bias.
"""

import pandas as pd
import numpy as np
from .moving_averages import ema


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands — volatility envelope around a simple moving average.

    Middle Band = SMA(close, period)
    Upper Band  = Middle + num_std * StdDev(close, period)
    Lower Band  = Middle - num_std * StdDev(close, period)

    Warm-up: first (period - 1) rows are NaN in all three bands.

    :param close:   pd.Series of closing prices.
    :param period:  Lookback window for SMA and StdDev (default 20).
    :param num_std: Number of standard deviations for band width (default 2.0).
    :return: pd.DataFrame with columns 'upper', 'middle', 'lower', same index.
    :raises ValueError: If period < 2 or num_std <= 0.
    """
    if period < 2:
        raise ValueError(f"Bollinger Bands period must be >= 2, got {period}.")
    if num_std <= 0:
        raise ValueError(f"Bollinger Bands num_std must be > 0, got {num_std}.")

    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=1)

    upper = middle + num_std * std
    lower = middle - num_std * std

    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower},
        index=close.index,
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR) — Wilder's smoothed method.

    True Range (TR) = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR             = Wilder EMA of TR over `period` bars.

    Warm-up: first `period` rows are NaN (period rows needed to seed Wilder's smoothing,
    plus 1 row for the prev-close diff in TR — total: period rows).

    :param high:   pd.Series of High prices.
    :param low:    pd.Series of Low prices.
    :param close:  pd.Series of Close prices.
    :param period: ATR lookback window (default 14).
    :return: pd.Series of ATR values, same index.
    :raises ValueError: If period < 1.
    """
    if period < 1:
        raise ValueError(f"ATR period must be >= 1, got {period}.")

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (alpha = 1/period) for ATR
    atr_series = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr_series.iloc[:period] = float("nan")

    return atr_series


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Keltner Channels — ATR-based volatility envelope around an EMA.

    Middle Band = EMA(close, ema_period)
    Upper Band  = Middle + atr_multiplier * ATR(atr_period)
    Lower Band  = Middle - atr_multiplier * ATR(atr_period)

    Differs from Bollinger Bands (which use StdDev): ATR-based bands are less
    sensitive to short-term volatility spikes and better capture sustained directional moves.

    Warm-up: max(ema_period, atr_period + 1) - 1 rows are NaN.

    :param high:           pd.Series of High prices.
    :param low:            pd.Series of Low prices.
    :param close:          pd.Series of Close prices.
    :param ema_period:     EMA window for middle band (default 20).
    :param atr_period:     ATR window (default 10).
    :param atr_multiplier: ATR multiplier for band width (default 2.0).
    :return: pd.DataFrame with columns 'upper', 'middle', 'lower', same index.
    :raises ValueError: If ema_period < 1, atr_period < 1, or atr_multiplier <= 0.
    """
    if ema_period < 1:
        raise ValueError(f"Keltner ema_period must be >= 1, got {ema_period}.")
    if atr_period < 1:
        raise ValueError(f"Keltner atr_period must be >= 1, got {atr_period}.")
    if atr_multiplier <= 0:
        raise ValueError(f"Keltner atr_multiplier must be > 0, got {atr_multiplier}.")

    middle = ema(close, ema_period)
    atr_vals = atr(high, low, close, atr_period)

    upper = middle + atr_multiplier * atr_vals
    lower = middle - atr_multiplier * atr_vals

    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower},
        index=close.index,
    )


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI — Wilder's method.

    Directional Movement:
      +DM = max(High[t] - High[t-1], 0) if > max(Low[t-1] - Low[t], 0) else 0
      -DM = max(Low[t-1] - Low[t], 0) if > max(High[t] - High[t-1], 0) else 0

    +DI = 100 * Wilder_EMA(+DM, period) / ATR(period)
    -DI = 100 * Wilder_EMA(-DM, period) / ATR(period)
    DX  = 100 * |+DI - -DI| / (+DI + -DI)
    ADX = Wilder_EMA(DX, period)

    Warm-up: first (2 * period) rows are NaN (period for DI smoothing + period for ADX smoothing).

    :param high:   pd.Series of High prices.
    :param low:    pd.Series of Low prices.
    :param close:  pd.Series of Close prices.
    :param period: ADX smoothing period (default 14).
    :return: pd.DataFrame with columns 'adx', 'plus_di', 'minus_di', same index.
    :raises ValueError: If period < 2.
    """
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}.")

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    # Wilder's smoothed ATR, +DM, -DM
    atr_vals = atr(high, low, close, period)

    smoothed_plus_dm = plus_dm_s.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    smoothed_minus_dm = minus_dm_s.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100 * smoothed_plus_dm / atr_vals.replace(0, float("nan"))
    minus_di = 100 * smoothed_minus_dm / atr_vals.replace(0, float("nan"))

    # DX and ADX
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    adx_series = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Enforce warm-up: first (2 * period) rows NaN
    warmup = 2 * period
    adx_series.iloc[:warmup] = float("nan")
    plus_di.iloc[:period] = float("nan")
    minus_di.iloc[:period] = float("nan")

    return pd.DataFrame(
        {"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di},
        index=close.index,
    )
