import pandas as pd
import numpy as np
from .moving_averages import ema


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands (upper, middle, lower)."""
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
    """Average True Range (ATR)."""
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
    """Keltner Channels (upper, middle, lower)."""
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
    """Average Directional Index (ADX, plus_di, minus_di)."""
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}.")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    atr_vals = atr(high, low, close, period)

    smoothed_plus_dm = plus_dm_s.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    smoothed_minus_dm = minus_dm_s.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100 * smoothed_plus_dm / atr_vals.replace(0, float("nan"))
    minus_di = 100 * smoothed_minus_dm / atr_vals.replace(0, float("nan"))

    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    adx_series = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    warmup = 2 * period
    adx_series.iloc[:warmup] = float("nan")
    plus_di.iloc[:period] = float("nan")
    minus_di.iloc[:period] = float("nan")

    return pd.DataFrame(
        {"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di},
        index=close.index,
    )
