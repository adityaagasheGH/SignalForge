import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average (SMA)."""
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, got {period}.")
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}.")

    result = close.ewm(span=period, adjust=adjust, min_periods=period).mean()
    result.iloc[: period - 1] = float("nan")
    return result
