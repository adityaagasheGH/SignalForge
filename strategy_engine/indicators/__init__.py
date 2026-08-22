"""
SignalForge Strategy Engine - Indicators Package

Pure, stateless indicator functions. All functions:
- Accept a pandas DataFrame/Series + numeric parameters.
- Return an aligned Series/DataFrame with the same index.
- Document their warm-up period (leading NaN rows) in docstrings.
- Introduce zero side effects or hidden state.
"""
from .moving_averages import sma, ema
from .oscillators import rsi, stochastic, roc, macd
from .channels_volatility import (
    bollinger_bands,
    donchian_channels,
    atr,
    keltner_channels,
    adx,
)

__all__ = [
    "sma",
    "ema",
    "rsi",
    "stochastic",
    "roc",
    "macd",
    "bollinger_bands",
    "donchian_channels",
    "atr",
    "keltner_channels",
    "adx",
]
