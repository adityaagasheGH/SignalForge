from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate bar-by-bar signals (1=BUY, 0=HOLD, -1=SELL)."""

    @abstractmethod
    def calculate_position(self, df: pd.DataFrame) -> pd.Series:
        """Translate signals into position sizes (0.0 to 1.0)."""

    @abstractmethod
    def backtest(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run backtest on OHLCV DataFrame."""

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters and sweep grid."""
