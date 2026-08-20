"""
SignalForge — Abstract Base Strategy

Every trading strategy in SignalForge must subclass BaseStrategy and implement
all four interface methods. This enforces the contract described in
ENGINEERING_STANDARDS.md §2 and blueprint.md §1.

Interface methods:
  generate_signals(df)     -> pd.Series[int]       1/0/-1 signal per bar
  calculate_position(df)   -> pd.Series[float]     position size (0.0 to 1.0)
  backtest(df)             -> dict                  summary metrics dict
  get_parameters()         -> dict                  current params + grid definition

Rules enforced here:
  - Strategies must NOT recompute indicators inline; they import from
    strategy_engine.indicators.
  - Signals at bar T may only reference price data up to and including bar T.
    The indicator layer's rolling/shift conventions enforce this structurally.
  - No hardcoded parameter literals anywhere in subclasses — all configurable
    values must flow through __init__ and be exposed via get_parameters().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all SignalForge trading strategies.

    Subclasses must implement all four interface methods. Any strategy added
    to the library must require: one new file in strategies/, one new grid
    entry in config — and zero changes to the backtesting engine, scoring
    engine, or API layer.
    """

    # Subclasses set this to a human-readable name used in logs, UI, and DB.
    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a bar-by-bar signal series from an OHLCV DataFrame.

        :param df: DataFrame with columns Open, High, Low, Close, Volume
                   and a DatetimeIndex in ascending chronological order.
        :return: pd.Series[int] with values in {1, 0, -1} aligned to df.index.
                 1  = BUY  (enter / stay long)
                 0  = HOLD (no position / flat)
                -1  = SELL (exit / go short if supported)
                 NaN rows during warm-up are acceptable; the backtesting engine
                 must skip warm-up rows rather than treat NaN as a signal.
        """

    @abstractmethod
    def calculate_position(self, df: pd.DataFrame) -> pd.Series:
        """
        Translate signals into position sizes (0.0 = flat, 1.0 = fully invested).

        :param df: OHLCV DataFrame.
        :return: pd.Series[float] aligned to df.index.
        """

    @abstractmethod
    def backtest(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run a full backtest of this strategy on df.

        Delegates to the backtesting engine (strategy_engine/backtesting/).
        Must NOT implement its own simulation loop — the engine is the single
        source of truth for execution timing, P&L bookkeeping, and cost application.

        :param df: OHLCV DataFrame.
        :return: dict of performance metrics.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """
        Return the strategy's current parameter set and sweep grid definition.

        Expected structure:
          {
            "values":  {"param_name": current_value, ...},
            "grid":    {"param_name": [candidate1, candidate2, ...], ...},
          }

        The optimization layer iterates over "grid" values during calibration;
        "values" reflects the currently active (possibly post-optimization) config.

        :return: dict with 'values' and 'grid' sub-dicts.
        """
