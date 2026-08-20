from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd

from strategy_engine.indicators.moving_averages import ema, sma
from strategy_engine.strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Moving Average Crossover trend-following strategy."""

    name: str = "trend_following"

    DEFAULT_GRID: dict[str, list] = {
        "fast_period": [10, 20, 50],
        "slow_period": [20, 50, 100, 200],
        "ma_type": ["EMA", "SMA"],
    }

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        ma_type: Literal["EMA", "SMA"] = "EMA",
    ) -> None:
        if fast_period < 1:
            raise ValueError(f"fast_period must be >= 1, got {fast_period}.")
        if slow_period < 1:
            raise ValueError(f"slow_period must be >= 1, got {slow_period}.")
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})."
            )
        if ma_type not in ("EMA", "SMA"):
            raise ValueError(f"ma_type must be 'EMA' or 'SMA', got '{ma_type}'.")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate crossover signals."""
        if "Close" not in df.columns:
            raise KeyError("DataFrame must contain a 'Close' column.")

        close = df["Close"]

        _ma_fn = ema if self.ma_type == "EMA" else sma
        fast_ma = _ma_fn(close, self.fast_period)
        slow_ma = _ma_fn(close, self.slow_period)

        fast_ma_prev = fast_ma.shift(1)
        slow_ma_prev = slow_ma.shift(1)

        golden_cross = (fast_ma > slow_ma) & (fast_ma_prev <= slow_ma_prev)
        death_cross = (fast_ma < slow_ma) & (fast_ma_prev >= slow_ma_prev)

        signals = pd.Series(0.0, index=df.index, name="signal")
        signals[golden_cross] = 1.0
        signals[death_cross] = -1.0

        warmup_mask = slow_ma.isna()
        signals[warmup_mask] = float("nan")

        return signals

    def calculate_position(self, df: pd.DataFrame) -> pd.Series:
        """Translate signals into binary position series (1.0 or 0.0)."""
        signals = self.generate_signals(df)

        position = signals.copy()
        position[signals == 0.0] = float("nan")
        position[signals == -1.0] = 0.0
        position = position.ffill()
        return position

    def backtest(
        self, df: pd.DataFrame, config: Optional[Any] = None
    ) -> dict[str, Any]:
        """Run backtest on OHLCV DataFrame."""
        from strategy_engine.backtesting.simulator import Backtester, BacktestConfig

        signals = self.generate_signals(df)
        bt = Backtester(config or BacktestConfig())
        return bt.run(df, signals)

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters and sweep grid."""
        return {
            "values": {
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "ma_type": self.ma_type,
            },
            "grid": self.DEFAULT_GRID,
        }
