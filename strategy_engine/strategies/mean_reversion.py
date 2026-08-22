from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from strategy_engine.indicators.channels_volatility import bollinger_bands
from strategy_engine.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band Mean Reversion strategy.

    BUY when price closes at or below a configurable entry line derived from
    the lower band, exit (SELL) when price closes at or above a configurable
    exit line between the lower and middle band. Both lines are expressed as
    thresholds against the band geometry rather than hardcoded to the raw
    lower/middle band values, so entries/exits can be tuned to be more or
    less conservative without touching the indicator layer.
    """

    name: str = "bollinger_mean_reversion"

    DEFAULT_GRID: dict[str, list] = {
        "lookback": [10, 20, 30, 50],
        "num_std": [1.5, 2.0, 2.5, 3.0],
        "entry_threshold": [0.75, 1.0, 1.25],
        "exit_threshold": [0.5, 1.0, 1.25],
    }

    def __init__(
        self,
        lookback: int = 20,
        num_std: float = 2.0,
        entry_threshold: float = 1.0,
        exit_threshold: float = 1.0,
    ) -> None:
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}.")
        if num_std <= 0:
            raise ValueError(f"num_std must be > 0, got {num_std}.")
        if entry_threshold <= 0:
            raise ValueError(f"entry_threshold must be > 0, got {entry_threshold}.")
        if exit_threshold <= 0:
            raise ValueError(f"exit_threshold must be > 0, got {exit_threshold}.")

        self.lookback = lookback
        self.num_std = num_std
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals from Bollinger Band geometry.

        entry_threshold=1.0 reproduces the base rule (BUY at/below the lower
        band); >1.0 requires a deeper move below the lower band, <1.0
        triggers before price fully reaches it. exit_threshold=1.0
        reproduces exiting at the middle band; <1.0 exits on a partial
        reversion, closer to the lower band.
        """
        if "Close" not in df.columns:
            raise KeyError("DataFrame must contain a 'Close' column.")

        close = df["Close"]

        bands = bollinger_bands(close, period=self.lookback, num_std=self.num_std)
        band_width = bands["middle"] - bands["lower"]

        entry_line = bands["middle"] - self.entry_threshold * band_width
        exit_line = bands["lower"] + self.exit_threshold * band_width

        buy_cond = close <= entry_line
        sell_cond = close >= exit_line

        signals = pd.Series(0.0, index=df.index, name="signal")
        # sell first, buy overwrites on overlap (only possible with
        # non-default thresholds where entry_threshold + exit_threshold <= 1)
        signals[sell_cond] = -1.0
        signals[buy_cond] = 1.0

        warmup_mask = bands["middle"].isna()
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
                "lookback": self.lookback,
                "num_std": self.num_std,
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
            },
            "grid": self.DEFAULT_GRID,
        }
