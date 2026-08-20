"""
SignalForge — EMA/SMA Crossover Trend Following Strategy

Implements SMA Crossover and EMA Crossover as a single configurable class,
covering blueprint.md §1 strategies #4 and #5.

Strategy logic:
  - BUY  (1): fast MA crosses ABOVE slow MA (golden cross)
  - SELL (-1): fast MA crosses BELOW slow MA (death cross)
  - HOLD (0): no crossover on this bar; maintain prior directional position

Configurable parameters (never hardcoded):
  - fast_period:  short MA window (default 20)
  - slow_period:  long MA window  (default 50)
  - ma_type:      'SMA' or 'EMA'   (default 'EMA')

Look-Ahead Bias Guarantee:
  All MAs are computed with right-aligned rolling windows / recursive EWM
  (no center=True, no shift tricks that peek forward). The crossover
  comparison at bar T uses only MA values that themselves only consumed
  prices through bar T. This is enforced structurally by the indicator layer.

Warm-up:
  The first (slow_period - 1) bars produce NaN signals because the slow MA
  has not yet accumulated enough history. The backtesting engine must skip
  these rows when simulating entries/exits.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd

from strategy_engine.indicators.moving_averages import ema, sma
from strategy_engine.strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    Moving Average Crossover trend-following strategy.

    Supports both SMA and EMA via the `ma_type` parameter.

    Strategies covered:
      - #4 SMA Crossover (blueprint §1): ma_type='SMA'
      - #5 EMA Crossover (blueprint §1): ma_type='EMA' (default)

    :param fast_period:  Short MA window. Default 20.
    :param slow_period:  Long MA window. Default 50. Must be > fast_period.
    :param ma_type:      'EMA' (default) or 'SMA'.
    :raises ValueError:  If fast_period >= slow_period, or either < 1.
    """

    name: str = "trend_following"

    # Default parameter grid per blueprint §4 (calibration sweeps only)
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

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate bar-by-bar crossover signals.

        Signal logic:
          - Computes fast_ma and slow_ma for every bar.
          - At bar T, if fast_ma[T] > slow_ma[T] AND fast_ma[T-1] <= slow_ma[T-1]:
              → signal = 1  (golden cross: enter long)
          - At bar T, if fast_ma[T] < slow_ma[T] AND fast_ma[T-1] >= slow_ma[T-1]:
              → signal = -1 (death cross: exit / go flat)
          - Otherwise:
              → signal = 0  (no new crossover on this bar)
          - Warm-up rows (first slow_period - 1 bars): NaN

        All indicator inputs at bar T are computed solely from prices[0..T],
        ensuring zero look-ahead bias.

        :param df: OHLCV DataFrame with at least a 'Close' column.
        :return: pd.Series[float] with values {1.0, 0.0, -1.0, NaN}, same index.
        :raises KeyError: If 'Close' column is absent from df.
        """
        if "Close" not in df.columns:
            raise KeyError("DataFrame must contain a 'Close' column.")

        close = df["Close"]

        # Compute MAs — pure functions from indicator layer, no inline recomputation
        _ma_fn = ema if self.ma_type == "EMA" else sma
        fast_ma = _ma_fn(close, self.fast_period)
        slow_ma = _ma_fn(close, self.slow_period)

        # Shifted by 1 to read the PREVIOUS bar's MA — no look-ahead
        fast_ma_prev = fast_ma.shift(1)
        slow_ma_prev = slow_ma.shift(1)

        # Golden cross: fast crosses above slow
        golden_cross = (fast_ma > slow_ma) & (fast_ma_prev <= slow_ma_prev)
        # Death cross: fast crosses below slow
        death_cross = (fast_ma < slow_ma) & (fast_ma_prev >= slow_ma_prev)

        signals = pd.Series(0.0, index=df.index, name="signal")
        signals[golden_cross] = 1.0
        signals[death_cross] = -1.0

        # Enforce warm-up: rows where slow_ma is NaN have no valid signal
        warmup_mask = slow_ma.isna()
        signals[warmup_mask] = float("nan")

        return signals

    def calculate_position(self, df: pd.DataFrame) -> pd.Series:
        """
        Translate crossover signals into a binary position series:
          - After a BUY signal (1): hold position = 1.0 (fully invested) until next SELL.
          - After a SELL signal (-1): position = 0.0 (flat / cash) until next BUY.
          - Warm-up rows: NaN (not investable).

        Uses forward-fill so the position persists across HOLD (0) bars.

        :param df: OHLCV DataFrame.
        :return: pd.Series[float] in {0.0, 1.0, NaN}, same index as df.
        """
        signals = self.generate_signals(df)

        # Convert 1/-1 → 1.0/0.0; 0 (hold) → NaN so ffill carries the last state
        position = signals.copy()
        position[signals == 0.0] = float("nan")
        position[signals == -1.0] = 0.0
        # Forward-fill: carry last known position across HOLD bars
        position = position.ffill()
        # Any remaining NaN is warm-up — leave as NaN
        return position

    def backtest(
        self, df: pd.DataFrame, config: Optional[BacktestConfig] = None
    ) -> dict[str, Any]:
        """
        Run a full backtest of this strategy on df using Backtester.

        :param df: OHLCV DataFrame.
        :param config: Optional BacktestConfig instance for transaction costs / initial capital.
        :return: Dict containing equity_curve, trade_log, metrics, and config.
        """
        from strategy_engine.backtesting.simulator import Backtester, BacktestConfig

        signals = self.generate_signals(df)
        bt = Backtester(config or BacktestConfig())
        return bt.run(df, signals)

    def get_parameters(self) -> dict[str, Any]:
        """
        Return current parameter values and their calibration sweep grid.

        :return: dict with keys 'values' (current config) and 'grid' (sweep space).
        """
        return {
            "values": {
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "ma_type": self.ma_type,
            },
            "grid": self.DEFAULT_GRID,
        }
