from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd

from strategy_engine.indicators.channels_volatility import donchian_channels
from strategy_engine.strategies.base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian Channel Breakout strategy.

    BUY when today's Close breaks above the Donchian upper channel computed
    from the PRIOR ``period`` bars, i.e. the highest High of the last
    ``period`` bars *excluding today*. SELL/exit when Close breaks below the
    prior lower channel (default) or the prior middle channel, selectable via
    ``exit_rule``.

    Exit-rule choice: the default is ``"lower"`` — exit when Close breaks
    below the lowest Low of the prior ``period`` bars. This mirrors the
    classic Donchian breakout / turtle system and the blueprint's example
    ("exit on breaking the lower channel"). ``"middle"`` is offered as a
    tighter, quicker exit (the channel midpoint) for callers who prefer to
    give back less of an advance, but is not the default.

    No look-ahead — the critical detail for this strategy:
        ``donchian_channels`` at row T includes row T's own High/Low in its
        rolling window. Comparing today's Close to that same-bar channel
        would be tautological (a new high trivially equals the rolling max
        that just absorbed it), so we ``shift(1)`` the channel by one bar.
        After the shift, the channel value at row T is derived only from bars
        [T-period, T-1] — strictly past data. Row T's Close is therefore
        tested against a level fixed *before* row T existed, which is exactly
        the real-world breakout question and cannot leak future information.
    """

    name: str = "donchian_breakout"

    DEFAULT_GRID: dict[str, list] = {
        "period": [10, 20, 50, 100],
        "exit_rule": ["lower", "middle"],
    }

    def __init__(
        self,
        period: int = 20,
        exit_rule: Literal["lower", "middle"] = "lower",
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}.")
        if exit_rule not in ("lower", "middle"):
            raise ValueError(
                f"exit_rule must be 'lower' or 'middle', got '{exit_rule}'."
            )

        self.period = period
        self.exit_rule = exit_rule

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate Donchian breakout signals.

        BUY (1.0) when Close > prior upper channel; SELL (-1.0) when Close <
        prior lower/middle channel; HOLD (0.0) otherwise. Warm-up rows where
        the prior channel is undefined are NaN.
        """
        for col in ("High", "Low", "Close"):
            if col not in df.columns:
                raise KeyError(f"DataFrame must contain a '{col}' column.")

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        bands = donchian_channels(high, low, period=self.period)

        # shift(1): compare today's Close against the channel formed by the
        # PRIOR `period` bars only. This is what keeps the breakout test
        # honest and look-ahead-free (see class docstring).
        upper_prev = bands["upper"].shift(1)
        lower_prev = bands["lower"].shift(1)
        middle_prev = bands["middle"].shift(1)

        exit_line = lower_prev if self.exit_rule == "lower" else middle_prev

        buy_cond = close > upper_prev
        sell_cond = close < exit_line

        signals = pd.Series(0.0, index=df.index, name="signal")
        # sell first, buy overwrites on any overlap (not possible with these
        # conditions since upper_prev >= exit_line, but kept for consistency
        # with the other strategies' signal-assignment ordering)
        signals[sell_cond] = -1.0
        signals[buy_cond] = 1.0

        warmup_mask = upper_prev.isna()
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
                "period": self.period,
                "exit_rule": self.exit_rule,
            },
            "grid": self.DEFAULT_GRID,
        }
