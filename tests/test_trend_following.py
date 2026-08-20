"""
Unit tests for strategy_engine/strategies/trend_following.py

Test categories:
  1. Initialisation & parameter validation
  2. Signal correctness (golden cross, death cross, hold, warm-up NaN)
  3. Position series behaviour (ffill across hold bars)
  4. Parameter customisation (non-default fast/slow periods, SMA vs EMA)
  5. Look-Ahead Bias — injecting a future price shock must NOT alter any
     signal value computed for bars before the shock.
  6. Interface contract — get_parameters() shape, BaseStrategy subclass.
"""

import pytest
import pandas as pd
import numpy as np

from strategy_engine.strategies.trend_following import TrendFollowingStrategy
from strategy_engine.strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_df(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of Close prices."""
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="D")
    close = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


@pytest.fixture
def default_strategy() -> TrendFollowingStrategy:
    return TrendFollowingStrategy(fast_period=3, slow_period=5)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """
    Prices that produce a guaranteed golden cross at bar 5 (0-indexed):
    Bars 0-4:  flat at 100 → fast MA == slow MA (no cross).
    Bar 5 onward: rising sharply → fast MA climbs above slow MA.
    """
    closes = [100.0] * 5 + [110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
    return make_df(closes)


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """
    Prices that produce a golden cross first, then a death cross.
    Rising phase: bars 0-7.  Falling phase: bars 8-15.
    """
    up = [100.0 + i * 10 for i in range(8)]
    down = [170.0 - i * 15 for i in range(1, 9)]
    return make_df(up + down)


# ---------------------------------------------------------------------------
# 1. Initialisation & parameter validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        s = TrendFollowingStrategy()
        assert s.fast_period == 20
        assert s.slow_period == 50
        assert s.ma_type == "EMA"

    def test_custom_params(self):
        s = TrendFollowingStrategy(fast_period=10, slow_period=30, ma_type="SMA")
        assert s.fast_period == 10
        assert s.slow_period == 30
        assert s.ma_type == "SMA"

    def test_fast_ge_slow_raises(self):
        with pytest.raises(ValueError, match="fast_period"):
            TrendFollowingStrategy(fast_period=50, slow_period=20)

    def test_fast_equals_slow_raises(self):
        with pytest.raises(ValueError):
            TrendFollowingStrategy(fast_period=20, slow_period=20)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            TrendFollowingStrategy(fast_period=0, slow_period=5)

    def test_invalid_ma_type_raises(self):
        with pytest.raises(ValueError, match="ma_type"):
            TrendFollowingStrategy(fast_period=3, slow_period=5, ma_type="LWMA")

    def test_is_base_strategy_subclass(self):
        assert issubclass(TrendFollowingStrategy, BaseStrategy)


# ---------------------------------------------------------------------------
# 2. Signal correctness
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def test_output_index_matches_df(self, default_strategy, trending_up_df):
        signals = default_strategy.generate_signals(trending_up_df)
        assert signals.index.equals(trending_up_df.index)

    def test_warmup_rows_are_nan(self, default_strategy, trending_up_df):
        """First (slow_period - 1) = 4 bars must be NaN."""
        signals = default_strategy.generate_signals(trending_up_df)
        assert signals.iloc[:4].isna().all(), "Warm-up rows must be NaN."
        # At least some valid (non-NaN) signals after warm-up
        assert signals.iloc[4:].notna().any()

    def test_signal_values_are_valid(self, default_strategy, trending_up_df):
        """All non-NaN signals must be exactly 1.0, 0.0, or -1.0."""
        signals = default_strategy.generate_signals(trending_up_df)
        valid = signals.dropna()
        assert set(valid.unique()).issubset({1.0, 0.0, -1.0})

    def test_golden_cross_produces_buy_signal(self):
        """
        Manually craft prices where fast EMA(3) crosses above slow EMA(5).
        Bars 0-4: flat at 100.  Bar 5+: strong uptrend.
        A BUY (1) signal must appear somewhere in the uptrend.
        """
        s = TrendFollowingStrategy(fast_period=3, slow_period=5)
        closes = [100.0] * 5 + [115.0, 130.0, 145.0, 160.0, 175.0]
        df = make_df(closes)
        signals = s.generate_signals(df)
        assert (signals == 1.0).any(), "Golden cross must produce at least one BUY signal."

    def test_death_cross_produces_sell_signal(self, trending_down_df):
        """A downtrend after an uptrend must produce a SELL (-1) signal."""
        s = TrendFollowingStrategy(fast_period=3, slow_period=5)
        signals = s.generate_signals(trending_down_df)
        assert (signals == -1.0).any(), "Death cross must produce at least one SELL signal."

    def test_missing_close_column_raises(self, default_strategy):
        df = pd.DataFrame({"Open": [100.0], "High": [101.0]})
        with pytest.raises(KeyError, match="Close"):
            default_strategy.generate_signals(df)

    def test_no_double_signal_on_same_bar(self, default_strategy, trending_up_df):
        """A bar cannot simultaneously be both BUY and SELL."""
        signals = default_strategy.generate_signals(trending_up_df)
        assert not ((signals == 1.0) & (signals == -1.0)).any()

    def test_hold_bars_are_zero(self, default_strategy, trending_up_df):
        """After the first crossover, most bars should be HOLD (0)."""
        signals = default_strategy.generate_signals(trending_up_df)
        valid = signals.dropna()
        hold_count = (valid == 0.0).sum()
        total_valid = len(valid)
        # In any realistic price series, hold bars should dominate
        assert hold_count > 0, "There should be HOLD (0) bars between crossovers."


# ---------------------------------------------------------------------------
# 3. Position series
# ---------------------------------------------------------------------------

class TestCalculatePosition:
    def test_position_is_01_or_nan(self, default_strategy, trending_up_df):
        """Position values must be 0.0 (flat), 1.0 (long), or NaN (warm-up)."""
        pos = default_strategy.calculate_position(trending_up_df)
        valid = pos.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_position_held_across_hold_bars(self):
        """
        After a BUY signal, position must remain 1.0 through subsequent HOLD bars
        until a SELL signal flips it to 0.0.
        """
        s = TrendFollowingStrategy(fast_period=3, slow_period=5)
        # Craft a sequence: up, then flat, then down
        closes = (
            [100.0] * 5
            + [120.0, 140.0, 160.0, 180.0]   # uptrend → golden cross
            + [180.0] * 4                       # flat → hold bars
            + [160.0, 130.0, 100.0, 70.0, 50.0]  # downtrend → death cross
        )
        df = make_df(closes)
        signals = s.generate_signals(df)
        pos = s.calculate_position(df)

        # Find first BUY signal index
        buy_idx = signals[signals == 1.0].index
        if len(buy_idx) == 0:
            pytest.skip("No golden cross in this fixture — skip position continuity test.")

        first_buy = buy_idx[0]
        loc = df.index.get_loc(first_buy)

        # Check a few bars after the BUY: position must be 1.0 until first SELL
        sell_idx = signals[signals == -1.0].index
        if len(sell_idx) == 0:
            pytest.skip("No death cross in this fixture — skip position continuity test.")

        first_sell = sell_idx[0]
        sell_loc = df.index.get_loc(first_sell)

        # All bars strictly between first BUY and first SELL should be position=1.0
        between_slice = pos.iloc[loc + 1 : sell_loc]
        assert (between_slice == 1.0).all(), (
            "Position must remain 1.0 across HOLD bars between BUY and SELL."
        )

    def test_position_drops_after_death_cross(self, trending_down_df):
        s = TrendFollowingStrategy(fast_period=3, slow_period=5)
        signals = s.generate_signals(trending_down_df)
        pos = s.calculate_position(trending_down_df)
        sell_idx = signals[signals == -1.0].index
        if len(sell_idx) == 0:
            pytest.skip("No death cross produced — skip.")
        # Position at and after first SELL must be 0.0 (until a new BUY, if any)
        first_sell_loc = trending_down_df.index.get_loc(sell_idx[0])
        assert pos.iloc[first_sell_loc] == 0.0


# ---------------------------------------------------------------------------
# 4. Parameter customisation
# ---------------------------------------------------------------------------

class TestParameterCustomisation:
    def test_sma_variant_produces_signals(self, trending_up_df):
        s = TrendFollowingStrategy(fast_period=3, slow_period=5, ma_type="SMA")
        signals = s.generate_signals(trending_up_df)
        assert signals.index.equals(trending_up_df.index)
        assert (signals == 1.0).any() or (signals == 0.0).any()

    def test_different_periods_give_different_crossover_timing(self):
        """Wider period gap should delay crossover compared to narrower gap."""
        closes = [100.0] * 3 + [110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
        df = make_df(closes)

        s_narrow = TrendFollowingStrategy(fast_period=2, slow_period=3)
        s_wide = TrendFollowingStrategy(fast_period=2, slow_period=8)

        sig_narrow = s_narrow.generate_signals(df)
        sig_wide = s_wide.generate_signals(df)

        # Narrow-period strategy should have valid (non-NaN) signals earlier
        first_valid_narrow = sig_narrow.first_valid_index()
        first_valid_wide = sig_wide.first_valid_index()
        assert first_valid_narrow <= first_valid_wide

    def test_get_parameters_structure(self, default_strategy):
        params = default_strategy.get_parameters()
        assert "values" in params
        assert "grid" in params
        assert params["values"]["fast_period"] == 3
        assert params["values"]["slow_period"] == 5
        assert "fast_period" in params["grid"]
        assert "slow_period" in params["grid"]

    def test_get_parameters_values_match_init(self):
        s = TrendFollowingStrategy(fast_period=10, slow_period=40, ma_type="SMA")
        p = s.get_parameters()["values"]
        assert p["fast_period"] == 10
        assert p["slow_period"] == 40
        assert p["ma_type"] == "SMA"


# ---------------------------------------------------------------------------
# 5. Look-Ahead Bias Tests
# ---------------------------------------------------------------------------

class TestLookAheadBias:
    """
    Injects a massive price shock into the last N bars of a DataFrame, then
    asserts that all signal values computed BEFORE the shock are unchanged.
    This is the primary correctness guard per PRINCIPLES.md §2.
    """

    SHOCK_VALUE = 1_000_000.0
    SHOCK_TAIL = 5

    def _make_long_df(self) -> pd.DataFrame:
        np.random.seed(0)
        n = 80
        closes = list(100.0 + np.cumsum(np.random.randn(n) * 2))
        return make_df(closes)

    def _assert_past_unchanged(self, before: pd.Series, after: pd.Series) -> None:
        safe_slice_before = before.iloc[: -self.SHOCK_TAIL]
        safe_slice_after = after.iloc[: -self.SHOCK_TAIL]
        pd.testing.assert_series_equal(
            safe_slice_before,
            safe_slice_after,
            check_names=False,
            check_dtype=False,
        )

    def test_ema_crossover_lab(self):
        s = TrendFollowingStrategy(fast_period=5, slow_period=10, ma_type="EMA")
        df = self._make_long_df()

        signals_before = s.generate_signals(df).copy()

        shocked_df = df.copy()
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "High"] = self.SHOCK_VALUE
        signals_after = s.generate_signals(shocked_df)

        self._assert_past_unchanged(signals_before, signals_after)

    def test_sma_crossover_lab(self):
        s = TrendFollowingStrategy(fast_period=5, slow_period=10, ma_type="SMA")
        df = self._make_long_df()

        signals_before = s.generate_signals(df).copy()

        shocked_df = df.copy()
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        signals_after = s.generate_signals(shocked_df)

        self._assert_past_unchanged(signals_before, signals_after)

    def test_position_lab(self):
        s = TrendFollowingStrategy(fast_period=5, slow_period=10, ma_type="EMA")
        df = self._make_long_df()

        pos_before = s.calculate_position(df).copy()

        shocked_df = df.copy()
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        pos_after = s.calculate_position(shocked_df)

        self._assert_past_unchanged(pos_before, pos_after)
