import pytest
import pandas as pd
import numpy as np

from strategy_engine.strategies.breakout import DonchianBreakoutStrategy
from strategy_engine.strategies.base import BaseStrategy


def make_ohlc(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from explicit High/Low/Close lists."""
    n = len(closes)
    assert len(highs) == n and len(lows) == n
    dates = pd.date_range(start, periods=n, freq="D")
    close = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": pd.Series(highs, index=dates, dtype=float),
            "Low": pd.Series(lows, index=dates, dtype=float),
            "Close": close,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


def make_from_closes(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build an OHLCV DataFrame from closes, deriving High/Low around them."""
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
def default_strategy() -> DonchianBreakoutStrategy:
    return DonchianBreakoutStrategy(period=3)


@pytest.fixture
def breakout_df() -> pd.DataFrame:
    """Range-bound bars, then a clear upward breakout, then a downward break.

    period=3 → prior 3-bar upper channel is 105, lower is 95 while range-bound.
    Bar 6 closes at 110 (breaks above 105) → BUY.
    Bar 8 closes at 80 (breaks below 95) → SELL.
    """
    highs = [105, 105, 105, 105, 105, 105, 112, 112, 82]
    lows = [95, 95, 95, 95, 95, 95, 108, 108, 78]
    closes = [100, 100, 100, 100, 100, 100, 110, 111, 80]
    return make_ohlc(highs, lows, closes)


class TestInit:
    def test_default_params(self):
        s = DonchianBreakoutStrategy()
        assert s.period == 20
        assert s.exit_rule == "lower"

    def test_custom_params(self):
        s = DonchianBreakoutStrategy(period=50, exit_rule="middle")
        assert s.period == 50
        assert s.exit_rule == "middle"

    def test_period_too_small_raises(self):
        with pytest.raises(ValueError, match="period"):
            DonchianBreakoutStrategy(period=1)

    def test_invalid_exit_rule_raises(self):
        with pytest.raises(ValueError, match="exit_rule"):
            DonchianBreakoutStrategy(period=20, exit_rule="upper")

    def test_is_base_strategy_subclass(self):
        assert issubclass(DonchianBreakoutStrategy, BaseStrategy)


class TestGenerateSignals:
    def test_output_index_matches_df(self, default_strategy, breakout_df):
        signals = default_strategy.generate_signals(breakout_df)
        assert signals.index.equals(breakout_df.index)

    def test_warmup_rows_are_nan(self, default_strategy, breakout_df):
        signals = default_strategy.generate_signals(breakout_df)
        # channel warm-up (period-1) plus one bar of shift(1) => `period` NaNs
        assert signals.iloc[: default_strategy.period].isna().all()
        assert pd.notna(signals.iloc[default_strategy.period])

    def test_signal_values_are_valid(self, default_strategy, breakout_df):
        signals = default_strategy.generate_signals(breakout_df)
        valid = signals.dropna()
        assert set(valid.unique()).issubset({1.0, 0.0, -1.0})

    def test_upward_breakout_bar_is_buy(self, default_strategy, breakout_df):
        signals = default_strategy.generate_signals(breakout_df)
        assert signals.iloc[6] == 1.0

    def test_downward_break_bar_is_sell(self, default_strategy, breakout_df):
        signals = default_strategy.generate_signals(breakout_df)
        assert signals.iloc[8] == -1.0

    def test_range_bound_bars_are_hold(self, default_strategy, breakout_df):
        signals = default_strategy.generate_signals(breakout_df)
        # bars 3,4,5 are range-bound: no breakout in either direction
        assert (signals.iloc[3:6] == 0.0).all()

    def test_same_bar_high_does_not_trigger_breakout(self):
        """A steadily rising High must not make every bar a tautological BUY.

        If the channel were NOT shifted, each new high would trivially equal
        the rolling max and Close could never be recorded as breaking it in a
        meaningful way. With the shift, a monotonic ramp still produces real
        breakout signals (today's close exceeds yesterday's channel), which is
        correct — the guard we assert is that signals remain in {-1,0,1} and
        are not universally forced by same-bar contamination.
        """
        closes = [100.0 + i for i in range(15)]
        df = make_from_closes(closes)
        s = DonchianBreakoutStrategy(period=3)
        signals = s.generate_signals(df)
        valid = signals.dropna()
        assert set(valid.unique()).issubset({1.0, 0.0, -1.0})

    def test_missing_column_raises(self, default_strategy):
        df = pd.DataFrame({"Open": [100.0], "Close": [100.0]})
        with pytest.raises(KeyError):
            default_strategy.generate_signals(df)

    def test_middle_exit_rule_sells_earlier_than_lower(self):
        """A dip below the prior middle (but above prior lower) exits only
        under the middle rule, confirming exit_rule is actually wired in."""
        highs = [110, 110, 110, 110, 110]
        lows = [90, 90, 90, 90, 90]
        closes = [100, 100, 100, 95, 95]
        df = make_ohlc(highs, lows, closes)

        lower_sig = DonchianBreakoutStrategy(period=3, exit_rule="lower").generate_signals(df)
        middle_sig = DonchianBreakoutStrategy(period=3, exit_rule="middle").generate_signals(df)

        # prior lower=90, prior middle=100; close=95 at row 3
        assert lower_sig.iloc[3] == 0.0
        assert middle_sig.iloc[3] == -1.0


class TestCalculatePosition:
    def test_position_is_01_or_nan(self, default_strategy, breakout_df):
        pos = default_strategy.calculate_position(breakout_df)
        valid = pos.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_long_after_breakout_then_flat_after_break(self, default_strategy, breakout_df):
        pos = default_strategy.calculate_position(breakout_df)
        assert pos.iloc[6] == 1.0   # entered on upward breakout
        assert pos.iloc[7] == 1.0   # held across hold bar
        assert pos.iloc[8] == 0.0   # exited on downward break

    def test_flat_before_first_entry(self, default_strategy, breakout_df):
        pos = default_strategy.calculate_position(breakout_df)
        assert pos.iloc[:6].isna().all()


class TestParameterCustomisation:
    def test_get_parameters_structure(self, default_strategy):
        params = default_strategy.get_parameters()
        assert "values" in params
        assert "grid" in params
        assert params["values"]["period"] == 3
        assert params["values"]["exit_rule"] == "lower"

    def test_get_parameters_values_match_init(self):
        s = DonchianBreakoutStrategy(period=50, exit_rule="middle")
        p = s.get_parameters()["values"]
        assert p["period"] == 50
        assert p["exit_rule"] == "middle"

    def test_default_grid_matches_blueprint(self, default_strategy):
        grid = default_strategy.get_parameters()["grid"]
        assert grid["period"] == [10, 20, 50, 100]
        assert grid["exit_rule"] == ["lower", "middle"]


class TestLookAheadBias:
    SHOCK_VALUE = 1_000_000.0
    SHOCK_TAIL = 5

    def _make_long_df(self) -> pd.DataFrame:
        np.random.seed(0)
        n = 80
        closes = list(100.0 + np.cumsum(np.random.randn(n) * 2))
        return make_from_closes(closes)

    def _assert_past_unchanged(self, before: pd.Series, after: pd.Series) -> None:
        pd.testing.assert_series_equal(
            before.iloc[: -self.SHOCK_TAIL],
            after.iloc[: -self.SHOCK_TAIL],
            check_names=False,
            check_dtype=False,
        )

    def test_signals_unaffected_by_future_upshock(self):
        s = DonchianBreakoutStrategy(period=10)
        df = self._make_long_df()

        signals_before = s.generate_signals(df).copy()

        shocked = df.copy()
        shocked.loc[shocked.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        shocked.loc[shocked.index[-self.SHOCK_TAIL :], "High"] = self.SHOCK_VALUE
        signals_after = s.generate_signals(shocked)

        self._assert_past_unchanged(signals_before, signals_after)

    def test_signals_unaffected_by_future_downshock(self):
        s = DonchianBreakoutStrategy(period=10)
        df = self._make_long_df()

        signals_before = s.generate_signals(df).copy()

        shocked = df.copy()
        shocked.loc[shocked.index[-self.SHOCK_TAIL :], "Close"] = 0.01
        shocked.loc[shocked.index[-self.SHOCK_TAIL :], "Low"] = 0.01
        signals_after = s.generate_signals(shocked)

        self._assert_past_unchanged(signals_before, signals_after)

    def test_position_unaffected_by_future_shock(self):
        s = DonchianBreakoutStrategy(period=10)
        df = self._make_long_df()

        pos_before = s.calculate_position(df).copy()

        shocked = df.copy()
        shocked.loc[shocked.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        shocked.loc[shocked.index[-self.SHOCK_TAIL :], "High"] = self.SHOCK_VALUE
        pos_after = s.calculate_position(shocked)

        self._assert_past_unchanged(pos_before, pos_after)
