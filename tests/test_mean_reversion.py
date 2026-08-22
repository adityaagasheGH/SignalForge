import pytest
import pandas as pd
import numpy as np

from strategy_engine.strategies.mean_reversion import MeanReversionStrategy
from strategy_engine.strategies.base import BaseStrategy


def make_df(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build minimal OHLCV DataFrame from close prices."""
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
def default_strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(lookback=5, num_std=1.5)


@pytest.fixture
def reversion_df() -> pd.DataFrame:
    baseline = [100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 100.5, 99.5, 100.0, 101.0]
    drop = [85.0, 80.0, 78.0, 76.0, 75.0, 74.0]
    recover = [90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
    return make_df(baseline + drop + recover)


class TestInit:
    def test_default_params(self):
        s = MeanReversionStrategy()
        assert s.lookback == 20
        assert s.num_std == 2.0
        assert s.entry_threshold == 1.0
        assert s.exit_threshold == 1.0

    def test_custom_params(self):
        s = MeanReversionStrategy(
            lookback=10, num_std=2.5, entry_threshold=1.2, exit_threshold=0.5
        )
        assert s.lookback == 10
        assert s.num_std == 2.5
        assert s.entry_threshold == 1.2
        assert s.exit_threshold == 0.5

    def test_lookback_too_small_raises(self):
        with pytest.raises(ValueError, match="lookback"):
            MeanReversionStrategy(lookback=1)

    def test_non_positive_num_std_raises(self):
        with pytest.raises(ValueError, match="num_std"):
            MeanReversionStrategy(num_std=0)
        with pytest.raises(ValueError, match="num_std"):
            MeanReversionStrategy(num_std=-1.0)

    def test_non_positive_entry_threshold_raises(self):
        with pytest.raises(ValueError, match="entry_threshold"):
            MeanReversionStrategy(entry_threshold=0)

    def test_non_positive_exit_threshold_raises(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            MeanReversionStrategy(exit_threshold=-0.1)

    def test_is_base_strategy_subclass(self):
        assert issubclass(MeanReversionStrategy, BaseStrategy)


class TestGenerateSignals:
    def test_output_index_matches_df(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        assert signals.index.equals(reversion_df.index)

    def test_warmup_rows_are_nan(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        assert signals.iloc[: default_strategy.lookback - 1].isna().all()
        assert signals.iloc[default_strategy.lookback - 1 :].notna().any()

    def test_signal_values_are_valid(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        valid = signals.dropna()
        assert set(valid.unique()).issubset({1.0, 0.0, -1.0})

    def test_drop_produces_buy_signal(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        assert (signals == 1.0).any()

    def test_recovery_produces_sell_signal(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        assert (signals == -1.0).any()

    def test_missing_close_column_raises(self, default_strategy):
        df = pd.DataFrame({"Open": [100.0], "High": [101.0]})
        with pytest.raises(KeyError, match="Close"):
            default_strategy.generate_signals(df)

    def test_buy_only_when_close_at_or_below_entry_line(
        self, default_strategy, reversion_df
    ):
        from strategy_engine.indicators.channels_volatility import bollinger_bands

        signals = default_strategy.generate_signals(reversion_df)
        bands = bollinger_bands(
            reversion_df["Close"],
            period=default_strategy.lookback,
            num_std=default_strategy.num_std,
        )
        band_width = bands["middle"] - bands["lower"]
        entry_line = bands["middle"] - default_strategy.entry_threshold * band_width

        buy_bars = signals[signals == 1.0].index
        assert len(buy_bars) > 0
        assert (reversion_df.loc[buy_bars, "Close"] <= entry_line.loc[buy_bars]).all()

    def test_stricter_entry_threshold_reduces_or_matches_buy_count(
        self, reversion_df
    ):
        loose = MeanReversionStrategy(lookback=5, num_std=1.5, entry_threshold=1.0)
        strict = MeanReversionStrategy(lookback=5, num_std=1.5, entry_threshold=2.0)

        loose_buys = (loose.generate_signals(reversion_df) == 1.0).sum()
        strict_buys = (strict.generate_signals(reversion_df) == 1.0).sum()

        assert strict_buys <= loose_buys

    def test_hold_bars_are_zero(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        valid = signals.dropna()
        hold_count = (valid == 0.0).sum()
        assert hold_count > 0


class TestCalculatePosition:
    def test_position_is_01_or_nan(self, default_strategy, reversion_df):
        pos = default_strategy.calculate_position(reversion_df)
        valid = pos.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_position_goes_flat_after_exit(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        pos = default_strategy.calculate_position(reversion_df)
        sell_idx = signals[signals == -1.0].index
        if len(sell_idx) == 0:
            pytest.skip("No exit signal produced.")
        first_sell_loc = reversion_df.index.get_loc(sell_idx[0])
        assert pos.iloc[first_sell_loc] == 0.0

    def test_position_held_between_entry_and_exit(self, default_strategy, reversion_df):
        signals = default_strategy.generate_signals(reversion_df)
        pos = default_strategy.calculate_position(reversion_df)

        buy_idx = signals[signals == 1.0].index
        sell_idx = signals[signals == -1.0].index
        if len(buy_idx) == 0 or len(sell_idx) == 0:
            pytest.skip("Need both an entry and an exit in fixture.")

        first_buy_loc = reversion_df.index.get_loc(buy_idx[0])
        later_sells = [reversion_df.index.get_loc(d) for d in sell_idx if reversion_df.index.get_loc(d) > first_buy_loc]
        if not later_sells:
            pytest.skip("No exit after first entry in fixture.")
        first_sell_loc = later_sells[0]

        between_slice = pos.iloc[first_buy_loc:first_sell_loc]
        assert (between_slice == 1.0).all()


class TestParameterCustomisation:
    def test_get_parameters_structure(self, default_strategy):
        params = default_strategy.get_parameters()
        assert "values" in params
        assert "grid" in params
        assert params["values"]["lookback"] == 5
        assert params["values"]["num_std"] == 1.5
        assert params["values"]["entry_threshold"] == 1.0
        assert params["values"]["exit_threshold"] == 1.0

    def test_get_parameters_values_match_init(self):
        s = MeanReversionStrategy(
            lookback=15, num_std=2.5, entry_threshold=1.1, exit_threshold=0.8
        )
        p = s.get_parameters()["values"]
        assert p["lookback"] == 15
        assert p["num_std"] == 2.5
        assert p["entry_threshold"] == 1.1
        assert p["exit_threshold"] == 0.8

    def test_default_grid_matches_blueprint(self, default_strategy):
        grid = default_strategy.get_parameters()["grid"]
        assert grid["lookback"] == [10, 20, 30, 50]
        assert grid["num_std"] == [1.5, 2.0, 2.5, 3.0]
        assert grid["entry_threshold"] == [0.75, 1.0, 1.25]
        assert grid["exit_threshold"] == [0.5, 1.0, 1.25]


class TestLookAheadBias:
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

    def test_signals_unaffected_by_future_shock(self):
        s = MeanReversionStrategy(lookback=10, num_std=2.0)
        df = self._make_long_df()

        signals_before = s.generate_signals(df).copy()

        shocked_df = df.copy()
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "High"] = self.SHOCK_VALUE
        signals_after = s.generate_signals(shocked_df)

        self._assert_past_unchanged(signals_before, signals_after)

    def test_position_unaffected_by_future_shock(self):
        s = MeanReversionStrategy(lookback=10, num_std=2.0)
        df = self._make_long_df()

        pos_before = s.calculate_position(df).copy()

        shocked_df = df.copy()
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Close"] = self.SHOCK_VALUE
        pos_after = s.calculate_position(shocked_df)

        self._assert_past_unchanged(pos_before, pos_after)

    def test_shock_downward_also_leaves_past_unchanged(self):
        s = MeanReversionStrategy(lookback=10, num_std=2.0)
        df = self._make_long_df()

        signals_before = s.generate_signals(df).copy()

        shocked_df = df.copy()
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Close"] = 0.01
        shocked_df.loc[shocked_df.index[-self.SHOCK_TAIL :], "Low"] = 0.01
        signals_after = s.generate_signals(shocked_df)

        self._assert_past_unchanged(signals_before, signals_after)
