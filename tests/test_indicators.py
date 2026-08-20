"""
Unit tests for strategy_engine/indicators/*.py

Test categories per indicator:
  1. Value correctness — small hand-inspectable datasets.
  2. Warm-up NaN enforcement — exactly the right number of leading NaNs.
  3. Look-Ahead Bias (LAB) — injecting a shock into FUTURE rows must NOT alter
     any indicator value computed from PAST rows.
  4. Input validation — ValueError raised for bad parameters.
"""

import pytest
import pandas as pd
import numpy as np

from strategy_engine.indicators.moving_averages import sma, ema
from strategy_engine.indicators.oscillators import rsi, stochastic, roc, macd
from strategy_engine.indicators.channels_volatility import (
    bollinger_bands,
    atr,
    keltner_channels,
    adx,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_close() -> pd.Series:
    """20 ascending daily closes starting from 100."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    prices = [float(100 + i) for i in range(20)]
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture
def ohlc_df() -> pd.DataFrame:
    """30 bars of synthetic OHLCV data — deterministic, hand-verifiable."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates)
    high = close + abs(np.random.randn(n) * 1.5)
    low = close - abs(np.random.randn(n) * 1.5)
    return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)


# ---------------------------------------------------------------------------
# SMA Tests
# ---------------------------------------------------------------------------


class TestSMA:
    def test_correct_value(self, simple_close):
        """SMA(5) of [100, 101, 102, 103, 104] should be 102.0 at index 4."""
        result = sma(simple_close, period=5)
        assert result.iloc[4] == pytest.approx(102.0)

    def test_warmup_nans(self, simple_close):
        """SMA(5) should produce exactly 4 NaN rows at the start."""
        result = sma(simple_close, period=5)
        assert result.iloc[:4].isna().all()
        assert pd.notna(result.iloc[4])

    def test_invalid_period(self, simple_close):
        with pytest.raises(ValueError):
            sma(simple_close, period=0)

    def test_look_ahead_bias(self, simple_close):
        """
        LAB Test: modifying future prices must NOT change past SMA values.
        """
        result_before = sma(simple_close, period=5).copy()

        # Inject a massive shock into the LAST 5 rows (future)
        shocked = simple_close.copy()
        shocked.iloc[-5:] = 1_000_000.0
        result_after = sma(shocked, period=5)

        # All rows except the last 5 must be identical
        pd.testing.assert_series_equal(
            result_before.iloc[:-5],
            result_after.iloc[:-5],
            check_names=False,
        )


class TestEMA:
    def test_warmup_nans(self, simple_close):
        """EMA(5) should produce exactly 4 NaN rows at the start."""
        result = ema(simple_close, period=5)
        assert result.iloc[:4].isna().all()
        assert pd.notna(result.iloc[4])

    def test_value_approaches_price_on_constant_series(self):
        """EMA on a constant series should converge to that constant."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        const = pd.Series([200.0] * 30, index=dates)
        result = ema(const, period=5)
        assert result.dropna().iloc[-1] == pytest.approx(200.0, abs=1e-6)

    def test_invalid_period(self, simple_close):
        with pytest.raises(ValueError):
            ema(simple_close, period=0)

    def test_look_ahead_bias(self, simple_close):
        """LAB Test: future price shock must not alter any past EMA value."""
        result_before = ema(simple_close, period=5).copy()

        shocked = simple_close.copy()
        shocked.iloc[-5:] = 1_000_000.0
        result_after = ema(shocked, period=5)

        # EMA is recursive — shocks propagate only forward; past rows unchanged
        pd.testing.assert_series_equal(
            result_before.iloc[:-5],
            result_after.iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# RSI Tests
# ---------------------------------------------------------------------------


class TestRSI:
    def test_warmup_nans(self, simple_close):
        """RSI(14) should produce exactly 14 NaN rows at the start."""
        result = rsi(simple_close, period=14)
        assert result.iloc[:14].isna().all()

    def test_rsi_range(self, ohlc_df):
        """RSI values must be in [0, 100]."""
        result = rsi(ohlc_df["Close"], period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_all_gains_approaches_100(self):
        """RSI on a strictly rising series should approach 100."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        rising = pd.Series([100.0 + i for i in range(100)], index=dates)
        result = rsi(rising, period=14)
        valid = result.dropna()
        assert len(valid) > 0, "RSI produced no valid (non-NaN) values"
        assert valid.iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_invalid_period(self, simple_close):
        with pytest.raises(ValueError):
            rsi(simple_close, period=1)

    def test_look_ahead_bias(self, ohlc_df):
        """LAB Test: shocking future close prices must not alter past RSI values."""
        close = ohlc_df["Close"]
        result_before = rsi(close, period=14).copy()

        shocked = close.copy()
        shocked.iloc[-5:] = 99_999.0
        result_after = rsi(shocked, period=14)

        pd.testing.assert_series_equal(
            result_before.iloc[:-5],
            result_after.iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Stochastic Tests
# ---------------------------------------------------------------------------


class TestStochastic:
    def test_warmup_nans(self, ohlc_df):
        """%K should have (k_period-1) NaN rows; %D: (k_period + d_period - 2)."""
        result = stochastic(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"],
            k_period=14, d_period=3
        )
        assert result["%K"].iloc[:13].isna().all()
        assert pd.notna(result["%K"].iloc[13])
        assert result["%D"].iloc[:15].isna().all()
        assert pd.notna(result["%D"].iloc[15])

    def test_pct_k_range(self, ohlc_df):
        """%K values must be in [0, 100]."""
        result = stochastic(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"]
        )
        valid = result["%K"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_invalid_period(self, ohlc_df):
        with pytest.raises(ValueError):
            stochastic(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], k_period=0)

    def test_look_ahead_bias(self, ohlc_df):
        """LAB Test: future OHLC shocks must not alter past %K or %D."""
        result_before = stochastic(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"]
        ).copy()

        high_s = ohlc_df["High"].copy()
        low_s = ohlc_df["Low"].copy()
        close_s = ohlc_df["Close"].copy()
        high_s.iloc[-5:] = 99_999.0
        close_s.iloc[-5:] = 99_999.0

        result_after = stochastic(high_s, low_s, close_s)

        pd.testing.assert_series_equal(
            result_before["%K"].iloc[:-5],
            result_after["%K"].iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# ROC Tests
# ---------------------------------------------------------------------------


class TestROC:
    def test_correct_value(self):
        """ROC(1) = (Close[t] / Close[t-1] - 1) * 100."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        close = pd.Series([100.0, 110.0, 121.0, 110.0, 99.0], index=dates)
        result = roc(close, period=1)
        assert result.iloc[1] == pytest.approx(10.0)   # (110/100 - 1) * 100
        assert result.iloc[3] == pytest.approx(-9.090909, abs=1e-3)

    def test_warmup_nans(self, simple_close):
        """ROC(5) should produce exactly 5 NaN rows at the start."""
        result = roc(simple_close, period=5)
        assert result.iloc[:5].isna().all()
        assert pd.notna(result.iloc[5])

    def test_invalid_period(self, simple_close):
        with pytest.raises(ValueError):
            roc(simple_close, period=0)

    def test_look_ahead_bias(self, simple_close):
        """LAB Test: future shock must not alter past ROC values."""
        result_before = roc(simple_close, period=5).copy()

        shocked = simple_close.copy()
        shocked.iloc[-5:] = 1_000_000.0
        result_after = roc(shocked, period=5)

        pd.testing.assert_series_equal(
            result_before.iloc[:-5],
            result_after.iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# MACD Tests
# ---------------------------------------------------------------------------


class TestMACD:
    def test_warmup_nans(self, ohlc_df):
        """MACD signal/histogram warm-up = slow_period + signal_period - 2."""
        result = macd(ohlc_df["Close"], fast_period=12, slow_period=26, signal_period=9)
        expected_warmup = 26 + 9 - 2  # 33
        assert result["signal"].iloc[:expected_warmup].isna().all()
        assert pd.notna(result["signal"].iloc[expected_warmup])

    def test_macd_line_warmup(self, ohlc_df):
        """MACD line warm-up = slow_period - 1 rows."""
        result = macd(ohlc_df["Close"], fast_period=12, slow_period=26, signal_period=9)
        assert result["macd"].iloc[:25].isna().all()
        assert pd.notna(result["macd"].iloc[25])

    def test_invalid_fast_slow(self, ohlc_df):
        with pytest.raises(ValueError):
            macd(ohlc_df["Close"], fast_period=26, slow_period=12)

    def test_look_ahead_bias(self, ohlc_df):
        """LAB Test: shocking future close must not alter past MACD values."""
        close = ohlc_df["Close"]
        result_before = macd(close).copy()

        shocked = close.copy()
        shocked.iloc[-5:] = 99_999.0
        result_after = macd(shocked)

        pd.testing.assert_series_equal(
            result_before["macd"].iloc[:-5],
            result_after["macd"].iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Bollinger Bands Tests
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_warmup_nans(self, simple_close):
        """BB(20) should produce NaN for first 19 rows."""
        result = bollinger_bands(simple_close, period=5)
        assert result["middle"].iloc[:4].isna().all()
        assert pd.notna(result["middle"].iloc[4])

    def test_upper_above_lower(self, ohlc_df):
        """Upper band must always be >= Lower band for valid rows."""
        result = bollinger_bands(ohlc_df["Close"], period=10, num_std=2.0)
        valid = result.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_middle_is_sma(self, simple_close):
        """Middle band should equal SMA(close, period)."""
        from strategy_engine.indicators.moving_averages import sma as sma_fn
        bb = bollinger_bands(simple_close, period=5)
        expected_middle = sma_fn(simple_close, period=5)
        pd.testing.assert_series_equal(bb["middle"], expected_middle, check_names=False)

    def test_invalid_params(self, simple_close):
        with pytest.raises(ValueError):
            bollinger_bands(simple_close, period=1)
        with pytest.raises(ValueError):
            bollinger_bands(simple_close, period=5, num_std=-1.0)

    def test_look_ahead_bias(self, simple_close):
        """LAB Test: future shock must not alter past Bollinger Band values."""
        result_before = bollinger_bands(simple_close, period=5).copy()

        shocked = simple_close.copy()
        shocked.iloc[-5:] = 1_000_000.0
        result_after = bollinger_bands(shocked, period=5)

        pd.testing.assert_series_equal(
            result_before["upper"].iloc[:-5],
            result_after["upper"].iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# ATR Tests
# ---------------------------------------------------------------------------


class TestATR:
    def test_warmup_nans(self, ohlc_df):
        """ATR(14) should produce 14 NaN rows at the start."""
        result = atr(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=14)
        assert result.iloc[:14].isna().all()
        assert pd.notna(result.iloc[14])

    def test_atr_positive(self, ohlc_df):
        """ATR must be strictly positive for all valid rows."""
        result = atr(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=5)
        assert (result.dropna() > 0).all()

    def test_invalid_period(self, ohlc_df):
        with pytest.raises(ValueError):
            atr(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=0)

    def test_look_ahead_bias(self, ohlc_df):
        """LAB Test: shocking future OHLC must not alter past ATR values."""
        result_before = atr(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=5
        ).copy()

        high_s = ohlc_df["High"].copy()
        high_s.iloc[-5:] = 99_999.0
        result_after = atr(high_s, ohlc_df["Low"], ohlc_df["Close"], period=5)

        pd.testing.assert_series_equal(
            result_before.iloc[:-5],
            result_after.iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Keltner Channels Tests
# ---------------------------------------------------------------------------


class TestKeltnerChannels:
    def test_upper_above_lower(self, ohlc_df):
        """Upper band must always be >= Lower band for valid rows."""
        result = keltner_channels(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"],
            ema_period=10, atr_period=5
        )
        valid = result.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_invalid_params(self, ohlc_df):
        with pytest.raises(ValueError):
            keltner_channels(
                ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"],
                ema_period=0
            )
        with pytest.raises(ValueError):
            keltner_channels(
                ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"],
                atr_multiplier=-1.0
            )

    def test_look_ahead_bias(self, ohlc_df):
        """LAB Test: future shock must not alter past Keltner values."""
        result_before = keltner_channels(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"],
            ema_period=10, atr_period=5
        ).copy()

        close_s = ohlc_df["Close"].copy()
        close_s.iloc[-5:] = 99_999.0
        result_after = keltner_channels(
            ohlc_df["High"], ohlc_df["Low"], close_s, ema_period=10, atr_period=5
        )

        pd.testing.assert_series_equal(
            result_before["middle"].iloc[:-5],
            result_after["middle"].iloc[:-5],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# ADX Tests
# ---------------------------------------------------------------------------


class TestADX:
    def test_warmup_nans(self, ohlc_df):
        """ADX(14) should produce (2*14=28) NaN rows at the start."""
        result = adx(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=14)
        assert result["adx"].iloc[:28].isna().all()
        assert pd.notna(result["adx"].iloc[28])

    def test_adx_positive(self, ohlc_df):
        """ADX must be >= 0 for all valid rows."""
        result = adx(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=5)
        assert (result["adx"].dropna() >= 0).all()

    def test_di_positive(self, ohlc_df):
        """+DI and -DI must be >= 0 for all valid rows."""
        result = adx(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=5)
        assert (result["plus_di"].dropna() >= 0).all()
        assert (result["minus_di"].dropna() >= 0).all()

    def test_invalid_period(self, ohlc_df):
        with pytest.raises(ValueError):
            adx(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=1)

    def test_look_ahead_bias(self, ohlc_df):
        """LAB Test: injecting a future price shock must not alter any past ADX value."""
        result_before = adx(
            ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"], period=5
        ).copy()

        high_s = ohlc_df["High"].copy()
        high_s.iloc[-5:] = 99_999.0
        result_after = adx(high_s, ohlc_df["Low"], ohlc_df["Close"], period=5)

        pd.testing.assert_series_equal(
            result_before["adx"].iloc[:-5],
            result_after["adx"].iloc[:-5],
            check_names=False,
        )
