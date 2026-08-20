"""
Unit tests for strategy_engine/backtesting/ (Backtester & metrics engine)
"""

import pytest
import pandas as pd
import numpy as np

from strategy_engine.backtesting.simulator import Backtester, BacktestConfig
from strategy_engine.backtesting.metrics import (
    compute_metrics,
    _total_return,
    _cagr,
    _annualised_volatility,
    _max_drawdown,
    _sharpe_ratio,
    _sortino_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(prices: list[float], start_date: str = "2020-01-01") -> pd.DataFrame:
    """Helper to generate a simple OHLCV DataFrame from close prices."""
    n = len(prices)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    close_series = pd.Series(prices, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "Open": close_series,  # simple 1:1 open=close for basic math testing
            "High": close_series * 1.01,
            "Low": close_series * 0.99,
            "Close": close_series,
            "Volume": [100_000] * n,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_total_return(self):
        eq = pd.Series([100.0, 110.0, 120.0, 150.0])
        assert _total_return(eq) == pytest.approx(0.50)

    def test_max_drawdown(self):
        # 100 -> 200 -> 100 -> 150. Peak=200, Trough=100 -> Max DD = (200-100)/200 = 0.50 (50%)
        eq = pd.Series([100.0, 200.0, 100.0, 150.0])
        assert _max_drawdown(eq) == pytest.approx(0.50)

    def test_cagr(self):
        # 252 bars = 1 year. Start=100, End=120 -> CAGR should be 20% (0.20)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        prices = np.linspace(100, 120, 252)
        eq = pd.Series(prices, index=dates)
        assert _cagr(eq) == pytest.approx(0.20, abs=0.01)

    def test_sharpe_ratio_zero_vol(self):
        # Constant returns -> std=0 -> Sharpe should be 0.0
        returns = pd.Series([0.01] * 10)
        assert _sharpe_ratio(returns) == 0.0

    def test_sortino_no_downside(self):
        # All positive returns -> perfect Sortino (inf)
        returns = pd.Series([0.01, 0.02, 0.015, 0.03])
        assert _sortino_ratio(returns) == float("inf")


# ---------------------------------------------------------------------------
# 2. Simulator Execution & Timing Tests
# ---------------------------------------------------------------------------

class TestBacktesterExecution:
    def test_t_plus_one_open_execution(self):
        """
        Hard Rule Test: Signal generated on bar T (Close[T]) MUST execute on bar T+1 (Open[T+1]).
        """
        # 4-bar scenario
        # Bar 0 (Day 1): Close=100, Signal=1.0 (Buy)
        # Bar 1 (Day 2): Open=110, Close=110, Signal=0.0
        # Bar 2 (Day 3): Open=120, Close=120, Signal=-1.0 (Sell)
        # Bar 3 (Day 4): Open=130, Close=130, Signal=0.0
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0, 110.0, 120.0, 130.0],
                "High": [105.0, 115.0, 125.0, 135.0],
                "Low": [95.0, 105.0, 115.0, 125.0],
                "Close": [100.0, 110.0, 120.0, 130.0],
                "Volume": [1000] * 4,
            },
            index=dates,
        )

        signals = pd.Series([1.0, 0.0, -1.0, 0.0], index=dates)

        # Zero transaction costs / zero slippage for exact price verification
        config = BacktestConfig(initial_capital=1000.0, transaction_cost_pct=0.0, slippage_pct=0.0)
        bt = Backtester(config)
        res = bt.run(df, signals)

        trade_log = res["trade_log"]
        assert len(trade_log) == 1
        trade = trade_log[0]

        # Buy signal on Day 1 (index 0) executes at Day 2 (index 1) Open = 110.0
        assert trade["entry_date"] == dates[1]
        assert trade["entry_price"] == pytest.approx(110.0)

        # Sell signal on Day 3 (index 2) executes at Day 4 (index 3) Open = 130.0
        assert trade["exit_date"] == dates[3]
        assert trade["exit_price"] == pytest.approx(130.0)

        # Shares bought with $1000 at $110 = 9 shares ($990 value, $10 cash left)
        assert trade["shares"] == 9

    def test_transaction_cost_and_slippage_deduction(self):
        """
        Verify that slippage increases buy fill price, decreases sell fill price,
        and transaction cost reduces cash.
        """
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0, 100.0],
                "High": [105.0, 105.0, 105.0, 105.0],
                "Low": [95.0, 95.0, 95.0, 95.0],
                "Close": [100.0, 100.0, 100.0, 100.0],
                "Volume": [1000] * 4,
            },
            index=dates,
        )

        signals = pd.Series([1.0, 0.0, -1.0, 0.0], index=dates)

        # 1% slippage, 1% transaction fee
        config = BacktestConfig(
            initial_capital=10000.0,
            transaction_cost_pct=0.01,  # 1%
            slippage_pct=0.01,         # 1%
        )
        bt = Backtester(config)
        res = bt.run(df, signals)

        trade_log = res["trade_log"]
        assert len(trade_log) == 1
        trade = trade_log[0]

        # Quoted Open = 100. Buy fill price with 1% slippage = 101.0
        assert trade["entry_price"] == pytest.approx(101.0)

        # Quoted Open = 100. Sell fill price with 1% slippage = 99.0
        assert trade["exit_price"] == pytest.approx(99.0)

        # Total costs charged for buy + sell legs should be strictly > 0
        assert trade["total_costs"] > 0
        # Gross PnL: shares * (99.0 - 101.0) = negative
        # Net PnL: Gross PnL - total_costs
        assert trade["net_pnl"] < trade["gross_pnl"]

    def test_invalid_df_raises(self):
        """Missing required columns raises ValueError."""
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        invalid_df = pd.DataFrame({"Close": [10.0, 12.0]}, index=dates)
        signals = pd.Series([1.0, 0.0], index=dates)

        bt = Backtester()
        with pytest.raises(ValueError, match="missing required columns"):
            bt.run(invalid_df, signals)
