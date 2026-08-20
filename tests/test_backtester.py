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


def make_ohlcv(prices: list[float], start_date: str = "2020-01-01") -> pd.DataFrame:
    """Helper to generate OHLCV DataFrame from close prices."""
    n = len(prices)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    close_series = pd.Series(prices, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "Open": close_series,
            "High": close_series * 1.01,
            "Low": close_series * 0.99,
            "Close": close_series,
            "Volume": [100_000] * n,
        },
        index=dates,
    )


class TestMetrics:
    def test_total_return(self):
        eq = pd.Series([100.0, 110.0, 120.0, 150.0])
        assert _total_return(eq) == pytest.approx(0.50)

    def test_max_drawdown(self):
        eq = pd.Series([100.0, 200.0, 100.0, 150.0])
        assert _max_drawdown(eq) == pytest.approx(0.50)

    def test_cagr(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        prices = np.linspace(100, 120, 252)
        eq = pd.Series(prices, index=dates)
        assert _cagr(eq) == pytest.approx(0.20, abs=0.01)

    def test_sharpe_ratio_zero_vol(self):
        returns = pd.Series([0.01] * 10)
        assert _sharpe_ratio(returns) == 0.0

    def test_sortino_no_downside(self):
        returns = pd.Series([0.01, 0.02, 0.015, 0.03])
        assert _sortino_ratio(returns) == float("inf")


class TestBacktesterExecution:
    def test_t_plus_one_open_execution(self):
        """Verify signal at T executes on T+1 Open."""
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

        config = BacktestConfig(initial_capital=1000.0, transaction_cost_pct=0.0, slippage_pct=0.0)
        bt = Backtester(config)
        res = bt.run(df, signals)

        trade_log = res["trade_log"]
        assert len(trade_log) == 1
        trade = trade_log[0]

        assert trade["entry_date"] == dates[1]
        assert trade["entry_price"] == pytest.approx(110.0)
        assert trade["exit_date"] == dates[3]
        assert trade["exit_price"] == pytest.approx(130.0)
        assert trade["shares"] == 9

    def test_transaction_cost_and_slippage_deduction(self):
        """Verify slippage and cost deductions on trade log."""
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

        config = BacktestConfig(
            initial_capital=10000.0,
            transaction_cost_pct=0.01,
            slippage_pct=0.01,
        )
        bt = Backtester(config)
        res = bt.run(df, signals)

        trade_log = res["trade_log"]
        assert len(trade_log) == 1
        trade = trade_log[0]

        assert trade["entry_price"] == pytest.approx(101.0)
        assert trade["exit_price"] == pytest.approx(99.0)
        assert trade["total_costs"] > 0
        assert trade["net_pnl"] < trade["gross_pnl"]

    def test_invalid_df_raises(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        invalid_df = pd.DataFrame({"Close": [10.0, 12.0]}, index=dates)
        signals = pd.Series([1.0, 0.0], index=dates)

        bt = Backtester()
        with pytest.raises(ValueError, match="missing required columns"):
            bt.run(invalid_df, signals)
