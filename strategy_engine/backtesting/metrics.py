"""
SignalForge — Performance Metrics Engine

Computes all quantitative metrics used by the Strategy Scoring Engine (blueprint §8–9).
All functions are pure — they take a daily returns/equity Series and return scalar values.

Metrics computed (blueprint §8):
  Returns:         total_return, cagr
  Risk:            annualised_volatility, max_drawdown
  Risk-adjusted:   sharpe_ratio, sortino_ratio
  Trading stats:   win_rate, profit_factor, num_trades,
                   avg_trade_return, avg_win, avg_loss
  Exposure:        exposure_time (fraction of bars in market)

Conventions:
  - Daily returns are used throughout; annualisation uses 252 trading days / year.
  - Sharpe and Sortino use a configurable risk-free rate (default 0.0).
  - Max drawdown is expressed as a positive fraction, e.g. 0.25 means a 25% drawdown.
  - A trade is "winning" if its net P&L (after costs) > 0.
  - Profit factor = gross_wins / gross_losses; returns inf if no losses.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


def compute_metrics(
    equity_curve: pd.Series,
    trade_log: list[dict[str, Any]],
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Compute the full performance metric suite from an equity curve and trade log.

    :param equity_curve: pd.Series of portfolio values indexed by date (daily).
                         Must contain at least 2 data points.
    :param trade_log:    List of closed-trade dicts from Backtester.
                         Each dict must have keys: 'net_pnl', 'gross_pnl', 'in_market'.
    :param risk_free_rate: Annual risk-free rate as a decimal (default 0.0).
    :return: dict of metric name → scalar value.
    """
    metrics: dict[str, Any] = {}

    # --- Daily returns from equity curve ------------------------------------------
    daily_returns = equity_curve.pct_change().dropna()

    metrics["total_return"] = _total_return(equity_curve)
    metrics["cagr"] = _cagr(equity_curve)
    metrics["annualised_volatility"] = _annualised_volatility(daily_returns)
    metrics["max_drawdown"] = _max_drawdown(equity_curve)
    metrics["sharpe_ratio"] = _sharpe_ratio(daily_returns, risk_free_rate)
    metrics["sortino_ratio"] = _sortino_ratio(daily_returns, risk_free_rate)
    metrics["exposure_time"] = _exposure_time(equity_curve, trade_log)

    # --- Trade-level stats --------------------------------------------------------
    trade_metrics = _trade_stats(trade_log)
    metrics.update(trade_metrics)

    return metrics


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def _total_return(equity_curve: pd.Series) -> float:
    """(Final value / Initial value) - 1, expressed as a decimal."""
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start == 0:
        return 0.0
    return float(end / start - 1)


def _cagr(equity_curve: pd.Series) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (final / initial) ^ (1 / years) - 1
    Years is computed from number of bars / TRADING_DAYS_PER_YEAR.
    """
    n_bars = len(equity_curve)
    if n_bars < 2:
        return 0.0
    years = n_bars / TRADING_DAYS_PER_YEAR
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start <= 0 or years <= 0:
        return 0.0
    ratio = end / start
    if ratio <= 0:
        return -1.0
    return float(ratio ** (1 / years) - 1)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def _annualised_volatility(daily_returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown, returned as a positive fraction.
    e.g. 0.25 means equity fell 25% from a peak before recovering.
    """
    if len(equity_curve) < 2:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(-drawdown.min())   # positive convention


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------

def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.

    Sharpe = (mean_daily_excess_return / std_daily_return) * sqrt(252)

    Returns 0.0 if std is near zero (constant returns).
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    std = float(excess.std(ddof=1))
    if np.isclose(std, 0.0, atol=1e-12) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sortino ratio — penalises only downside volatility.

    Sortino = (mean_excess_return / downside_std) * sqrt(252)

    Downside std uses only returns below the risk-free rate.
    Returns 0.0 if no negative excess returns exist.
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")   # no negative days → perfect Sortino
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def _exposure_time(
    equity_curve: pd.Series,
    trade_log: list[dict[str, Any]],
) -> float:
    """
    Fraction of total bars where the strategy was in the market (holding a position).

    :return: Float in [0.0, 1.0]; 1.0 = always invested.
    """
    total_bars = len(equity_curve)
    if total_bars == 0:
        return 0.0
    in_market_bars = sum(t.get("bars_held", 0) for t in trade_log)
    return min(float(in_market_bars / total_bars), 1.0)


# ---------------------------------------------------------------------------
# Trade-level statistics
# ---------------------------------------------------------------------------

def _trade_stats(trade_log: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute win rate, profit factor, and trade-level P&L statistics.

    :param trade_log: List of closed-trade dicts with at least 'net_pnl' key.
    :return: dict of trade stats.
    """
    if not trade_log:
        return {
            "num_trades": 0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_trade_return": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
        }

    net_pnls = [t["net_pnl"] for t in trade_log]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p < 0]

    num_trades = len(net_pnls)
    win_rate = len(wins) / num_trades if num_trades > 0 else float("nan")
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_trade_return = float(np.mean(net_pnls)) if net_pnls else float("nan")
    avg_win = float(np.mean(wins)) if wins else float("nan")
    avg_loss = float(np.mean(losses)) if losses else float("nan")

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }
