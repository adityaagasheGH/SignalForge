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
    """Compute performance metric suite."""
    metrics: dict[str, Any] = {}

    daily_returns = equity_curve.pct_change().dropna()

    metrics["total_return"] = _total_return(equity_curve)
    metrics["cagr"] = _cagr(equity_curve)
    metrics["annualised_volatility"] = _annualised_volatility(daily_returns)
    metrics["max_drawdown"] = _max_drawdown(equity_curve)
    metrics["sharpe_ratio"] = _sharpe_ratio(daily_returns, risk_free_rate)
    metrics["sortino_ratio"] = _sortino_ratio(daily_returns, risk_free_rate)
    metrics["exposure_time"] = _exposure_time(equity_curve, trade_log)

    trade_metrics = _trade_stats(trade_log)
    metrics.update(trade_metrics)

    return metrics


def _total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start == 0:
        return 0.0
    return float(end / start - 1)


def _cagr(equity_curve: pd.Series) -> float:
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


def _annualised_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(-drawdown.min())


def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    std = float(excess.std(ddof=1))
    if np.isclose(std, 0.0, atol=1e-12) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _exposure_time(
    equity_curve: pd.Series,
    trade_log: list[dict[str, Any]],
) -> float:
    total_bars = len(equity_curve)
    if total_bars == 0:
        return 0.0
    in_market_bars = sum(t.get("bars_held", 0) for t in trade_log)
    return min(float(in_market_bars / total_bars), 1.0)


def _trade_stats(trade_log: list[dict[str, Any]]) -> dict[str, float]:
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
