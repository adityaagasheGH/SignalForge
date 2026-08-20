from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import numpy as np

from .metrics import compute_metrics


@dataclass
class BacktestConfig:
    """Configuration for backtest execution and costs."""
    initial_capital: float = 100_000.0
    transaction_cost_pct: float = 0.001    # 0.1% per trade leg
    slippage_pct: float = 0.0005           # 0.05% fill slippage per trade leg
    risk_free_rate: float = 0.0            # Annual risk-free rate


class Backtester:
    """Bar-by-bar backtesting engine enforcing T+1 Open execution."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame, signals: pd.Series) -> dict[str, Any]:
        """Execute bar-by-bar backtest simulation."""
        self._validate_inputs(df, signals)

        equity_values: list[float] = []
        trade_log: list[dict[str, Any]] = []

        cash = self.config.initial_capital
        shares_held: int = 0
        entry_price_net: float = 0.0
        entry_date: Optional[pd.Timestamp] = None
        entry_bar: int = 0
        total_costs_this_trade: float = 0.0

        n = len(df)

        for t in range(n):
            # Prior bar signal executed at current bar's Open price (T+1 Open rule)
            prior_signal = float("nan") if t == 0 else signals.iloc[t - 1]

            open_price = df["Open"].iloc[t]
            close_price = df["Close"].iloc[t]

            if not pd.isna(prior_signal) and not (open_price <= 0):
                if prior_signal == 1.0 and shares_held == 0:
                    # Execute Buy at Open
                    fill_price = self._apply_slippage(open_price, side="buy")
                    shares_held = self._max_shares(cash, fill_price)
                    if shares_held > 0:
                        cost = self._cost(shares_held, fill_price)
                        cash -= shares_held * fill_price + cost
                        total_costs_this_trade = cost
                        entry_price_net = fill_price
                        entry_date = df.index[t]
                        entry_bar = t

                elif prior_signal == -1.0 and shares_held > 0:
                    # Execute Sell at Open
                    fill_price = self._apply_slippage(open_price, side="sell")
                    cost = self._cost(shares_held, fill_price)
                    proceeds = shares_held * fill_price - cost
                    total_costs_this_trade += cost
                    gross_pnl = shares_held * (fill_price - entry_price_net)
                    net_pnl = gross_pnl - total_costs_this_trade

                    trade_log.append({
                        "entry_date": entry_date,
                        "exit_date": df.index[t],
                        "entry_price": entry_price_net,
                        "exit_price": fill_price,
                        "shares": shares_held,
                        "gross_pnl": round(gross_pnl, 6),
                        "net_pnl": round(net_pnl, 6),
                        "total_costs": round(total_costs_this_trade, 6),
                        "bars_held": t - entry_bar,
                    })

                    cash += proceeds
                    shares_held = 0
                    entry_price_net = 0.0
                    total_costs_this_trade = 0.0

            # Record daily portfolio value at Close
            portfolio_value = cash + shares_held * close_price
            equity_values.append(portfolio_value)

        equity_curve = pd.Series(
            equity_values, index=df.index, name="equity"
        )

        result_metrics = compute_metrics(
            equity_curve, trade_log, self.config.risk_free_rate
        )

        return {
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "metrics": result_metrics,
            "config": self.config,
        }

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage adjustment to quoted price."""
        factor = 1 + self.config.slippage_pct if side == "buy" else 1 - self.config.slippage_pct
        return price * factor

    def _cost(self, shares: int, fill_price: float) -> float:
        """Calculate single-leg transaction cost."""
        return shares * fill_price * self.config.transaction_cost_pct

    def _max_shares(self, cash: float, fill_price: float) -> int:
        """Calculate max whole shares affordable with cash."""
        if fill_price <= 0:
            return 0
        effective_price = fill_price * (1 + self.config.transaction_cost_pct)
        return int(cash // effective_price)

    @staticmethod
    def _validate_inputs(df: pd.DataFrame, signals: pd.Series) -> None:
        """Validate input DataFrame and signal alignment."""
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )
        if not df.index.equals(signals.index):
            raise ValueError(
                "DataFrame index and signals index must be identical."
            )
        if len(df) < 2:
            raise ValueError(
                "DataFrame must contain at least 2 bars to run a backtest."
            )
