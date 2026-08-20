"""
SignalForge — Bar-by-Bar Backtesting Simulator

Implements the backtesting engine described in blueprint.md §7.

Key invariants (hard rules, never relaxed):
  1. EXECUTION TIMING: a signal from bar T's close EXECUTES at bar T+1's OPEN.
     The simulator never reads bar T's open to fill bar T's signal.
  2. LONG-ONLY: v1 supports only long positions (no short selling).
  3. FULL-POSITION SIZING: when invested, the full available equity is deployed.
     Fractional share sizing: whole shares only (blueprint §29).
  4. COST ACCOUNTING: every entry and exit charges transaction_cost_pct of trade
     value, plus a fixed slippage on the fill price.
  5. NO SURVIVAL BIAS: the engine never skips NaN bars; it simply stays in its
     current state (hold or flat) when signals are NaN (warm-up).

Engine flow per bar (T):
  - Read signal[T-1] (yesterday's signal — already available at today's open).
  - If signal == 1 and currently flat  → BUY  at Open[T] (+ costs/slippage).
  - If signal == -1 and currently long → SELL at Open[T] (+ costs/slippage).
  - Record portfolio value at Close[T] for equity curve.

Outputs:
  - equity_curve:  pd.Series of daily portfolio values.
  - trade_log:     list[dict] — one entry per closed round-trip trade.
  - summary:       dict of all performance metrics via metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import math

import pandas as pd
import numpy as np

from .metrics import compute_metrics


@dataclass
class BacktestConfig:
    """
    All configurable backtesting parameters — zero hardcoded literals in the engine.
    Per blueprint §29 and ENGINEERING_STANDARDS §5.
    """
    initial_capital: float = 100_000.0
    transaction_cost_pct: float = 0.001    # 0.1% of trade value per side
    slippage_pct: float = 0.0005           # 0.05% fill-price slippage per side
    risk_free_rate: float = 0.0            # Annual risk-free rate for Sharpe/Sortino


class Backtester:
    """
    Bar-by-bar backtesting engine.

    Usage:
        config = BacktestConfig(initial_capital=100_000, transaction_cost_pct=0.001)
        bt = Backtester(config)
        result = bt.run(df, signals)

    :param config: BacktestConfig instance controlling all cost/capital parameters.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, signals: pd.Series) -> dict[str, Any]:
        """
        Execute the bar-by-bar simulation.

        :param df: OHLCV DataFrame (must have 'Open', 'High', 'Low', 'Close', 'Volume').
                   Index must be a DatetimeIndex in ascending chronological order.
        :param signals: pd.Series[float] aligned to df.index.
                        Values: 1.0 = BUY, -1.0 = SELL/EXIT, 0.0 = HOLD, NaN = warm-up.
        :return: dict with keys:
                   'equity_curve': pd.Series of daily portfolio values
                   'trade_log':    list of closed-trade dicts
                   'metrics':      dict of all performance metrics
                   'config':       BacktestConfig used for this run
        """
        self._validate_inputs(df, signals)

        equity_values: list[float] = []
        trade_log: list[dict[str, Any]] = []

        cash = self.config.initial_capital
        shares_held: int = 0
        entry_price_net: float = 0.0    # fill price including slippage (for P&L)
        entry_date: Optional[pd.Timestamp] = None
        entry_bar: int = 0              # bar index of entry (for bars_held)
        total_costs_this_trade: float = 0.0

        n = len(df)

        for t in range(n):
            # ----------------------------------------------------------------
            # Determine the actionable signal: signal from bar T-1
            # At bar T=0, there is no prior signal → stay flat.
            # ----------------------------------------------------------------
            prior_signal = float("nan") if t == 0 else signals.iloc[t - 1]

            open_price = df["Open"].iloc[t]
            close_price = df["Close"].iloc[t]

            # ----------------------------------------------------------------
            # Execute entry/exit at today's OPEN based on yesterday's signal
            # (Hard rule: execution at T+1's open, never T's close)
            # ----------------------------------------------------------------

            if not pd.isna(prior_signal) and not (open_price <= 0):
                if prior_signal == 1.0 and shares_held == 0:
                    # --- BUY ---
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
                    # --- SELL / EXIT ---
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

            # ----------------------------------------------------------------
            # Record portfolio value at end of bar using Close price
            # ----------------------------------------------------------------
            portfolio_value = cash + shares_held * close_price
            equity_values.append(portfolio_value)

        equity_curve = pd.Series(
            equity_values, index=df.index, name="equity"
        )

        # If still holding at end, close the position (mark-to-market, no forced exit cost)
        # This open trade is NOT added to the trade log (incomplete round-trip).

        result_metrics = compute_metrics(
            equity_curve, trade_log, self.config.risk_free_rate
        )

        return {
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "metrics": result_metrics,
            "config": self.config,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Adjust fill price for slippage.
        Buys fill HIGHER than quoted (adverse), sells fill LOWER.

        :param price: Quoted open price.
        :param side: 'buy' or 'sell'.
        :return: Adjusted fill price.
        """
        factor = 1 + self.config.slippage_pct if side == "buy" else 1 - self.config.slippage_pct
        return price * factor

    def _cost(self, shares: int, fill_price: float) -> float:
        """
        Compute one-sided transaction cost for a trade leg.

        :param shares: Number of shares in the trade.
        :param fill_price: Fill price per share.
        :return: Total cost in currency units.
        """
        return shares * fill_price * self.config.transaction_cost_pct

    def _max_shares(self, cash: float, fill_price: float) -> int:
        """
        Compute maximum whole shares purchasable with available cash,
        accounting for transaction costs on the buy side.

        Whole shares only (blueprint §29 — no fractional shares in v1).
        Solves: shares * price + shares * price * cost_pct <= cash
                shares <= cash / (price * (1 + cost_pct))
        """
        if fill_price <= 0:
            return 0
        effective_price = fill_price * (1 + self.config.transaction_cost_pct)
        return int(cash // effective_price)

    @staticmethod
    def _validate_inputs(df: pd.DataFrame, signals: pd.Series) -> None:
        """
        Validate DataFrame and signals before simulation.

        :raises ValueError: If required columns are missing or indices don't align.
        """
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
