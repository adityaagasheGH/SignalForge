# SignalForge — Data Provider Architecture & Decision Record

**Document Status:** Approved Architecture Decision Record (ADR)  
**Target Module:** Data Ingestion & Storage (`data/`, `strategy_engine/`)  
**Primary Provider:** `yfinance`  

---

## 1. Executive Summary

For SignalForge's historical market data ingestion engine, **`yfinance`** has been selected as the primary market data provider for daily equity OHLCV series. This document records the evaluation criteria, technical requirements, rationale, and operational constraints governing this selection.

---

## 2. Decision Context & Requirements

SignalForge requires a reliable, scalable, and cost-effective source of historical price data to drive its strategy evaluation engine, technical indicator calculations, parameter optimization sweeps, and regime-classification algorithms.

Per `blueprint.md` (§2) and `PRINCIPLES.md`, the data provider must satisfy the following non-negotiable requirements:

1. **Indian Equities (NSE) Support:** Must support National Stock Exchange of India (NSE) ticker symbols with standard suffix conventions (`.NS`, e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`).
2. **Historical Depth (2015 – Present):** Must cover at least a 10-year historical window to span our structural evaluation periods:
   - **Calibration Period:** 2015–2021
   - **Validation Period:** 2022–2024
   - **Blind-Test Period:** 2025–present
3. **Corporate Action Adjustments:** Must provide **Adjusted Close (`Adj Close`)** prices to account for stock splits, bonuses, and dividend adjustments, preventing artificial price gaps from corrupting technical indicator calculations (e.g., SMA, Bollinger Bands, ATR).
4. **API Limits & Rate Limit Resilience:** Must avoid severe rate limits, credit caps, or restrictive paywalls that impede wide grid search optimization and bulk historical backtesting.
5. **Zero-Lock-In / Environment Safety:** API keys (if needed by fallback providers) must strictly load via environment variables without hardcoded credentials.

---

## 3. Provider Evaluation & Comparison

| Feature / Criteria | `yfinance` | Alpha Vantage (Free) | Finnhub / IEX Cloud | Polygon.io (Free) |
|---|---|---|---|---|
| **NSE (`.NS`) Support** | **Full Support** | Partial / Fragile | Limited | Paid Tier Only |
| **History Back to 2015** | **Yes (Full daily)** | 5-year limit / Compact default | Varies / Restricted | 2-year limit |
| **Adjusted Close Data** | **Included (`Adj Close`)** | Premium Endpoint | Premium Endpoint | Adjusted available |
| **Rate Limit Constraints** | **High / Resilient** | Severe (5 calls/min) | Strict API credit budgets | 5 calls/min |
| **Cost / Licensing** | Open Source / Free | Freemium Paywall | Freemium Paywall | Freemium Paywall |

---

## 4. Key Rationale for Choosing `yfinance`

1. **Native `.NS` Symbol Coverage:** `yfinance` cleanly wraps Yahoo Finance endpoints which natively index all NSE equities (e.g., `RELIANCE.NS`, `TCS.NS`, `TATAMOTORS.NS`, `HDFCBANK.NS`).
2. **Seamless 10+ Year Historical Depth:** Supports fetching full daily OHLCV series starting from 2015 to the present date in a single vectorized DataFrame payload (`yfinance.Ticker.history(period="max")` or `start="2015-01-01"`).
3. **Canonical Adjusted Close:** Automatically provides split and dividend adjusted prices via `Adj Close`. In SignalForge, `Adj Close` serves as the primary price series for indicator computation and returns calculation to eliminate corporate-action distortion.
4. **No Restrictive Rate Limits:** Unlike free-tier REST APIs with 5-call/minute caps (which stall multi-stock parameter sweeps), `yfinance` handles batch requests efficiently and smoothly when paired with local caching.
5. **Pandas Vectorization Integration:** Data is directly delivered as a `pandas.DataFrame` indexed by UTC/localized `DatetimeIndex` with standardized column headers (`Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`), integrating directly into `strategy_engine/indicators`.

---

## 5. Implementation Strategy & Data Hygiene Rules

To adhere strictly to `blueprint.md` §2 and `PRINCIPLES.md` §2, the ingestion layer (`data/`) will implement the following rules:

### A. Persistent Local Caching
To minimize external network queries and guard against temporary endpoint disruptions, raw data pulled from `yfinance` is cached in local SQLite/PostgreSQL storage (`database/`). Re-running backtests reads from local persistence unless an explicit refresh is triggered.

### B. Cleaning Pipeline
- **Missing Values:** Isolated single-day missing bars are forward-filled only when defensible; windows with extended gaps are logged and flagged rather than blindly interpolated.
- **Invalid Rows:** Rows with zero or negative prices, or OHLC inconsistencies where `High < Low` or `Volume < 0`, are purged and recorded in ingestion logs.
- **Timestamp Normalization:** All dates are stored as timezone-naive dates (`YYYY-MM-DD`) at market close.

### C. Look-Ahead Bias Enforcement
Data ingestion feeds bars into strategy simulations chronologically. Indicators must only consume rows up to bar $T$. Signals produced at bar $T$'s close execute strictly at bar $T+1$'s open.

---

## 6. Fallback & Extension Architecture

If symbol coverage or provider connectivity issues arise for specific global assets in the future, the ingestion engine is decoupled behind an abstract base class `BaseDataProvider`. Swapping or supplementing `yfinance` with an alternative source (e.g., NSE Direct data, Alpha Vantage, or custom CSV feeds) will require zero modifications to `strategy_engine` or the API layer.
