# SignalForge — Blueprint

**Multi-Strategy Quantitative Trading Intelligence Platform**

This is the master reference document for SignalForge. It elaborates every phase of the build in full — concept, methodology, architecture, roadmap. Any implementation decision that isn't covered here should be resolved by checking `PRINCIPLES.md`, `ENGINEERING_STANDARDS.md`, and `WORKFLOW.md` first, and should then be added back into this file so it stays the single source of truth.

---

## 0. What SignalForge Is (and Isn't)

SignalForge is a **trading-strategy evaluation and selection engine**, not a price-prediction model.

Given a stock symbol (e.g. `RELIANCE.NS`), the system:

1. Pulls historical market data for that stock.
2. Backtests a library of ~12 rule-based trading strategies against it, independently.
3. Scores and ranks those strategies on risk-adjusted, robustness, and consistency criteria — both overall and conditioned on the *current* market regime.
4. Determines what signal the best-suited strategy (or ensemble) is currently producing.
5. Cross-checks that signal against current news sentiment and how the stock historically reacted to similar news events.
6. Fuses all of the above into a single BUY / HOLD / SELL research signal with a confidence score and a plain-language explanation.

**Terminology note:** "models" = trading strategies, not ML models. This is not an XGBoost/LSTM/Random-Forest price-prediction project. ML is an optional *future* extension (a meta-model that predicts which strategy will outperform), never the core.

**The central research question the whole platform exists to answer:**

> *Which trading strategy is most suitable for this stock, under its historical and current market conditions — and does that strategy continue to work on data it has never seen?*

Everything downstream (dashboard, charts, news engine) is in service of answering that question honestly and legibly. If a feature doesn't help answer it, it's polish, not core.

---

## 1. Strategy Library (12 Strategies)

Each strategy implements a common interface: `generate_signal(data)`, `calculate_position(data)`, `backtest(data)`, `get_parameters()`. Parameters are never hardcoded as "optimal" — they are configurable and swept during optimization.

Strategies are grouped into four philosophies so the platform can demonstrate that different approaches dominate under different regimes — this diversity is the whole point, not an afterthought.

### Mean Reversion
1. **Bollinger Band Mean Reversion** — BUY when price falls below the lower band, exit toward the middle band or a configurable exit rule. Configurable: lookback period, std-dev multiplier, entry/exit thresholds.
2. **RSI Mean Reversion** — BUY when RSI < oversold threshold (default 30), SELL when RSI > overbought threshold (default 70). Thresholds configurable.
3. **Stochastic Oscillator** — BUY when %K < 20, SELL when %K > 80. Faster/noisier than RSI; useful contrast for robustness checks. Configurable lookback and thresholds.

### Trend Following
4. **SMA Crossover** — BUY when short SMA crosses above long SMA (default 50/200), SELL on reverse cross. Multiple period pairs tested, not just 50/200.
5. **EMA Crossover** — Same logic as SMA, exponential weighting (default 20/50). Tests sensitivity to responsiveness vs SMA.
6. **MACD Crossover** — BUY when MACD line crosses above signal line, SELL on reverse. Configurable fast/slow/signal periods (e.g. 12/26/9, 8/21/5, 20/50/9).
7. **ADX Trend Following** — Confirms trend strength (ADX > configurable threshold, default 25) combined with directional movement (+DI/-DI). Doubles as a regime-confirmation signal, not just a standalone strategy.

### Breakout
8. **Donchian Breakout** — BUY when close breaks above the highest close of the previous N days, exit on breaking the lower channel or another configurable exit. N swept across {10, 20, 50, 100}.
9. **ATR / Volatility Breakout** — Combines ATR expansion (volatility regime change) with a price breakout trigger. Implementation choice must be documented and justified in code comments.
10. **Keltner Channel Breakout** *(recommended 12th strategy, replacing a generic placeholder)* — ATR-based bands (not stdev-based like Bollinger), so it behaves differently in gappy/fat-tailed conditions. Genuinely distinct math from Donchian and Bollinger, not padding.

### Momentum / Hybrid
11. **N-Day Rate-of-Change Momentum** — `ROC = (Price_t / Price_t-N) - 1`. BUY when ROC exceeds a configurable positive threshold, SELL below a configurable negative threshold. N swept across {5, 10, 20, 60}.
12. **Moving Average Pullback** — Establishes trend via price > long-term MA, then looks for a pullback toward a short-term MA followed by bullish confirmation. Hybrid of trend-following and mean-reversion logic — trades *with* the trend but *against* short-term noise.

**Strategy diversity matrix** (for the scoring/regime engine to reason over): each strategy is tagged with its philosophy (Mean Reversion / Trend / Breakout / Momentum), so regime-aware selection can be validated against the expectation that mean-reversion strategies outperform in Sideways regimes and trend/breakout strategies outperform in trending regimes.

---

## 2. Historical Data

- **Universe:** Indian equities / NSE symbols initially (e.g. `RELIANCE.NS`).
- **Coverage:** 2015 → present, where available. Document any symbol with a shorter history rather than silently truncating.
- **Provider:** selected on availability, reliability, historical depth, API limits, and ease of deployment (see `WORKFLOW.md` for the evaluation checklist before committing). No hardcoded credentials — all API keys via environment variables, never committed.
- **Fields retrieved:** Date, Open, High, Low, Close, Adjusted Close (if available), Volume.

### Data Cleaning
- Missing values: forward-fill only where defensible (e.g. isolated single-day gaps); otherwise flag and exclude the affected window from backtests rather than interpolate blindly.
- Duplicate timestamps: deduplicated deterministically (keep the record from the canonical/primary source).
- Invalid rows (zero/negative prices, OHLC inconsistencies where High < Low, etc.): dropped and logged.
- Corporate actions (splits, bonuses, dividends): adjusted-close preferred as the canonical price series for indicator calculation; document which adjustment convention is used.

### No Look-Ahead Guarantee
All indicators must only use information available as of the bar they're computed on. This is enforced structurally (rolling windows computed with `shift`/lag conventions that cannot see forward) and verified by a dedicated test suite (see §17, §37 in the original spec / `PRINCIPLES.md` §2).

---

## 3. Technical Indicator Engine

A single reusable indicator layer that all strategies consume — strategies never recompute their own indicators from raw OHLCV.

Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, ROC, realized volatility, momentum, rolling returns, Keltner Channels.

Each indicator function:
- Takes a price DataFrame and parameters, returns a Series/DataFrame aligned to the same index.
- Is pure (no hidden state, no side effects) so it can be unit-tested independently of any strategy.
- Documents its warm-up period (how many leading rows will be NaN) so backtests correctly exclude the warm-up window rather than treating NaN as a signal.

---

## 4. Parameter Optimization

Strategies are never evaluated on a single arbitrary parameter set. Each strategy has a small, documented grid (examples below; final grids live in `strategy_configs/*.yaml` or similar, not hardcoded in strategy classes).

| Strategy | Example grid |
|---|---|
| Bollinger | lookback ∈ {10,20,30,50} × std-dev ∈ {1.5,2.0,2.5,3.0} |
| MACD | (fast,slow,signal) ∈ {(8,21,5), (12,26,9), (20,50,9)} |
| Donchian | N ∈ {10,20,50,100} |
| ROC | N ∈ {5,10,20,60} |

**Rule:** grid search runs *only* on the calibration period. The winning configuration per strategy is chosen by calibration-period score alone and is then frozen before touching validation or test data. All grid results (not just the winner) are persisted — this powers a parameter-sensitivity view in the Strategy Lab and is itself evidence against overfitting (a winner surrounded by similarly-good neighbors is more trustworthy than an isolated spike).

---

## 5. Calibration / Validation / Blind Test Periods

Because these are rule-based strategies, not ML models, the terminology is:

- **Calibration** (default 2015–2021): parameter optimization happens here, and only here.
- **Validation** (default 2022–2024): strategy selection and scoring happen here, via walk-forward evaluation (§6).
- **Out-of-sample blind test** (default 2025–present): touched exactly once, after the strategy and parameters are already frozen. Never used to influence strategy selection or parameter choice. This is the single most important invariant in the whole system.

All three date ranges are configurable (not hardcoded), but the *separation* between them is a hard invariant enforced in code (see `PRINCIPLES.md` §1).

---

## 6. Walk-Forward Backtesting

Anchored (expanding-window) walk-forward validation across the calibration+validation range:

```
Calibrate 2015–2018 → Validate 2019
Calibrate 2015–2019 → Validate 2020
Calibrate 2015–2020 → Validate 2021
Calibrate 2015–2021 → Validate 2022
Calibrate 2015–2022 → Validate 2023
Calibrate 2015–2023 → Validate 2024
```

Each fold re-runs parameter optimization on that fold's calibration window, then scores the frozen parameters on that fold's validation year only. A strategy's validation score is the **aggregate across all folds** (mean and variance), not a single static backtest over 2022–2024. Variance across folds feeds directly into the "stability" component of the strategy score (§9) — a strategy that only worked in one fold is not robust, even if its blended average looks fine.

The 2025–present blind test window is never part of this walk-forward loop.

---

## 7. Backtesting Engine

- **Simulation style:** event-driven, bar-by-bar (not purely vectorized) so entry/exit/cash/position state is explicit and auditable at every step — vectorized-only backtests make it easy to accidentally leak lookahead through careless indexing.
- **Execution timing (hard rule):** a signal generated from day T's close executes at day T+1's open. Never execute at the same close the signal was computed from.
- **Tracked state:** entry, exit, position size, cash, portfolio value, transaction costs, slippage.
- **Trading style:** long-only initially. No short-selling unless later justified with a clear technical rationale.
- **Configurable:** transaction costs, initial capital, position sizing method (§13).
- Every closed trade is persisted with entry/exit price & date, size, gross P&L, costs, net P&L — this is the raw material for every performance metric and for the entry/exit markers on the price chart.

---

## 8. Performance Metrics

Strategies are never ranked on total profit alone. Computed per strategy, per backtest run:

**Returns:** total return, CAGR
**Risk:** volatility (annualized), maximum drawdown
**Risk-adjusted:** Sharpe ratio, Sortino ratio
**Trading statistics:** win rate, profit factor, number of trades, average trade return, average winning trade, average losing trade
**Robustness:** performance by calendar year, performance by market regime, performance across walk-forward validation folds (mean + std dev of fold scores)

Metrics are computed both **before costs** and **after costs**; after-cost is treated as the realistic, headline number everywhere in the UI, with before-cost available as a secondary view.

---

## 9. Strategy Scoring Engine

A normalized composite score (0–100) per strategy, default weighting (fully configurable, stored in DB/config, never assumed optimal):

| Component | Default weight |
|---|---|
| Risk-adjusted return | 30% |
| CAGR | 20% |
| Max drawdown (penalized) | 15% |
| Sharpe ratio | 15% |
| Win rate | 10% |
| Stability across folds | 10% |

Each component is normalized (z-score or min-max) across the strategy set being compared so weights are meaningfully comparable. The score and its weight vector are always shown together in the UI — never a bare number with no way to audit how it was produced.

**Minimum-sample guardrail:** a strategy/fold/regime combination with too few trades (threshold configurable, suggested ≥15–20 trades for a score to be trusted) is flagged as low-confidence rather than presented with false precision alongside statistically well-supported scores.

---

## 10. Strategy Ranking

The Strategy Lab and dashboard display a ranked leaderboard (score, return, Sharpe, drawdown, win rate, trade count) with drill-down: clicking any strategy reveals the metric breakdown and weight contributions that produced its score, plus its equity curve versus buy-and-hold.

---

## 11. Market Regime Detection

Regimes: Strong Bull Trend, Weak Bull Trend, Sideways, Weak Bear Trend, Strong Bear Trend, each optionally cross-tagged with High Volatility / Low Volatility.

**Methodology (documented, not arbitrary):**
- **Trend direction:** price relative to 50-day and 200-day SMA; slope of the 200-day SMA.
- **Trend strength:** ADX (>25 = trending, <20 = sideways/weak).
- **Volatility overlay:** current ATR or realized volatility vs its own trailing percentile (e.g. above 70th percentile = High Vol).

**Stability rule:** regime classification uses hysteresis — a regime change requires the new classification to persist for a configurable number of days (default 5–10) before the system switches its labeled regime. This prevents daily flapping that would make regime-conditioned scores noisy and untrustworthy. Exact thresholds live in `docs/regime_methodology.md` and are treated as a documented, adjustable convention, not ground truth.

---

## 12. Regime-Aware Strategy Selection

The single most differentiating feature of the platform. For every strategy, two scores are computed and stored:

- **Overall score** — blended across all regimes/folds.
- **Per-regime score** — score restricted to the subset of days/folds/trades that occurred under each regime.

Example:

```
Donchian Breakout      Overall: 87   Bull: 94   Sideways: 51   Bear: 64
Bollinger Reversion    Overall: 70   Bull: 58   Sideways: 92   Bear: 74
```

When selecting "the best strategy right now," the engine weighs the per-regime score for the currently detected regime more heavily than the overall score. The exact blend (e.g. 70% current-regime score / 30% overall score, configurable) and the reasoning are surfaced transparently in the UI — this is not a black box.

---

## 13. Position Sizing

Kept deliberately simple in v1 — the research focus is strategy selection, not portfolio construction:
- Fixed percentage allocation, or
- Equal capital allocation across positions.

No portfolio optimization, no Kelly sizing, no volatility-targeted sizing in v1. This is explicitly logged as a "future extension," not silently omitted.

---

## 14. Current Strategy Signal

After the best-suited strategy (or ensemble) is selected for the current regime, it is run against the most recent data to produce today's signal:

```
Selected Strategy: Donchian Breakout
Current Signal: BUY
Reason: Price broke above the 50-day high.
Signal Strength: 82/100
```

Signal strength is derived from how far the triggering condition exceeded its threshold (e.g. how far above the breakout level, how deep into oversold RSI territory), normalized 0–100.

---

## 15. Strategy Consensus

All ~12 strategies (run with their own optimized parameters) are evaluated on current data. Consensus is a simple tally:

```
BUY: 6   HOLD: 3   SELL: 1  →  "Moderately Bullish"
```

Consensus is **supporting evidence only** — it feeds into the final fusion with its own (smaller) weight and must never silently override the regime-selected best strategy's signal.

---

## 16. News Intelligence Engine

Pipeline stages, run for each tracked stock:

1. **Retrieval** — pull recent company-related news from a news API, keyed by ticker/company name.
2. **Deduplication** — hash/near-duplicate detection on headline + source + timestamp window.
3. **Company relevance detection** — ticker/company-name match plus keyword filtering to drop syndicated noise.
4. **Sentiment analysis** — pretrained finance-tuned classifier (e.g. FinBERT), not a custom-trained model.
5. **Event classification** — fixed taxonomy: earnings, M&A, regulatory, management change, guidance, litigation, macro/sector, other. Keyword-rule-based initially; can be upgraded to a classifier later.
6. **Recency weighting** — exponential decay so older articles matter less without being discarded outright.
7. **Importance scoring** — function of source credibility, event category, and sentiment magnitude; weights configurable and documented.

Output per article: sentiment, relevance %, impact (Low/Med/High), event category, timestamp, and an aggregate News Score (0–100) for the stock.

---

## 17. Historical News Similarity

For a current important article:

1. Embed it (sentence-transformer, cosine similarity).
2. Retrieve top-K similar historical articles for the *same stock* first; fall back to *same sector* only if too few same-stock matches exist.
3. Look up each matched event's subsequent 1-day / 5-day / 20-day returns (precomputed and cached when articles are first ingested).
4. Report the average return at each horizon, the number of comparable events, and a directional label (Bullish/Bearish/Neutral).

**Hard rule:** if the number of comparable historical events is below a configurable minimum (suggested ≥8–10), the system reports **"Insufficient historical evidence"** and contributes nothing (or a heavily downweighted, clearly-labeled contribution) to the final signal. Never fabricate or interpolate a result from a thin sample.

*(Scoping note: this module is explicitly flagged in `WORKFLOW.md` as best-effort / stretch for the timeline — see §21.)*

---

## 18. Final Signal Engine

Combines all evidence into one score, default weighting (configurable, never claimed objectively optimal):

| Component | Default weight |
|---|---|
| Strategy / backtest evidence (score of selected strategy) | 40% |
| Current strategy signal | 20% |
| News intelligence | 20% |
| Historical similar events | 10% |
| Strategy consensus | 10% |

**Two-framing output (required).** The final signal is never collapsed into a single "trust this" verdict. It always presents two framings side by side — a risk-adjusted/active framing and a passive/Buy-&-Hold framing — so the user applies their own risk tolerance rather than having one applied for them (see `docs/architecture_decisions.md`). The output always includes, in this order:

**1. Best risk-adjusted strategy** — the top-ranked *active* strategy (excluding Buy & Hold). Shows its name, composite score (0–100), current BUY/HOLD/SELL signal, confidence %, and — as the headline metrics justifying the "risk-adjusted" framing — its Sharpe ratio and maximum drawdown.

**2. Buy & Hold comparison** — shown *every time*, never omitted or hidden based on which side performed better (the point is transparency, not favoring one side). Shows Buy & Hold's total return over the same period, the best active strategy's total return for direct comparison, and Buy & Hold's actual rank in the full leaderboard of all 13 candidates (the 12 strategies plus Buy & Hold).

**3. Recommendation** — two sentences, one per framing, filled from the actual numbers:
- "If you want disciplined, risk-managed exposure with defined entries/exits → *[best active strategy]* is today's best-supported active choice."
- "If you're optimizing for raw long-term return and can tolerate deep drawdowns → Buy & Hold has *[outperformed / underperformed]* the best active strategy tested on this stock, historically." — the outperformed/underperformed direction is derived from the real returns, never hardcoded.

Output:

```
① BEST RISK-ADJUSTED STRATEGY
   Donchian Breakout   Score: 78/100   SIGNAL: BUY   Confidence: 76%
   Sharpe: 1.42        Max Drawdown: -18%

② BUY & HOLD COMPARISON
   Buy & Hold total return:       +64%
   Best active total return:      +51%   (Donchian Breakout)
   Buy & Hold leaderboard rank:   3 of 13

③ RECOMMENDATION
   • Risk-managed: Donchian Breakout is today's best-supported active choice.
   • Raw return: Buy & Hold has OUTPERFORMED the best active strategy tested
     on this stock, historically.
```

The composite score, its confidence, and the divergence check below all still apply to the active-strategy framing (block ①); Buy & Hold in block ② is reported on total return and leaderboard rank, not run through the news/consensus fusion.

**Conflict handling:** before the weighted sum is trusted at face value, a divergence check runs — if strategy evidence and news evidence point in materially opposite directions, confidence is capped and the output is pulled toward HOLD regardless of the raw weighted number. Example from spec: strategy BUY + consensus BUY but news STRONGLY NEGATIVE + historical news NEGATIVE + regime SIDEWAYS → final signal HOLD at reduced confidence, not a blind BUY.

---

## 19. Explanation Engine

Every final signal is accompanied by a templated (not free-form-generated) explanation populated from the actual computed scores, so it is accurate and testable, e.g.:

```
WHY?
• Donchian Breakout is currently the highest-ranked strategy for this stock.
• The stock has entered a bullish breakout (price > 50-day high).
• This strategy performs strongly in the current (Bull) market regime.
• 6/10 strategies currently indicate BUY.
• However, recent company news is negative.
• Similar historical news events produced mixed returns.

Conclusion: Moderately bullish, but uncertainty remains high.
```

---

## 20. Backtest Comparison Page

Lets a user compare any subset of strategies on the same stock and date range, always including **Buy & Hold** as a baseline. Displays: equity curves, returns, drawdowns, Sharpe, Sortino, CAGR, win rate, trade count — overlaid or tabbed for direct comparison.

---

## 21. Strategy Lab

A dedicated page where the user can:
- Select a stock, strategies, parameters, and date range.
- Run backtests on demand.
- Compare results in a leaderboard (Strategy / Return / Sharpe / Drawdown / Win Rate / Trades).
- Drill into the parameter-sensitivity view from §4.

This is the technical core of the product from a demo/portfolio standpoint — it's where the "does this actually work, and how do we know" question is answered visibly.

---

## 22. Main Dashboard

Search a symbol → shows:

- **Stock Overview** — current price, daily change, trend, volatility, market regime.
- **Strategy Analysis** — best strategy, its score, its current signal, and the consensus tally.
- **News** — recent headlines, sentiment, impact.
- **Historical Events** — similar past news and how the stock reacted (or "insufficient evidence").
- **Final Signal** — the two-framing structure from §18 rather than a single BUY/HOLD/SELL badge: the "Best risk-adjusted strategy" block (name, score, signal, confidence, Sharpe, max drawdown), the "Buy & Hold comparison" block (both total returns and Buy & Hold's leaderboard rank, always shown), and the two-sentence "Recommendation", alongside the plain-language "Why?" explanation.

---

## 23. Technical Charts

- Candlestick/price chart with volume.
- Overlay toggles: moving averages, Bollinger Bands.
- Sub-charts: RSI, MACD.
- Strategy entry/exit markers and buy/sell signal annotations, tied to the actual trade log from the backtest engine (§7), not illustrative placeholders.
- User controls which indicators/overlays are visible.

---

## 24. Technology Stack

- **Frontend:** React + TypeScript, Tailwind CSS.
- **Charts:** `lightweight-charts` (TradingView) for the candlestick/price chart; Recharts for standard line/bar indicator and metric charts.
- **Backend:** Python + FastAPI.
- **Data/Backtesting:** Python + Pandas + NumPy.
- **Indicators:** `ta` or `pandas-ta` for standard indicators; hand-rolled implementations for 2–3 core indicators where transparency/control matters.
- **Database:** PostgreSQL.
- **Caching:** Redis only if a real bottleneck is identified — not added speculatively.
- **Deployment:** Frontend on Vercel (or equivalent); backend on Railway/Render (or equivalent) with Docker where it genuinely simplifies things. Infrastructure is not over-engineered for a solo 4–5 week project.

---

## 25. Database Design

Core entities (exact schema detail lives alongside the code, this is the conceptual model):

- `Stocks` — symbol, name, sector, exchange, listing metadata.
- `HistoricalPrices` — OHLCV per stock per date, adjusted-close, source, ingestion timestamp.
- `Strategies` — strategy identifier, philosophy tag (mean reversion / trend / breakout / momentum), interface version.
- `StrategyParameters` — parameter sets per strategy (grid entries and the frozen "winning" set per calibration window).
- `BacktestRuns` — stock, strategy, parameter set, date range, run type (calibration/validation/blind/ad-hoc), timestamp, config snapshot.
- `StrategyMetrics` — all §8 metrics per backtest run, tagged by regime and by walk-forward fold where applicable.
- `StrategySignals` — current signal per strategy per stock (BUY/HOLD/SELL, strength, timestamp).
- `NewsArticles` — raw ingested articles, dedup hash, source, timestamp, linked stock(s).
- `NewsAnalysis` — sentiment, relevance, importance, event category per article.
- `HistoricalEvents` — precomputed embeddings + subsequent-return outcomes per article, for similarity lookup.
- `FinalSignals` — the fused BUY/HOLD/SELL output, confidence, component scores, explanation text, timestamp — persisted so historical signals can be reviewed later, not just the current one.

No tables are added "just in case" — every table above is directly load-bearing for a section in this document.

---

## 26. API Design

```
GET  /stocks/search
GET  /stocks/{symbol}
GET  /stocks/{symbol}/history
GET  /stocks/{symbol}/strategies
POST /backtest
GET  /backtest/{id}
GET  /stocks/{symbol}/best-strategy
GET  /stocks/{symbol}/regime
GET  /stocks/{symbol}/signal
GET  /stocks/{symbol}/news
GET  /stocks/{symbol}/historical-events
POST /analysis
```

All endpoints use request/response validation (Pydantic models), consistent error shapes, and appropriate HTTP status codes. `/stocks/{symbol}/regime` is an addition to the original endpoint list — the regime is a first-class output, not something buried inside another response.

---

## 27. Performance & Caching

- Backtests are not re-run from scratch on every dashboard visit. Completed `BacktestRuns` and their `StrategyMetrics` are persisted and reused.
- Historical price data and indicator calculations are cached/persisted per stock rather than recomputed per request.
- The user can explicitly request a fresh analysis (e.g. a "Refresh Analysis" action), which invalidates the relevant cached artifacts.

---

## 28. No Data Leakage (system-wide invariant)

Future information must never influence: indicators, parameter selection, strategy ranking, market regime classification, or news-event analysis. Specific failure modes to guard against explicitly:

- Executing at the same close a signal was generated from.
- News articles whose timestamp is after the simulated decision time being included in "current" news for that decision.
- Rolling-window calculations that accidentally include the current/future bar due to off-by-one indexing.
- Parameter optimization touching validation or blind-test data.
- Strategy selection being influenced by blind-test performance.

All assumptions here are documented in `docs/lookahead_assumptions.md`, and enforced by tests (§30 below, and `PRINCIPLES.md` §2).

---

## 29. Transaction Costs & Realistic Trading Assumptions

- Configurable brokerage, fees, slippage, with reasonable NSE-appropriate defaults.
- Results always distinguish **before-cost** vs **after-cost**; after-cost is the headline number.
- Documented explicitly (in `docs/trading_assumptions.md`): execution timing (T+1 open), entry/exit price convention, fractional shares (not allowed — whole shares only, v1), overnight positions (allowed — no intraday-only constraint in v1), position sizing method in use.

---

## 30. Testing Strategy

Test coverage required for: indicators, strategies, signal generation, backtesting engine, transaction-cost application, strategy scoring, market regime classification, news scoring.

**Highest priority: a dedicated look-ahead bias test suite** (`tests/test_lookahead.py`) — inject a known future shock into historical data and assert that signals computed *before* that point in time are unchanged. This is treated as a release-blocking test category, not optional coverage.

---

## 31. Project Structure

```
signalforge/
  frontend/
  backend/
  strategy_engine/
    strategies/
    indicators/
    backtesting/
    optimization/
    scoring/
    regime/
  news_engine/
    ingestion/
    sentiment/
    events/
    similarity/
  data/
  database/
  tests/
  scripts/
  docs/
  README.md
```

Deviations from this structure are allowed if a better rationale exists, but must be explained in `docs/` when made.

---

## 32. Development Roadmap (4–5 Weeks)

**Week 1 — Data + Foundation:** repo scaffold, frontend shell, backend shell, database schema, stock search, historical data ingestion, data cleaning pipeline, indicator engine, basic price chart.

**Week 2 — Strategy Engine:** all 12 strategies behind the common interface, signal generation, parameter system, event-driven backtesting engine with transaction costs, unit tests per strategy.

**Week 3 — Strategy Evaluation (core deliverable):** parameter optimization, walk-forward validation, performance metrics, regime detection, regime-aware strategy scoring, Strategy Lab UI, buy-and-hold comparison. *If time runs short anywhere in the project, everything after this week compresses first — this week does not.*

**Week 4 — News Intelligence + Final Signal:** news ingestion, sentiment, event classification, final signal fusion engine with conflict handling, explanation engine, dashboard integration. Historical news-similarity is attempted here as best-effort (see `WORKFLOW.md` §on scope management) and ships with a graceful "insufficient evidence" fallback if it doesn't fully mature.

**Week 5 — Polish + Deployment:** lookahead test suite, chart polish (entry/exit markers, indicator toggles), caching/persistence tuning, Docker, deployment, methodology docs, demo mode, final README, screenshots.

---

## 33. Biggest Technical Risks

1. Silent look-ahead bias in rolling windows, same-close execution, or parameter optimization peeking at future data.
2. Thin-sample overfitting compounding across regime × fold × strategy cells — mitigated by minimum-sample guardrails everywhere (§9, §17).
3. News-similarity module scope creep — mitigated by hard time-boxing and a documented graceful-degradation path.
4. Overall time budget across 30+ interconnected sections in 4–5 weeks — mitigated by the roadmap's explicit compression order (Week 3 is never cut; Week 5 polish is cut first if needed).

---

## 34. Financial Disclaimer

Displayed prominently in the UI (footer and/or final-signal card):

> "SignalForge provides analytical signals for research and educational purposes. It does not constitute financial advice and does not guarantee future market performance."

A BUY signal is never presented as a guarantee of profit. Uncertainty (confidence %, low-sample flags, conflicting-evidence flags) is always visible alongside the signal, never hidden to make the output look more decisive than the evidence supports.

---

## 35. Future ML Extension (explicitly out of scope for v1)

A future meta-model could learn, from: stock features + market regime + strategy performance history + volatility + news sentiment + strategy signals → the probability that each strategy will outperform going forward. This is documented here as a roadmap item only. It is not built until the rule-based system above is complete, tested, and stable.
