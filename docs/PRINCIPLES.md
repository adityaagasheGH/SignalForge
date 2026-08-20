# SignalForge — Core Principles

**This file is a pre-flight check, not background reading.** Before writing or modifying any code that touches data, indicators, strategies, backtesting, scoring, regime detection, or news, re-read the relevant section below. If a proposed change would violate any rule here, stop and flag it rather than implementing it — even if it makes a metric look better.

Refer to `blueprint.md` for *what* to build. This file governs *how it must behave* regardless of what's being built.

---

## 1. The Calibration / Validation / Blind-Test Wall Is Absolute

- **Calibration period** (default 2015–2021): parameter optimization happens here, and only here.
- **Validation period** (default 2022–2024): strategy selection, scoring, and walk-forward evaluation happen here.
- **Blind test period** (default 2025–present): read exactly once, after strategy and parameters are already frozen. It never feeds back into any decision.

Concretely, this means:
- No function that computes an "optimal" parameter may ever receive blind-test rows as input.
- No strategy-ranking or strategy-selection code path may branch on blind-test performance.
- If a refactor makes it easier to accidentally pass the full dataset into an optimization routine (e.g. collapsing calibration/validation/test into one DataFrame passed everywhere), don't do the refactor that way — keep the boundary structurally enforced (e.g. distinct function signatures, distinct DataFrame objects, or an explicit date-range guard at the top of any optimization/scoring function).
- When in doubt about which period a given computation is allowed to see, treat it as calibration-only until proven otherwise.

## 2. No Look-Ahead Bias, Anywhere

This is the single most important correctness property of the whole system. Specific rules:

- **Execution timing:** a signal computed from day T's close executes at day T+1's open. Never at day T's close. This is not a stylistic choice — it is the difference between a real backtest and a fabricated one.
- **Rolling windows:** any rolling/expanding calculation (SMA, RSI, ATR, ADX, volatility, etc.) must be indexed so that the value at row T uses only rows ≤ T. When in doubt, explicitly `shift()` before computing, and write a unit test asserting the row-T value is unchanged when row T+1's data is mutated.
- **News timestamps:** an article is only "current" for a decision at time D if `article_timestamp <= D`. Filter this explicitly at the query layer, not just by convention.
- **Regime classification:** the regime label assigned to day T must be computable using only data available through day T.
- **Every new indicator, strategy, or scoring function must ship with a look-ahead test** that mutates future rows and asserts past outputs are unaffected. This is not optional coverage — treat it as part of the definition of "done" for that function, and part of `tests/test_lookahead.py`.

## 3. Never Fabricate Confidence

- If a strategy, regime, or fold has fewer trades/samples than the configured minimum threshold, its score/signal must be visibly flagged as low-confidence — never silently presented at full precision next to well-supported numbers.
- Historical news similarity: if fewer than the configured minimum number of comparable events exist, report **"Insufficient historical evidence"** and contribute nothing (or a heavily downweighted, clearly labeled contribution) to the final signal. Do not interpolate, estimate, or "fill in" a plausible-sounding number from a thin sample.
- The final signal's confidence score must reflect actual evidence agreement/disagreement (see §4) — it is not a cosmetic number tuned to look reasonable.
- Never claim a scoring weight vector, regime threshold, or optimization grid is "objectively optimal." These are documented design choices. Say so in the UI and in comments wherever a weight or threshold appears.

## 4. Conflicting Evidence Must Produce Conflict, Not a Forced Answer

When strategy evidence, consensus, regime, news, and historical-event evidence disagree in the final fusion:
- Detect the divergence explicitly (don't just let the weighted sum quietly average it away).
- Cap confidence and pull the output toward HOLD when evidence meaningfully conflicts — do not let one strong component silently dominate the presentation.
- The explanation engine must state the disagreement in plain language, not paper over it.

## 5. Strategies Are Rule-Based, Not Predictive Models

- "Model" in conversation and in code/comments means a trading strategy (rule-based `generate_signal`/`backtest` implementation), not a machine-learned predictor.
- Do not introduce ML price prediction (LSTM, XGBoost, Random Forest, or similar) into the core strategy library, scoring engine, or final signal engine. ML is an explicitly deferred future extension (`blueprint.md` §35) and stays out of the codebase until the rule-based system is complete and stable.
- Sentiment/embedding models (e.g. FinBERT, sentence-transformers) are acceptable as pretrained components inside the news engine — they are not the core prediction mechanism the platform is built around, and they must not be described or treated as if they were.

## 6. Every Score Must Be Explainable

- Any number shown to the user (strategy score, regime label, news score, final signal confidence) must be traceable back to its inputs and weights through the explanation engine or a drill-down view.
- No black-box scores. If a component can't currently be explained simply, that's a signal to simplify the component, not to hide the explanation.
- Weight vectors (scoring weights, fusion weights) are always configuration, never magic numbers buried in function bodies.

## 7. Realism Over Optimism in Backtests

- Report after-transaction-cost results as the headline number everywhere in the UI; before-cost is a secondary, clearly labeled view.
- Document real trading assumptions (execution price convention, fractional shares, overnight positions, position sizing) in `docs/trading_assumptions.md` and keep that file in sync with what the code actually does.
- Do not implement short-selling, leverage, or intraday-only execution unless explicitly requested and justified — the default is long-only with realistic frictions.

## 8. The Disclaimer Is Non-Negotiable

The disclaimer text below is displayed on any surface presenting a final signal, and is never removed, shortened to the point of losing meaning, or made conditional:

> "SignalForge provides analytical signals for research and educational purposes. It does not constitute financial advice and does not guarantee future market performance."

A BUY/SELL signal is never phrased in a way that implies guaranteed profit or certainty.
