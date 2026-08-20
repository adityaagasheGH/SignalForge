# SignalForge — Engineering Standards

Architecture and coding conventions. `blueprint.md` defines *what*, `PRINCIPLES.md` defines *the non-negotiable behavioral rules*, this file defines *how the code should be structured and written* so the project stays maintainable across a 4–5 week solo build.

---

## 1. Project Structure

Follow the structure in `blueprint.md` §31 unless there's a documented reason to deviate (write the reason in `docs/architecture_decisions.md` if you do):

```
signalforge/
  frontend/
  backend/
  strategy_engine/
    strategies/       # one file per strategy, implementing the common interface
    indicators/        # pure functions, no strategy-specific logic
    backtesting/       # the event-driven simulation engine
    optimization/      # grid search + walk-forward orchestration
    scoring/            # strategy scoring engine
    regime/              # regime detection
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

Do not create new top-level packages for a single small utility — put it in the most relevant existing package. Do not collapse `strategy_engine` submodules into one flat file "for now" — the separation is what keeps strategies, indicators, backtesting, optimization, and scoring independently testable.

## 2. The Strategy Interface Is a Contract

Every strategy implements the same interface (`generate_signal(data)`, `calculate_position(data)`, `backtest(data)`, `get_parameters()`). Concretely:

- New strategies subclass/implement a shared base (abstract base class or protocol), never duplicate the interface ad hoc.
- Strategies consume indicators from `strategy_engine/indicators/` — a strategy file must never recompute an SMA/RSI/etc. inline. If a strategy needs an indicator that doesn't exist yet, add it to the indicator layer first, then use it.
- `get_parameters()` returns the strategy's current parameter set in a form the optimization layer can iterate over (e.g. a dict of name → value, with a companion "grid definition" describing the sweep range) — don't hardcode a single parameter set inside the strategy class.
- Adding a 13th strategy later should require: one new file in `strategies/`, one new grid entry in config, zero changes to the backtesting engine, scoring engine, or API layer. If adding a strategy requires touching those other layers, the interface has leaked and needs fixing before proceeding.

## 3. Indicators Are Pure and Reusable

- Every indicator function takes a price DataFrame + parameters and returns an aligned Series/DataFrame. No hidden state, no I/O, no side effects.
- Each indicator documents its warm-up period (how many leading NaN rows to expect) in its docstring, and calling code respects that rather than treating NaN as a real value.
- Indicators are unit-tested against hand-computed or well-known reference values on a small fixed dataset, independent of any strategy.

## 4. Backtesting Engine Is the Single Source of Truth for Simulation

- All strategies run through the same event-driven backtest engine — no strategy implements its own ad hoc simulation loop.
- The engine owns: execution timing (T+1 open), position/cash/portfolio bookkeeping, transaction cost application, trade logging.
- Trade logs (entry/exit price & date, size, gross/net P&L) are the canonical artifact — performance metrics (§8 in blueprint) are computed *from* the trade log and equity curve, not recomputed independently in multiple places with potentially inconsistent logic.

## 5. Configuration, Not Hardcoding

The following must live in config (YAML/JSON/DB), never as literals buried in code:

- Strategy parameter grids.
- Calibration/validation/blind-test date boundaries.
- Scoring weight vectors (strategy score components, final signal fusion weights).
- Regime classification thresholds (ADX cutoffs, volatility percentile cutoffs, hysteresis window length).
- Minimum-sample thresholds (for strategy scores, regime cells, and historical news similarity).
- Transaction cost defaults (brokerage, fees, slippage).
- API keys and credentials — environment variables only, never committed, never hardcoded even as "temporary" placeholders.

If you find yourself writing `0.3` or `25` or a date literal directly inside a scoring/regime/optimization function, stop and move it to config first.

## 6. Database Conventions

- Follow the entity model in `blueprint.md` §25. Every table must trace back to a section of the blueprint — don't add speculative tables.
- `BacktestRuns` always stores a config snapshot (parameters, date range, run type) alongside results, so any historical result is reproducible without guessing what config produced it.
- `StrategyMetrics` and `FinalSignals` are append-only / versioned by run — never overwrite a prior signal or backtest result in place; a new run produces a new row, so historical signals remain reviewable.
- Migrations are used for schema changes (not ad hoc manual DB edits), even in a solo project — this is what keeps `docs/` and reality in sync.

## 7. API Conventions

- Pydantic models for every request/response — no raw dicts crossing the API boundary.
- Consistent error shape (status code + machine-readable error code + human message) across all endpoints.
- Endpoints are additive to the list in `blueprint.md` §26 as needed, but any new endpoint should be added back into that section so the API surface stays documented in one place.
- Long-running operations (a fresh backtest run, a full parameter sweep) return a job/run ID immediately and are polled or fetched via `GET /backtest/{id}`, rather than blocking the request — this matters even for a solo project once the Strategy Lab lets users trigger sweeps on demand.

## 8. Frontend Conventions

- React + TypeScript, function components, Tailwind for styling.
- Chart components are isolated and reusable: one component per chart type (candlestick+overlays, equity curve, leaderboard table), fed by typed props derived from the API response shapes — no chart component reaches into a global store or makes its own fetch calls.
- Strategy Lab and Dashboard share underlying data-fetching hooks where the same data (e.g. a stock's price history) is needed on both — don't duplicate fetch logic.
- Every screen that shows a signal, score, or regime label also surfaces (or links to) the explanation for it — a numeric-only view without an explanation affordance is incomplete per `PRINCIPLES.md` §6.

## 9. Testing Conventions

- Test files mirror source structure (`tests/strategy_engine/indicators/test_rsi.py` next to `strategy_engine/indicators/rsi.py`, etc.).
- `tests/test_lookahead.py` is a standing, growing suite — every new indicator, strategy, or scoring function adds a corresponding look-ahead assertion here, not just a generic correctness test elsewhere.
- Fixtures use small, fixed, hand-inspectable datasets for unit tests (not the full 2015–present history) so failures are easy to diagnose; the full historical dataset is reserved for integration-level backtest tests.

## 10. Code Review Checklist (self-applied before considering any component "done")

- [ ] Does this touch calibration/validation/blind-test data? If so, is the boundary structurally enforced, not just convention?
- [ ] Does this compute anything from a rolling/expanding window? If so, is there a look-ahead test for it?
- [ ] Are any weights, thresholds, or date ranges hardcoded that should be config?
- [ ] Does this produce a score or signal? If so, is there a path to explain it in the UI?
- [ ] Does this add a new top-level module/table/endpoint? If so, is it reflected back into `blueprint.md`?
- [ ] Does this strategy/metric have a minimum-sample guardrail before it's trusted?
