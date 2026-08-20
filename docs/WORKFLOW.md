# SignalForge — Development Workflow

Process rules for how work on this project should be approached, sequenced, and scoped. `blueprint.md` is the *what*, `PRINCIPLES.md` is the *behavioral guarantees*, `ENGINEERING_STANDARDS.md` is the *code shape*. This file is the *how we work*.

---

## 1. Before Writing Any Code

1. Re-read the relevant section(s) of `blueprint.md` for the feature being built.
2. Re-read `PRINCIPLES.md` in full if the feature touches data, indicators, strategies, backtesting, optimization, scoring, regime detection, or news — i.e. almost everything except pure UI polish.
3. Check `ENGINEERING_STANDARDS.md` for the relevant module's conventions.
4. If the feature isn't clearly covered by any of the three documents, propose an approach, get it confirmed, and then add it back into `blueprint.md` (or the appropriate principles file) so the docs stay authoritative and complete — undocumented decisions are the fastest way this project drifts from its own design.

## 2. Work Incrementally

- Implement one component at a time, in the order laid out in the Week 1–5 roadmap (`blueprint.md` §32), not opportunistically across weeks.
- Before changing existing code, inspect it first — understand what's there and why before modifying it. Don't overwrite working code unless there's a clear reason, and state that reason.
- After implementing a component, test it (per `ENGINEERING_STANDARDS.md` §9) before moving to the next one. Do not stack multiple untested components on top of each other.
- Explain non-obvious decisions inline (code comments) and, for anything architecturally significant, in `docs/architecture_decisions.md`.

## 3. Don't Front-Load Implementation

Do not dump an entire phase (e.g. all 12 strategies, or the whole news engine) into one pass without checkpoints. Prefer: build one strategy end-to-end (indicator → signal → backtest → test) and confirm it's correct before replicating the pattern across the remaining eleven. Mistakes caught in strategy #1 are cheap; the same mistake replicated across all twelve is expensive to unwind.

## 4. Scope Discipline

The project has a hard 4–5 week budget. When time pressure appears, compress in this order — **never reverse it**:

1. **Never compress:** Week 1 (data foundation) and Week 3 (strategy evaluation — optimization, walk-forward validation, regime-aware scoring). This is the technical core the whole thesis depends on; a working system without it is not SignalForge, it's a strategy backtester.
2. **Compress first:** historical news similarity (`blueprint.md` §17) — ship with the "insufficient evidence" fallback active and the UI honest about it, rather than rushing a half-working similarity engine into the final signal.
3. **Compress second:** chart polish, additional dashboard visual flourishes, demo mode.
4. **Compress last, and only if truly necessary:** deployment automation (Docker, CI) — a documented manual deployment process is an acceptable fallback; a broken or absent core scoring engine is not.

If a cut is made, document what was cut and why in `README.md` under a "Known Limitations / Scope Notes" section — don't let a cut feature silently disappear from the story of the project.

## 5. Data Provider Selection (do this before Week 1 coding starts)

Before committing to a market-data provider, verify concretely (not from memory):
- Actual historical depth available for NSE `.NS` symbols back to 2015.
- Rate limits relative to the number of symbols/backtests the project will realistically run.
- Whether adjusted-close is provided natively or must be derived.
- Licensing/ToS compatibility with a public portfolio project.

Record the decision and the verification findings in `docs/data_provider_decision.md`. If the chosen provider later proves inadequate (gaps, rate-limit issues), that's a documented pivot, not a silent workaround.

## 6. News Provider Selection

Similarly, before building the news engine in Week 4, verify NSE/Indian-company coverage quality for the candidate news API (not just that an API key can be obtained). Record findings in `docs/news_provider_decision.md`.

## 7. When a Metric or Result Looks "Too Good"

If a strategy's backtest, score, or signal looks unusually strong, the default response is suspicion, not celebration:
- Check trade count — is this a 4-trade fluke dressed up with a clean Sharpe ratio?
- Check whether calibration and validation performance diverge sharply (a red flag for overfitting, per `blueprint.md` §4).
- Check whether the result depends on a look-ahead bug (per `PRINCIPLES.md` §2) — rerun the relevant look-ahead test explicitly if one exists, write one if it doesn't.
- Only present the result once it survives this scrutiny, and note the scrutiny performed in `docs/` if the result is genuinely surprising.

## 8. Documentation That Must Stay Current

These files are treated as living documents, updated as part of the work that touches them (not written once and forgotten):

- `docs/regime_methodology.md` — exact regime thresholds and hysteresis rule in effect.
- `docs/lookahead_assumptions.md` — every explicit decision made to prevent look-ahead bias.
- `docs/trading_assumptions.md` — execution timing, cost defaults, position sizing, fractional shares, overnight positions.
- `docs/data_provider_decision.md`, `docs/news_provider_decision.md` — as above.
- `docs/architecture_decisions.md` — any deviation from `blueprint.md`'s structure or approach, with rationale.
- `README.md` — setup instructions, scope notes, disclaimer, screenshots, demo instructions (Week 5).

## 9. Definition of Done (per component)

A component (indicator, strategy, scoring function, API endpoint, dashboard section) is not done until:
- [ ] It matches its description in `blueprint.md`.
- [ ] It follows the relevant conventions in `ENGINEERING_STANDARDS.md`.
- [ ] It has passing unit tests, including a look-ahead test if it touches time-series data (`PRINCIPLES.md` §2).
- [ ] Any hardcoded values that should be config have been moved to config (`ENGINEERING_STANDARDS.md` §5).
- [ ] Any score/signal it produces is explainable in the UI or via drill-down (`PRINCIPLES.md` §6).
- [ ] Relevant `docs/` files are updated if this component introduced or changed an assumption.

## 10. Communication Rhythm

At the end of each week (or each major component, if working faster/slower than the roadmap), summarize: what was built, what deviated from `blueprint.md` and why, what's flagged as a known limitation, and what's next — so the project's actual state and its documentation never silently diverge.
