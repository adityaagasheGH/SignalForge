# SignalForge — Architecture Decisions

A reference log of major design decisions and their reasoning, so future work
stays consistent. Referenced by `ENGINEERING_STANDARDS.md` §1 and `WORKFLOW.md`
§2/§8. Scan-ability matters — keep entries short.

## Buy & Hold as a 13th Strategy Candidate
**Decision:** Buy & Hold is registered as a full competing candidate in the strategy leaderboard and scoring engine, not just a passive reference comparison. It can rank #1, and its recommendation can be chosen as the final signal.
**Reasoning:** Without this, the system is structurally biased toward always recommending active management even when passive holding is demonstrably better. That would make the final signal engine dishonest by construction.

## Two-Framing Final Signal Output
**Decision:** The final signal always presents both a "best risk-adjusted active strategy" framing and a "Buy & Hold comparison" framing side by side, with actual numbers and rankings for both.
**Reasoning:** A single collapsed verdict hides either the active-strategy option (when Buy & Hold wins) or the passive-return comparison (when an active strategy wins). Presenting both with real numbers lets users apply their own risk preference rather than having it applied for them.
