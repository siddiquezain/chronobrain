"""
app.decision — the canonical, deterministic decision pipeline.

    NormalizedLap  (app.data)
        -> feature extraction        (energy, overtake score, ML P(success))
        -> rival particle filter     (rival_estimator, Stack A)
        -> regulatory gate           (rule_gate, Stage 1)
        -> Monte Carlo planner       (planner, Stage 2)
        -> opportunity horizon       (opportunity_engine)
        -> confidence gate           (confidence_gate, Stage 3)
        -> DECISION ENGINE (fusion)  -> final mode + action + reason codes
        -> DecisionSnapshot (verified JSON)   <-- the LLM only ever sees this
        -> DecisionTrace

`run_decision()` returns a fully-formed `DecisionSnapshot` with NO language-model
call anywhere in it. `app.narrative.narrate()` is a separate, optional step that
takes the finished snapshot and cannot change a single number in it.

Determinism: every stochastic component (planner, horizon, particle filter) is
seeded from `DecisionConfig.seed`. Same lap list + same config + same seed =>
byte-identical snapshot (excluding `meta.generated_at`).
"""

from app.decision.config import DecisionConfig
from app.decision.engine import run_decision, run_scenario
from app.decision.snapshot import DecisionSnapshot

__all__ = ["DecisionConfig", "run_decision", "run_scenario", "DecisionSnapshot"]
