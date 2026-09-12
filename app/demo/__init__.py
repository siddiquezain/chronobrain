"""
app.demo — the interactive demonstration + validation layer.

It changes NOTHING about the decision architecture. It:

  * lets a caller override synthetic race INPUTS (our SoC, the hidden rival SoC,
    gap, rival terminal speed, deployment, telemetry noise) — the deterministic
    Python pipeline still computes the recommendation;
  * scores the Rival Energy Estimator against the simulator's hidden ground truth
    (which the estimator never sees) — DEMO ONLY;
  * replays the estimator lap-by-lap so a frontend can animate the posterior
    actually updating.

`POST /api/v1/decision` and `DecisionSnapshot` are untouched — ground truth never
appears there. The demo endpoints live under `/api/v1/demo/*`.
"""

from app.demo.models import InputOverrides
from app.demo.presets import DEMO_PRESETS, resolve_preset
from app.demo.rival_trace import rival_estimator_trace
from app.demo.validation import build_rival_validation

__all__ = [
    "InputOverrides",
    "DEMO_PRESETS",
    "resolve_preset",
    "rival_estimator_trace",
    "build_rival_validation",
]
