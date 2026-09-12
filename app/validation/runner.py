"""
runner.py — orchestrates a full controlled validation run.

Architecture:
  SyntheticProvider (with hidden rival SoC)
      ↓
  _walk() from rival_trace.py — runs estimator lap-by-lap, captures per-lap errors
      ↓
  compute_rival_soc_metrics() — aggregate MAE / RMSE / median / P90
  compute_classification_metrics() — accuracy / F1 / confusion matrix
      ↓
  ValidationSummary — returned to the API / tests

Ground truth isolation:
  - _walk() reads ground_truth_rival_soc_by_lap from SyntheticProvider
  - It ONLY attaches gt to the step dict AFTER the estimator has produced its estimate
  - The estimator (RivalStateEstimator) never receives gt as input
  - This module never imports from app.decision.engine to prevent accidental leakage
"""

from __future__ import annotations

from app.decision.config import DecisionConfig
from app.demo.rival_trace import _walk
from app.validation.classification import compute_classification_metrics
from app.validation.models import ValidationSummary
from app.validation.rival_soc import compute_rival_soc_metrics


def run_validation(
    scenario: str = "B",
    seed: int = 42,
    total_laps: int = 50,
) -> ValidationSummary:
    from app.data.providers import build_provider

    provider = build_provider("synthetic", scenario=scenario, seed=seed, total_laps=total_laps)
    cfg = DecisionConfig(seed=seed)

    steps, _estimator = _walk(provider, cfg, up_to_lap=None)

    rival_soc = compute_rival_soc_metrics(steps)
    rival_classification = compute_classification_metrics(
        steps,
        low_edge_mj=cfg.rival_low_soc_mj,
        high_edge_mj=cfg.rival_high_soc_mj,
    )

    return ValidationSummary(
        scenario=scenario,
        seed=seed,
        total_laps=total_laps,
        ground_truth_available=rival_soc.ground_truth_available,
        rival_soc=rival_soc,
        rival_classification=rival_classification,
    )
