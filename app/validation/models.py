"""
models.py — validation result types for the ChronoPace evaluation layer.

IMPORTANT: These models are NEVER returned by POST /api/v1/decision.
They are only accessible via GET /api/v1/validation/summary and POST /api/v1/demo/*.

Every metric has ground_truth_available: bool. If False, numeric fields are None
and the note explains why (e.g., "FastF1 does not publish rival ERS SoC").
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class RivalSocValidation(BaseModel):
    """
    Rival SoC estimation accuracy metrics.
    Computed ONLY when a legitimate hidden-state ground truth exists.

    All numeric fields are None when ground_truth_available is False.
    Do NOT present these as real-world F1 accuracy — they are controlled
    synthetic simulations.
    """
    ground_truth_available: bool
    validation_type: str = Field(
        ...,
        description=(
            "controlled_hidden_state — synthetic simulator with hidden rival SoC | "
            "unavailable — ground truth not present (FastF1 replay, live inference)"
        ),
    )
    mae_mj: Optional[float] = Field(None, description="Mean Absolute Error (MJ). None when no ground truth.")
    rmse_mj: Optional[float] = Field(None, description="Root Mean Square Error (MJ).")
    median_ae_mj: Optional[float] = Field(None, description="Median Absolute Error (MJ).")
    p90_ae_mj: Optional[float] = Field(None, description="90th-percentile Absolute Error (MJ).")
    sample_count: int = Field(0, description="Number of laps with ground truth (= evaluation window).")
    evaluation_note: str = Field(
        "",
        description=(
            "Human-readable label. Always clarifies whether this is synthetic/controlled. "
            "Must say 'MODELED — NOT MEASURED' for synthetic. "
            "Must say 'GROUND TRUTH UNAVAILABLE' for replay/live."
        ),
    )
    honesty_notice: str = Field(
        "MODELED — NOT MEASURED. Controlled synthetic simulation only. "
        "Not a claim of real-world F1 rival battery prediction accuracy.",
        description="Immutable reminder that this is not real-world accuracy.",
    )


class ClassificationValidation(BaseModel):
    """
    Rival energy-bucket classification metrics (LOW / MEDIUM / HIGH).
    Compares the estimator's argmax bucket against the ground-truth bucket.
    """
    ground_truth_available: bool
    validation_type: str
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    per_class_f1: Optional[dict] = Field(
        None,
        description="{'LOW': f1, 'MEDIUM': f1, 'HIGH': f1} — per-class F1 score.",
    )
    confusion_matrix: Optional[List[List[int]]] = Field(
        None,
        description="3×3 confusion matrix [[LL, LM, LH], [ML, MM, MH], [HL, HM, HH]] "
        "where rows=true, cols=predicted, order=LOW/MEDIUM/HIGH.",
    )
    sample_count: int = 0
    evaluation_note: str = ""


class NotImplementedMetric(BaseModel):
    """Placeholder for metrics not yet implemented or requiring unavailable ground truth."""
    implemented: bool = False
    ground_truth_available: bool = False
    note: str


class ValidationSummary(BaseModel):
    """
    Complete validation summary for a single controlled scenario run.

    Architecture: Inference → Decision → Validation/Evaluation.
    The evaluator is the ONLY layer that sees ground truth.
    Ground truth is NEVER passed to the inference pipeline.
    """
    scenario: str
    seed: int
    total_laps: int
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Wall-clock timestamp of this validation run.",
    )
    ground_truth_available: bool = Field(
        ...,
        description=(
            "True only for controlled synthetic simulations where the simulator "
            "generated the hidden rival SoC. False for FastF1 replay and live inference."
        ),
    )
    rival_soc: RivalSocValidation
    rival_classification: ClassificationValidation
    # Metrics not yet implemented — stubs with honest ground_truth_available=False
    overtake_calibration: NotImplementedMetric = Field(
        default_factory=lambda: NotImplementedMetric(
            note="Overtake success probability calibration requires real-world outcome data. Not available."
        )
    )
    decision_accuracy: NotImplementedMetric = Field(
        default_factory=lambda: NotImplementedMetric(
            note="Decision accuracy requires defined reference-optimal decisions per scenario. Not yet benchmarked."
        )
    )
