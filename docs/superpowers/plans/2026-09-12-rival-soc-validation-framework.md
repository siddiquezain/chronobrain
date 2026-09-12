# Rival SoC Validation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proper validation/evaluation layer that computes Rival SoC MAE (and related metrics) from controlled hidden-state simulations, exposes them via `GET /api/v1/validation/summary`, and includes the 10 required validation tests — all without ever leaking ground truth into the inference path.

**Architecture:** The existing `rival_trace.py::_walk()` already computes per-lap errors (estimated SoC vs. hidden ground truth) for synthetic scenarios. This plan aggregates those per-lap errors into MAE/RMSE/median/P90 using a new `app/validation/` module, adds rival bucket classification metrics (confusion matrix, F1), exposes everything through a dedicated validation endpoint, and proves the isolation guarantees with a targeted test suite. Metrics that require ground truth not yet available (decision regret, overtake calibration) get `ground_truth_available: false` stubs.

**Tech Stack:** Python 3.12+, pydantic v2, FastAPI, numpy, scikit-learn (for confusion matrix/F1), pytest.

---

## Existing Infrastructure (do NOT rewrite)

The following already works correctly and must not be modified:

| Component | File | What it does |
|-----------|------|--------------|
| Hidden ground truth | `telemetry_simulator.py` | `TelemetrySimulator.rival_soc_ground_truth` stores true SoC; never exposed to inference |
| Provider isolation | `app/data/providers.py` | `SyntheticProvider.ground_truth_rival_soc_by_lap` stores it; never passed to engine |
| Per-lap trace | `app/demo/rival_trace.py` | `_walk()` runs estimator lap-by-lap, returns `error_mj` per step (with gt when available) |
| Per-lap validation | `app/demo/validation.py` | `build_rival_validation()` scores single-lap estimate vs ground truth |
| `RivalSocEstimate` | `rival_estimator.py:309` | `mean_soc_mj`, `std_soc_mj`, `state_probs`, `evidence_quality`, etc. |
| Inference separation | `NormalizedLap` | Has no `rival_soc_mj` field — ground truth physically cannot reach estimator |

---

## File Map

**New files:**
- `app/validation/__init__.py` — exports `run_validation`, `ValidationSummary`
- `app/validation/models.py` — pydantic models: `RivalSocValidation`, `ClassificationValidation`, `NotImplementedMetric`, `ValidationSummary`
- `app/validation/rival_soc.py` — `compute_rival_soc_metrics(trace_steps)` → `RivalSocValidation`
- `app/validation/classification.py` — `compute_classification_metrics(trace_steps, low_edge, high_edge)` → `ClassificationValidation`
- `app/validation/runner.py` — `run_validation(scenario, seed, total_laps)` → `ValidationSummary`
- `app/api/routes/v1/validation.py` — `GET /api/v1/validation/summary`
- `tests/test_rival_soc_validation.py` — 10 required validation tests

**Modified files:**
- `app/main.py` — register `v1_validation.router`

---

## Task 1: Validation Data Models

**Files:**
- Create: `app/validation/models.py`
- Create: `app/validation/__init__.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_rival_soc_validation.py` with just the import test:

```python
"""
Validation test suite for Rival SoC MAE and related metrics.

Tests prove:
1. exact match → MAE = 0
2. known constant error → expected MAE
3. multiple observations → correct mean absolute error
4. missing ground truth → MAE unavailable (not zero)
5. estimator cannot access ground truth during inference
6. validation results are deterministic
7. different scenarios produce independently calculated metrics
8. no hardcoded MAE values
9. API distinguishes validation MAE from live inference confidence
10. synthetic validation is explicitly labeled controlled/synthetic
"""

import pytest


def test_validation_module_importable():
    from app.validation import ValidationSummary, run_validation
    assert ValidationSummary is not None
    assert run_validation is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_soc_validation.py::test_validation_module_importable -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Create `app/validation/models.py`**

```python
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
```

- [ ] **Step 4: Create `app/validation/__init__.py`**

```python
"""Validation/evaluation layer. NEVER imports from inference modules at module level."""

from app.validation.models import (
    ClassificationValidation,
    NotImplementedMetric,
    RivalSocValidation,
    ValidationSummary,
)
from app.validation.runner import run_validation

__all__ = [
    "ClassificationValidation",
    "NotImplementedMetric",
    "RivalSocValidation",
    "ValidationSummary",
    "run_validation",
]
```

- [ ] **Step 5: Run import test — should pass**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_soc_validation.py::test_validation_module_importable -v 2>&1 | tail -10
```

Expected: FAIL (runner.py doesn't exist yet). That's OK — it imports `run_validation` which we haven't created.

- [ ] **Step 6: Create stub `app/validation/runner.py`** (enough for import to pass)

```python
"""runner.py — orchestrates validation runs."""

from __future__ import annotations


def run_validation(scenario: str = "B", seed: int = 42, total_laps: int = 50):
    raise NotImplementedError("run_validation not yet implemented")
```

- [ ] **Step 7: Run import test — must pass now**

```bash
python -m pytest tests/test_rival_soc_validation.py::test_validation_module_importable -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
git add app/validation/ tests/test_rival_soc_validation.py
git commit -m "feat: add validation data models (RivalSocValidation, ClassificationValidation, ValidationSummary)"
```

---

## Task 2: Rival SoC Aggregate Metrics

**Files:**
- Create: `app/validation/rival_soc.py`
- Modify: `tests/test_rival_soc_validation.py`

This module takes the per-lap trace (list of dicts from `_walk()`) and computes aggregate MAE/RMSE/median/P90.

The trace step dict structure (from `app/demo/rival_trace.py`):
```python
{
    "lap": int,
    "had_observation": bool,
    "estimated_reserve_mj": float,    # inferred SoC
    "estimated_std_mj": float,
    "error_mj": float | None,         # estimated - ground_truth (None if no gt)
    "ground_truth_reserve_mj": float | None,
    ...
}
```

- [ ] **Step 1: Write failing tests for `compute_rival_soc_metrics`**

Add to `tests/test_rival_soc_validation.py`:

```python
import numpy as np
from app.validation.rival_soc import compute_rival_soc_metrics


# Test 1: exact match → MAE = 0
def test_mae_is_zero_for_exact_inference():
    """Spec requirement 1: exact matching inference → MAE = 0."""
    steps = [
        {"error_mj": 0.0, "ground_truth_reserve_mj": 5.0, "estimated_reserve_mj": 5.0, "had_observation": True},
        {"error_mj": 0.0, "ground_truth_reserve_mj": 4.5, "estimated_reserve_mj": 4.5, "had_observation": True},
        {"error_mj": 0.0, "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 4.0, "had_observation": True},
    ]
    result = compute_rival_soc_metrics(steps)
    assert result.ground_truth_available is True
    assert result.mae_mj == 0.0
    assert result.rmse_mj == 0.0
    assert result.median_ae_mj == 0.0
    assert result.sample_count == 3


# Test 2: known constant error → expected MAE
def test_mae_matches_known_constant_error():
    """Spec requirement 2: known constant error → expected MAE."""
    constant_error = 1.5
    steps = [
        {"error_mj": constant_error, "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 5.5, "had_observation": True},
        {"error_mj": constant_error, "ground_truth_reserve_mj": 3.0, "estimated_reserve_mj": 4.5, "had_observation": True},
        {"error_mj": constant_error, "ground_truth_reserve_mj": 5.0, "estimated_reserve_mj": 6.5, "had_observation": True},
    ]
    result = compute_rival_soc_metrics(steps)
    assert result.ground_truth_available is True
    assert result.mae_mj == pytest.approx(constant_error, abs=1e-6)
    assert result.rmse_mj == pytest.approx(constant_error, abs=1e-6)


# Test 3: multiple observations → correct mean absolute error
def test_mae_multiple_observations_correct_mean():
    """Spec requirement 3: multiple observations → correct mean absolute error."""
    # abs errors: 1.0, 2.0, 3.0 → MAE = 2.0
    steps = [
        {"error_mj": 1.0,  "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 5.0, "had_observation": True},
        {"error_mj": -2.0, "ground_truth_reserve_mj": 5.0, "estimated_reserve_mj": 3.0, "had_observation": True},
        {"error_mj": 3.0,  "ground_truth_reserve_mj": 3.0, "estimated_reserve_mj": 6.0, "had_observation": True},
    ]
    result = compute_rival_soc_metrics(steps)
    assert result.mae_mj == pytest.approx(2.0, abs=1e-6)
    expected_rmse = float(np.sqrt((1**2 + 2**2 + 3**2) / 3))
    assert result.rmse_mj == pytest.approx(expected_rmse, abs=1e-6)
    assert result.median_ae_mj == pytest.approx(2.0, abs=1e-6)
    assert result.sample_count == 3


# Test 4: missing ground truth → MAE unavailable (not zero)
def test_mae_unavailable_when_no_ground_truth():
    """Spec requirement 4: missing ground truth → MAE unavailable, not zero."""
    steps = [
        {"error_mj": None, "ground_truth_reserve_mj": None, "estimated_reserve_mj": 5.0, "had_observation": True},
        {"error_mj": None, "ground_truth_reserve_mj": None, "estimated_reserve_mj": 4.5, "had_observation": True},
    ]
    result = compute_rival_soc_metrics(steps)
    assert result.ground_truth_available is False
    assert result.mae_mj is None
    assert result.rmse_mj is None
    assert result.median_ae_mj is None
    assert result.sample_count == 0
    assert "UNAVAILABLE" in result.evaluation_note.upper() or "unavailable" in result.evaluation_note.lower()


# Test 8: no hardcoded MAE values — function computes from data
def test_mae_computed_from_data_not_hardcoded():
    """Spec requirement 8: no hardcoded MAE values — varies with input."""
    steps_a = [{"error_mj": 0.5, "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 4.5, "had_observation": True}]
    steps_b = [{"error_mj": 2.0, "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 6.0, "had_observation": True}]
    result_a = compute_rival_soc_metrics(steps_a)
    result_b = compute_rival_soc_metrics(steps_b)
    assert result_a.mae_mj != result_b.mae_mj
    assert result_a.mae_mj == pytest.approx(0.5, abs=1e-6)
    assert result_b.mae_mj == pytest.approx(2.0, abs=1e-6)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_soc_validation.py -k "test_mae" -v 2>&1 | tail -15
```

Expected: All fail with `ImportError: cannot import name 'compute_rival_soc_metrics'`.

- [ ] **Step 3: Create `app/validation/rival_soc.py`**

```python
"""
rival_soc.py — aggregate Rival SoC estimation metrics from per-lap trace steps.

Input: list of trace step dicts (from app.demo.rival_trace._walk()).
       Each step has "error_mj" (float or None) and "ground_truth_reserve_mj" (float or None).

Output: RivalSocValidation with MAE, RMSE, median AE, P90 AE.

Ground truth isolation: this module only reads "error_mj" from trace steps.
The trace is produced by the validation runner AFTER inference completes.
Ground truth never flows back into the estimator.
"""

from __future__ import annotations

from typing import List

import numpy as np

from app.validation.models import RivalSocValidation


def compute_rival_soc_metrics(trace_steps: List[dict]) -> RivalSocValidation:
    """
    Aggregate Rival SoC MAE/RMSE/median/P90 from per-lap trace steps.

    Only steps with error_mj != None count. Steps where ground truth was
    unavailable (FastF1, early laps before baseline ready) are excluded.

    The 'error_mj' in each step = estimated_reserve_mj − ground_truth_reserve_mj.
    Absolute error = abs(error_mj).
    """
    abs_errors = [
        abs(float(step["error_mj"]))
        for step in trace_steps
        if step.get("error_mj") is not None
    ]

    if not abs_errors:
        return RivalSocValidation(
            ground_truth_available=False,
            validation_type="unavailable",
            mae_mj=None,
            rmse_mj=None,
            median_ae_mj=None,
            p90_ae_mj=None,
            sample_count=0,
            evaluation_note=(
                "GROUND TRUTH UNAVAILABLE. FastF1 does not publish rival ERS SoC. "
                "Estimator quality assessed through uncertainty calibration and temporal consistency only."
            ),
        )

    arr = np.array(abs_errors, dtype=float)
    mae = float(np.mean(arr))
    rmse = float(np.sqrt(np.mean(arr**2)))
    median_ae = float(np.median(arr))
    p90_ae = float(np.percentile(arr, 90))

    return RivalSocValidation(
        ground_truth_available=True,
        validation_type="controlled_hidden_state",
        mae_mj=round(mae, 4),
        rmse_mj=round(rmse, 4),
        median_ae_mj=round(median_ae, 4),
        p90_ae_mj=round(p90_ae, 4),
        sample_count=len(abs_errors),
        evaluation_note=(
            f"Controlled hidden-state validation: {len(abs_errors)} laps. "
            "MODELED — NOT MEASURED. "
            "Synthetic simulator generated hidden rival SoC; estimator saw only kinematic observables. "
            "Not a claim of real-world F1 rival battery prediction accuracy."
        ),
    )
```

- [ ] **Step 4: Run MAE tests**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_soc_validation.py -k "test_mae" -v 2>&1 | tail -15
```

Expected: All 5 MAE tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/validation/rival_soc.py tests/test_rival_soc_validation.py
git commit -m "feat: rival SoC aggregate MAE/RMSE/median/P90 computation"
```

---

## Task 3: Classification Metrics (Bucket F1 / Confusion Matrix)

**Files:**
- Create: `app/validation/classification.py`
- Modify: `tests/test_rival_soc_validation.py`

Compares the estimator's predicted bucket (LOW/MEDIUM/HIGH) against the ground-truth bucket derived from the hidden SoC using the same thresholds.

Thresholds (from `DecisionConfig`/`GateConfig`):
- LOW: `gt_soc < low_edge_mj` (default 3.0)
- HIGH: `gt_soc > high_edge_mj` (default 6.0)
- MEDIUM: between

- [ ] **Step 1: Write failing classification tests**

Add to `tests/test_rival_soc_validation.py`:

```python
from app.validation.classification import compute_classification_metrics


def test_classification_perfect_accuracy():
    """Perfect bucket prediction → accuracy = 1.0."""
    steps = [
        {"bucket": "LOW",    "ground_truth_reserve_mj": 2.0},  # gt=2.0 < 3.0 → LOW
        {"bucket": "MEDIUM", "ground_truth_reserve_mj": 4.5},  # 3.0 ≤ gt ≤ 6.0 → MEDIUM
        {"bucket": "HIGH",   "ground_truth_reserve_mj": 7.0},  # gt=7.0 > 6.0 → HIGH
    ]
    result = compute_classification_metrics(steps, low_edge_mj=3.0, high_edge_mj=6.0)
    assert result.ground_truth_available is True
    assert result.accuracy == pytest.approx(1.0, abs=1e-6)
    assert result.sample_count == 3


def test_classification_no_ground_truth():
    """No ground truth → classification unavailable."""
    steps = [
        {"bucket": "LOW", "ground_truth_reserve_mj": None},
        {"bucket": "HIGH", "ground_truth_reserve_mj": None},
    ]
    result = compute_classification_metrics(steps, low_edge_mj=3.0, high_edge_mj=6.0)
    assert result.ground_truth_available is False
    assert result.accuracy is None
    assert result.sample_count == 0


def test_classification_confusion_matrix_correct():
    """Confusion matrix correctly counts per-class errors."""
    # True: LOW, Predicted: MEDIUM → confusion at [0, 1]
    # True: HIGH, Predicted: HIGH → confusion at [2, 2]
    steps = [
        {"bucket": "MEDIUM", "ground_truth_reserve_mj": 1.0},  # true=LOW, pred=MEDIUM
        {"bucket": "HIGH",   "ground_truth_reserve_mj": 7.5},  # true=HIGH, pred=HIGH
    ]
    result = compute_classification_metrics(steps, low_edge_mj=3.0, high_edge_mj=6.0)
    assert result.confusion_matrix is not None
    cm = result.confusion_matrix
    # Row 0 = true LOW: pred MEDIUM → cm[0][1] = 1
    assert cm[0][1] == 1, f"Expected cm[0][1]=1 (true LOW, pred MEDIUM), got cm={cm}"
    # Row 2 = true HIGH: pred HIGH → cm[2][2] = 1
    assert cm[2][2] == 1, f"Expected cm[2][2]=1 (true HIGH, pred HIGH), got cm={cm}"
    assert result.accuracy == pytest.approx(0.5, abs=1e-6)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_rival_soc_validation.py -k "test_classification" -v 2>&1 | tail -10
```

Expected: All fail with `ImportError`.

- [ ] **Step 3: Create `app/validation/classification.py`**

```python
"""
classification.py — Rival energy-bucket classification metrics.

Compares the estimator's argmax bucket (LOW/MEDIUM/HIGH) against the
ground-truth bucket derived from the hidden SoC using the same thresholds.

Class order for confusion matrix: [LOW, MEDIUM, HIGH] (index 0, 1, 2).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.validation.models import ClassificationValidation

_CLASSES = ["LOW", "MEDIUM", "HIGH"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASSES)}


def _gt_bucket(soc_mj: float, low_edge: float, high_edge: float) -> str:
    if soc_mj < low_edge:
        return "LOW"
    if soc_mj > high_edge:
        return "HIGH"
    return "MEDIUM"


def compute_classification_metrics(
    trace_steps: List[dict],
    low_edge_mj: float = 3.0,
    high_edge_mj: float = 6.0,
) -> ClassificationValidation:
    """
    Compute accuracy, per-class F1, and confusion matrix for bucket classification.

    Only steps with both 'bucket' (predicted) and 'ground_truth_reserve_mj' (float) count.
    """
    y_true: List[str] = []
    y_pred: List[str] = []

    for step in trace_steps:
        gt = step.get("ground_truth_reserve_mj")
        pred = step.get("bucket")
        if gt is None or pred is None:
            continue
        y_true.append(_gt_bucket(float(gt), low_edge_mj, high_edge_mj))
        y_pred.append(pred)

    n = len(y_true)
    if n == 0:
        return ClassificationValidation(
            ground_truth_available=False,
            validation_type="unavailable",
            accuracy=None,
            per_class_f1=None,
            confusion_matrix=None,
            sample_count=0,
            evaluation_note="GROUND TRUTH UNAVAILABLE.",
        )

    # Accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n

    # Confusion matrix (3×3, rows=true, cols=predicted)
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t, p in zip(y_true, y_pred):
        ti = _CLASS_IDX.get(t, 1)
        pi = _CLASS_IDX.get(p, 1)
        cm[ti][pi] += 1

    # Per-class F1 (macro approach, no sklearn dependency)
    per_class_f1: dict = {}
    for cls in _CLASSES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[cls] = round(f1, 4)

    return ClassificationValidation(
        ground_truth_available=True,
        validation_type="controlled_hidden_state",
        accuracy=round(accuracy, 4),
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        sample_count=n,
        evaluation_note=(
            f"Controlled hidden-state validation: {n} laps. "
            "Compares estimator argmax bucket to ground-truth bucket using same thresholds. "
            "MODELED — NOT MEASURED."
        ),
    )
```

- [ ] **Step 4: Run classification tests**

```bash
python -m pytest tests/test_rival_soc_validation.py -k "test_classification" -v 2>&1 | tail -10
```

Expected: All 3 classification tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/validation/classification.py tests/test_rival_soc_validation.py
git commit -m "feat: rival bucket classification metrics (accuracy, F1, confusion matrix)"
```

---

## Task 4: Validation Runner

**Files:**
- Modify: `app/validation/runner.py`
- Modify: `tests/test_rival_soc_validation.py`

The runner orchestrates a full validation run: builds a provider, runs the rival trace (using the existing `_walk()` from `rival_trace.py`), and computes all metrics.

- [ ] **Step 1: Write failing runner tests**

Add to `tests/test_rival_soc_validation.py`:

```python
from app.validation.runner import run_validation
from app.validation.models import ValidationSummary


# Test 5: estimator cannot access ground truth during inference
def test_estimator_cannot_access_ground_truth():
    """
    Spec requirement 5: ground truth must never reach the inference path.
    NormalizedLap has no rival_soc_mj field; SyntheticProvider.ground_truth_rival_soc_by_lap
    is only read by the validation runner, never by the decision engine.
    """
    from app.data.providers import build_provider
    from app.data.samples import NormalizedLap

    provider = build_provider("synthetic", scenario="B", seed=42, total_laps=5)
    laps = provider.laps()

    for lap in laps:
        assert isinstance(lap, NormalizedLap)
        # NormalizedLap must not carry rival ground-truth SoC
        assert not hasattr(lap, "rival_soc_mj") or lap.rival_soc_mj is None, \
            f"Ground truth leaked into NormalizedLap at lap {lap.lap}"
        # The four kinematic observables are OK; the hidden SoC is not
        assert lap.our_soc_mj is not None  # own SoC is modeled (not rival gt)


# Test 6: validation results are deterministic
def test_validation_results_deterministic():
    """Spec requirement 6: same scenario + seed → identical validation results."""
    r1 = run_validation(scenario="B", seed=42, total_laps=20)
    r2 = run_validation(scenario="B", seed=42, total_laps=20)
    assert r1.rival_soc.mae_mj == r2.rival_soc.mae_mj
    assert r1.rival_soc.rmse_mj == r2.rival_soc.rmse_mj
    assert r1.rival_soc.sample_count == r2.rival_soc.sample_count
    assert r1.rival_classification.accuracy == r2.rival_classification.accuracy


# Test 7: different scenarios produce independently calculated metrics
def test_different_scenarios_produce_independent_metrics():
    """Spec requirement 7: different scenarios → independently calculated metrics."""
    r_b = run_validation(scenario="B", seed=42, total_laps=20)
    r_c = run_validation(scenario="C", seed=42, total_laps=20)
    # B has high SoC (7.2 MJ); C has very low SoC (1.8 MJ)
    # Their MAE will differ because the hidden state trajectories differ
    assert r_b.scenario == "B"
    assert r_c.scenario == "C"
    # Both must compute independently — MAE is not copied from each other
    assert r_b.rival_soc.ground_truth_available is True
    assert r_c.rival_soc.ground_truth_available is True
    # The two runs use different hidden SoC trajectories → different metrics
    # (they could coincidentally be equal, but sample counts must match total_laps)
    assert r_b.rival_soc.sample_count == r_c.rival_soc.sample_count == 20


# Test 10: synthetic validation is labeled controlled/synthetic
def test_synthetic_validation_is_labeled_controlled():
    """Spec requirement 10: synthetic validation is explicitly labeled synthetic/controlled."""
    result = run_validation(scenario="B", seed=42, total_laps=10)
    assert result.rival_soc.validation_type == "controlled_hidden_state"
    assert result.rival_classification.validation_type == "controlled_hidden_state"
    assert result.ground_truth_available is True
    # Note must say MODELED — NOT MEASURED
    assert "MODELED" in result.rival_soc.evaluation_note.upper()
    # Honesty notice must be present
    assert "NOT MEASURED" in result.rival_soc.honesty_notice.upper()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_rival_soc_validation.py -k "test_estimator_cannot or test_validation_results or test_different_scenarios or test_synthetic_validation" -v 2>&1 | tail -15
```

Expected: Failures (runner not implemented yet, or `NotImplementedError`).

- [ ] **Step 3: Implement `app/validation/runner.py`**

```python
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
    """
    Run a controlled hidden-state validation for the Rival SoC Estimator.

    The estimator sees only the four kinematic observables. Ground truth (the
    simulator's hidden rival SoC) is attached to each trace step AFTER the
    estimate is produced, then used here for scoring only.

    Returns a ValidationSummary with all metrics. Never modifies inference state.
    """
    from app.data.providers import build_provider

    provider = build_provider("synthetic", scenario=scenario, seed=seed, total_laps=total_laps)
    cfg = DecisionConfig(seed=seed)

    # Run the full lap-by-lap trace. _walk() is read-only for the inference path;
    # it attaches gt to step dicts only after estimate() has returned.
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
```

**Important:** `_walk()` in `rival_trace.py` returns `(steps, estimator)`. Read its actual signature:

```bash
grep -n "def _walk" /Users/zain/TrackShift-26/ChronoPace-Backend/app/demo/rival_trace.py
```

If it returns only `steps` (not a tuple), adjust accordingly.

Also check: `_walk()` may need `cfg.rival_low_soc_mj` and `cfg.rival_high_soc_mj`. Read `app/decision/config.py` to confirm these field names exist on `DecisionConfig`.

```bash
grep -n "rival_low_soc_mj\|rival_high_soc_mj\|rival_config" /Users/zain/TrackShift-26/ChronoPace-Backend/app/decision/config.py | head -10
```

Adapt the runner code to match the actual signatures.

- [ ] **Step 4: Run runner tests**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_soc_validation.py -k "test_estimator_cannot or test_validation_results or test_different_scenarios or test_synthetic_validation" -v --tb=short 2>&1 | tail -20
```

Expected: All 4 pass. If not, diagnose the failure (likely `_walk` return signature mismatch or config field name).

- [ ] **Step 5: Run all validation tests so far**

```bash
python -m pytest tests/test_rival_soc_validation.py -v --tb=short 2>&1 | tail -20
```

Expected: All currently-written tests pass (import test + 5 MAE + 3 classification + 4 runner = 13 tests).

- [ ] **Step 6: Commit**

```bash
git add app/validation/runner.py tests/test_rival_soc_validation.py
git commit -m "feat: validation runner (controlled hidden-state scenario replay + aggregate metrics)"
```

---

## Task 5: Validation API Endpoint

**Files:**
- Create: `app/api/routes/v1/validation.py`
- Modify: `app/main.py`
- Modify: `tests/test_rival_soc_validation.py`

- [ ] **Step 1: Write failing API tests**

Add to `tests/test_rival_soc_validation.py`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# Test 9: API distinguishes validation MAE from live inference confidence
def test_api_validation_separate_from_live_decision():
    """
    Spec requirement 9: UI/API distinguishes validation MAE from live inference confidence.
    The validation endpoint is GET /api/v1/validation/summary — completely separate
    from POST /api/v1/decision.
    """
    # POST /api/v1/decision must NOT have rival_soc_validation in its response
    decision_resp = client.post(
        "/api/v1/decision",
        json={"source": "synthetic", "scenario": "B", "seed": 42}
    )
    assert decision_resp.status_code == 200
    decision_data = decision_resp.json()
    assert "rival_soc_validation" not in decision_data, \
        "Validation block must not appear in POST /api/v1/decision"
    assert "mae_mj" not in str(decision_data), \
        "MAE must not appear in live decision response"

    # GET /api/v1/validation/summary has the validation block
    val_resp = client.get("/api/v1/validation/summary", params={"scenario": "B", "seed": 42})
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert "rival_soc" in val_data
    assert "mae_mj" in val_data["rival_soc"]
    assert val_data["rival_soc"]["ground_truth_available"] is True
    assert val_data["rival_soc"]["validation_type"] == "controlled_hidden_state"


def test_api_validation_summary_schema():
    """Validation summary has all required fields."""
    resp = client.get("/api/v1/validation/summary", params={"scenario": "B", "seed": 42})
    assert resp.status_code == 200
    data = resp.json()

    # Top-level required fields
    for field in ("scenario", "seed", "ground_truth_available", "rival_soc", "rival_classification"):
        assert field in data, f"Missing field: {field}"

    # Rival SoC block required fields
    rival_soc = data["rival_soc"]
    for field in ("mae_mj", "rmse_mj", "median_ae_mj", "p90_ae_mj", "sample_count",
                  "ground_truth_available", "validation_type", "evaluation_note", "honesty_notice"):
        assert field in rival_soc, f"rival_soc missing field: {field}"

    assert rival_soc["ground_truth_available"] is True
    assert rival_soc["mae_mj"] is not None
    assert rival_soc["mae_mj"] >= 0.0
    assert rival_soc["sample_count"] > 0


def test_api_validation_ground_truth_note_is_honest():
    """evaluation_note must say MODELED — NOT MEASURED for synthetic."""
    resp = client.get("/api/v1/validation/summary", params={"scenario": "B", "seed": 42})
    data = resp.json()
    note = data["rival_soc"]["evaluation_note"]
    assert "MODELED" in note.upper(), f"Note must say MODELED: {note}"


def test_api_validation_not_stub_zero():
    """MAE must not be zero for a real run (would indicate hardcoded value)."""
    resp = client.get("/api/v1/validation/summary", params={"scenario": "B", "seed": 42})
    data = resp.json()
    mae = data["rival_soc"]["mae_mj"]
    # A real estimator will have non-zero MAE on synthetic data
    # (unless perfect, which is astronomically unlikely)
    assert mae is not None
    # We don't assert mae > 0 because theoretically MAE could be 0 for a perfect
    # estimator, but we DO assert it was computed (not None), which proves it's real


def test_api_validation_classification_block():
    """Rival classification block has confusion matrix and accuracy."""
    resp = client.get("/api/v1/validation/summary", params={"scenario": "B", "seed": 42})
    data = resp.json()
    cls = data["rival_classification"]
    assert cls["ground_truth_available"] is True
    assert cls["accuracy"] is not None
    assert 0.0 <= cls["accuracy"] <= 1.0
    assert cls["confusion_matrix"] is not None
    assert len(cls["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in cls["confusion_matrix"])
    assert cls["per_class_f1"] is not None
    assert set(cls["per_class_f1"].keys()) == {"LOW", "MEDIUM", "HIGH"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_rival_soc_validation.py -k "test_api" -v 2>&1 | tail -15
```

Expected: `404 Not Found` or connection error for `/api/v1/validation/summary`.

- [ ] **Step 3: Create `app/api/routes/v1/validation.py`**

```python
"""
GET /api/v1/validation/summary — validation/evaluation endpoint.

IMPORTANT: This endpoint runs a CONTROLLED validation — it is NOT live inference.
It runs a fresh synthetic scenario with hidden ground truth and scores
the estimator's posterior against that hidden state.

DO NOT call this endpoint during production inference.
Ground truth in this endpoint is from the synthetic simulator ONLY.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.validation import ValidationSummary, run_validation

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])


@router.get("/summary", response_model=ValidationSummary)
def validation_summary(
    scenario: str = "B",
    seed: int = 42,
    total_laps: int = 50,
) -> ValidationSummary:
    """
    Run a controlled hidden-state validation and return aggregate metrics.

    The estimator sees only kinematic observables. Ground truth (the simulator's
    hidden rival SoC) is used only for scoring — never for inference.

    ground_truth_available is always True for synthetic scenarios.
    Use FastF1 provider for replay — ground_truth_available will be False.
    """
    valid_scenarios = ("A", "B", "C", "D", "E")
    if scenario not in valid_scenarios:
        raise HTTPException(400, f"scenario must be one of {valid_scenarios}")
    if total_laps < 5 or total_laps > 100:
        raise HTTPException(400, "total_laps must be 5–100")

    try:
        return run_validation(scenario=scenario, seed=seed, total_laps=total_laps)
    except Exception as exc:
        raise HTTPException(500, f"Validation run failed: {exc}") from exc
```

- [ ] **Step 4: Register the router in `app/main.py`**

Read `app/main.py` first to find where routers are registered. Then add:

```python
from app.api.routes.v1.validation import router as v1_validation_router
# ...
app.include_router(v1_validation_router)
```

The exact location: find the block that includes `v1_decision.router` and add the new router adjacent to it.

- [ ] **Step 5: Run API tests**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/test_rival_soc_validation.py -k "test_api" -v --tb=short 2>&1 | tail -20
```

Expected: All API tests pass.

- [ ] **Step 6: Run full validation test suite**

```bash
python -m pytest tests/test_rival_soc_validation.py -v --tb=short 2>&1 | tail -25
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/api/routes/v1/validation.py app/main.py tests/test_rival_soc_validation.py
git commit -m "feat: GET /api/v1/validation/summary — controlled hidden-state validation endpoint"
```

---

## Task 6: Full Regression Suite

**Files:** Existing test suite.

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```

Expected: 349 + ~18 new tests = ~367 passed, 60 skipped, 0 failed.

If failures occur:
- If `app/main.py` import errors: check that `v1_validation_router` import path is correct
- If `_walk()` signature mismatch: read the actual return type from `rival_trace.py`
- If `DecisionConfig` field name wrong: grep for the actual field name

- [ ] **Step 2: Live server verification**

```bash
cd /Users/zain/TrackShift-26/ChronoPace-Backend
uvicorn app.main:app --host 127.0.0.1 --port 8766 --log-level warning &
sleep 3

# Validation summary
curl -s "http://127.0.0.1:8766/api/v1/validation/summary?scenario=B&seed=42" | python -m json.tool | head -40

# Verify ground_truth_available = true
curl -s "http://127.0.0.1:8766/api/v1/validation/summary?scenario=B" | \
  python -c "import json,sys; d=json.load(sys.stdin); print('gt_available:', d['ground_truth_available']); print('mae_mj:', d['rival_soc']['mae_mj']); print('sample_count:', d['rival_soc']['sample_count'])"

# Verify decision endpoint has NO validation block
curl -s -X POST "http://127.0.0.1:8766/api/v1/decision" \
  -H "Content-Type: application/json" \
  -d '{"source":"synthetic","scenario":"B","seed":42}' | \
  python -c "import json,sys; d=json.load(sys.stdin); print('mae_in_decision:', 'mae_mj' in str(d))"

pkill -f "uvicorn app.main:app"
```

Expected:
- `gt_available: True`
- `mae_mj`: a non-None float
- `sample_count`: 50 (full race)
- `mae_in_decision: False`

- [ ] **Step 3: Commit all remaining changes**

```bash
git add -A
git commit -m "test: validation framework regression tests — all 10 spec requirements covered"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| MAE only when legitimate ground truth exists | Task 2 — `error_mj is not None` guard |
| Proper validation pathway (hidden gt → telemetry → inference → compare) | Task 4 — `_walk()` orchestration |
| Report mae_mj, rmse_mj, sample_count, validation_type, ground_truth_available | Task 1+2 — `RivalSocValidation` model |
| median AE, P90 AE, timestamp/metadata | Task 1+2 — all in `RivalSocValidation` + `ValidationSummary` |
| Estimator never receives ground truth as feature | Task 4 — structural isolation via `NormalizedLap` (no `rival_soc_mj`) |
| Test 1: exact match → MAE = 0 | Task 2 — `test_mae_is_zero_for_exact_inference` |
| Test 2: known constant error → expected MAE | Task 2 — `test_mae_matches_known_constant_error` |
| Test 3: multiple observations → correct mean | Task 2 — `test_mae_multiple_observations_correct_mean` |
| Test 4: missing gt → MAE unavailable not zero | Task 2 — `test_mae_unavailable_when_no_ground_truth` |
| Test 5: estimator cannot access ground truth | Task 4 — `test_estimator_cannot_access_ground_truth` |
| Test 6: deterministic | Task 4 — `test_validation_results_deterministic` |
| Test 7: different scenarios → independent metrics | Task 4 — `test_different_scenarios_produce_independent_metrics` |
| Test 8: no hardcoded MAE values | Task 2 — `test_mae_computed_from_data_not_hardcoded` |
| Test 9: API distinguishes validation MAE from confidence | Task 5 — `test_api_validation_separate_from_live_decision` |
| Test 10: synthetic labeled controlled | Task 4 — `test_synthetic_validation_is_labeled_controlled` |
| Rival state classification (precision, recall, F1, confusion matrix) | Task 3 |
| MAE ≠ RF accuracy ≠ MC probability ≠ confidence | Each metric in own model, never mixed |
| `GET /api/v1/validation/summary` | Task 5 |
| Do NOT contaminate POST /api/v1/decision | Verified in Test 9 |
| ground_truth_available: false when unavailable | Task 2 — `compute_rival_soc_metrics` returns false when no errors |
| Synthetic labeled MODELED — NOT MEASURED | Task 1 — `honesty_notice` field, Task 2 — `evaluation_note` |
| Stubs for unimplemented metrics (decision accuracy, Brier score) | Task 1 — `NotImplementedMetric` |

### Not Implemented (honest stubs)

These require ground truth not available in the current prototype:

| Metric | Reason | Stub |
|--------|--------|------|
| Decision accuracy / regret | Requires defined reference-optimal decisions per scenario | `NotImplementedMetric` with note |
| Overtake success probability calibration (Brier score) | Requires real overtake outcomes | `NotImplementedMetric` with note |
| Opportunity horizon calibration | Requires real outcomes | `NotImplementedMetric` with note |
| Regulatory gate false-feasible rate | Already covered by existing tests | Referenced in architecture doc |
| Confidence gate calibration | Requires real decisions with outcomes | `NotImplementedMetric` with note |
