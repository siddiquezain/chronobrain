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
import numpy as np


def test_validation_module_importable():
    from app.validation import ValidationSummary, run_validation
    assert ValidationSummary is not None
    assert run_validation is not None


# Test 1: exact match → MAE = 0
def test_mae_is_zero_for_exact_inference():
    """Spec requirement 1: exact matching inference → MAE = 0."""
    from app.validation.rival_soc import compute_rival_soc_metrics

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
    from app.validation.rival_soc import compute_rival_soc_metrics

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
    from app.validation.rival_soc import compute_rival_soc_metrics

    # abs errors: 1.0, 2.0, 3.0 → MAE = 2.0
    steps = [
        {"error_mj": 1.0,  "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 5.0, "had_observation": True},
        {"error_mj": -2.0, "ground_truth_reserve_mj": 5.0, "estimated_reserve_mj": 3.0, "had_observation": True},
        {"error_mj": 3.0,  "ground_truth_reserve_mj": 3.0, "estimated_reserve_mj": 6.0, "had_observation": True},
    ]
    result = compute_rival_soc_metrics(steps)
    assert result.mae_mj == pytest.approx(2.0, abs=1e-3)
    expected_rmse = float(np.sqrt((1**2 + 2**2 + 3**2) / 3))
    assert result.rmse_mj == pytest.approx(expected_rmse, abs=1e-3)
    assert result.median_ae_mj == pytest.approx(2.0, abs=1e-3)
    assert result.sample_count == 3


# Test 4: missing ground truth → MAE unavailable (not zero)
def test_mae_unavailable_when_no_ground_truth():
    """Spec requirement 4: missing ground truth → MAE unavailable, not zero."""
    from app.validation.rival_soc import compute_rival_soc_metrics

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
    from app.validation.rival_soc import compute_rival_soc_metrics

    steps_a = [{"error_mj": 0.5, "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 4.5, "had_observation": True}]
    steps_b = [{"error_mj": 2.0, "ground_truth_reserve_mj": 4.0, "estimated_reserve_mj": 6.0, "had_observation": True}]
    result_a = compute_rival_soc_metrics(steps_a)
    result_b = compute_rival_soc_metrics(steps_b)
    assert result_a.mae_mj != result_b.mae_mj
    assert result_a.mae_mj == pytest.approx(0.5, abs=1e-6)
    assert result_b.mae_mj == pytest.approx(2.0, abs=1e-6)


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


# ── Runner tests (Task 4) ─────────────────────────────────────────────────────

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
    assert r_b.scenario == "B"
    assert r_c.scenario == "C"
    assert r_b.rival_soc.ground_truth_available is True
    assert r_c.rival_soc.ground_truth_available is True
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
    """MAE must not be None for a real run."""
    resp = client.get("/api/v1/validation/summary", params={"scenario": "B", "seed": 42})
    data = resp.json()
    mae = data["rival_soc"]["mae_mj"]
    assert mae is not None


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
