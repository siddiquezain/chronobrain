"""
Tests for the Rival Energy Cross-Validation (RECV) harness.

RECV does NOT use FIA ground truth. It compares:
  A's particle-filter estimate of B
  against
  B's own NormalizedLap.our_soc_mj (ChronoPace ReplayEnergyModel reference)
"""
import math
import pytest
import numpy as np
from app.decision.recv_validation import (
    RecvCheckpoint,
    RecvReport,
    RecvMetrics,
    compute_recv_report,
    recv_baselines,
)


def _ckpt(lap: int, pred: float, ref: float, eq: str = "moderate",
          race: str = "test") -> RecvCheckpoint:
    return RecvCheckpoint(
        race=race, session="R", lap=lap,
        observer_driver="A", rival_driver="B",
        predicted_energy_mj=pred,
        predicted_std_mj=0.5,
        reference_energy_mj=ref,
        absolute_error_mj=abs(pred - ref),
        signed_error_mj=pred - ref,
        confidence=0.5, evidence_quality=eq,
        posterior_health="healthy", baseline_ready=True,
        n_observations=lap,
    )


# ---------------------------------------------------------------------------
# 1. RecvCheckpoint fields
# ---------------------------------------------------------------------------
def test_recv_checkpoint_absolute_error_computed_correctly():
    c = _ckpt(lap=39, pred=0.5, ref=5.4)
    assert c.absolute_error_mj == pytest.approx(4.9, abs=0.01)
    assert c.signed_error_mj == pytest.approx(-4.9, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Metrics computation
# ---------------------------------------------------------------------------
def test_report_mae_rmse_bias():
    ckpts = [_ckpt(i, pred, 5.0) for i, pred in enumerate(
        [4.0, 5.5, 3.5], start=1
    )]
    report = compute_recv_report(ckpts, label="test")
    assert report.mae_mj == pytest.approx((1.0 + 0.5 + 1.5) / 3, abs=0.01)
    assert report.rmse_mj >= report.mae_mj
    assert abs(report.bias_mj - ((-1.0 + 0.5 + (-1.5)) / 3)) < 0.01


def test_report_empty_checkpoints():
    report = compute_recv_report([], label="empty")
    assert report.n_checkpoints == 0
    assert math.isnan(report.mae_mj)


def test_report_coverage_computation():
    """All predictions exactly 0.4 MJ off, std = 0.5 MJ -> 100% within 1σ."""
    ckpts = [RecvCheckpoint(
        race="t", session="R", lap=i,
        observer_driver="A", rival_driver="B",
        predicted_energy_mj=5.0, predicted_std_mj=0.5,
        reference_energy_mj=5.4,
        absolute_error_mj=0.4, signed_error_mj=-0.4,
        confidence=0.5, evidence_quality="strong",
        posterior_health="healthy", baseline_ready=True,
        n_observations=i,
    ) for i in range(1, 11)]
    report = compute_recv_report(ckpts, label="t")
    assert report.coverage_68pct == pytest.approx(1.0, abs=0.01)
    assert report.coverage_95pct == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Honest label — NOT FIA SoC
# ---------------------------------------------------------------------------
def test_report_reference_description_not_fia():
    report = compute_recv_report([], label="test")
    desc = report.reference_description.lower()
    assert "chronopace" in desc or "model" in desc
    assert "not" in desc
    assert "fia" not in desc or ("not" in desc and desc.index("not") < desc.index("fia"))


# ---------------------------------------------------------------------------
# 4. Breakdown by evidence quality
# ---------------------------------------------------------------------------
def test_report_breaks_down_by_evidence_quality():
    ckpts = (
        [_ckpt(i, 4.5, 5.0, eq="weak") for i in range(1, 6)] +
        [_ckpt(i, 4.5, 5.0, eq="strong") for i in range(6, 16)]
    )
    report = compute_recv_report(ckpts, label="test")
    assert "weak" in report.by_evidence_quality
    assert "strong" in report.by_evidence_quality
    assert report.by_evidence_quality["weak"].n_checkpoints == 5
    assert report.by_evidence_quality["strong"].n_checkpoints == 10


def test_report_breaks_down_by_race():
    ckpts = (
        [_ckpt(i, 4.5, 5.0, race="race_a") for i in range(1, 6)] +
        [_ckpt(i, 4.5, 5.0, race="race_b") for i in range(6, 11)]
    )
    report = compute_recv_report(ckpts, label="test")
    assert "race_a" in report.by_race
    assert "race_b" in report.by_race


# ---------------------------------------------------------------------------
# 5. Baselines
# ---------------------------------------------------------------------------
def test_baselines_naive_prior():
    ckpts = [_ckpt(i, 4.0 + i * 0.1, 5.0) for i in range(1, 11)]
    bl = recv_baselines(ckpts)
    assert "naive_prior" in bl
    assert "persistence" in bl
    # Naive prior always predicts 4.5
    assert bl["naive_prior"].mae_mj == pytest.approx(0.5, abs=0.01)


def test_baselines_persistence_first_lap_is_repeated():
    """Persistence baseline repeats the FIRST prediction for lap 1."""
    ckpts = [_ckpt(i, 3.0 + i * 0.5, 5.0) for i in range(1, 6)]
    bl = recv_baselines(ckpts)
    # First prediction: 3.5, then 3.5 repeated for lap 1
    persist_ckpts = bl["persistence"].checkpoints
    assert persist_ckpts[0].predicted_energy_mj == pytest.approx(3.5, abs=0.01)
    # Lap 2 persistence = lap 1 prediction = 3.5
    assert persist_ckpts[1].predicted_energy_mj == pytest.approx(3.5, abs=0.01)


def test_baselines_empty():
    """Empty checkpoints yields empty baseline reports."""
    bl = recv_baselines([])
    assert "naive_prior" in bl
    assert "persistence" in bl
    assert bl["naive_prior"].n_checkpoints == 0
    assert bl["persistence"].n_checkpoints == 0
