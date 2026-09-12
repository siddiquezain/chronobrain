"""Tests for ContextAttributionBlock in DecisionSnapshot."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _snapshot(scenario: str = "B") -> dict:
    resp = client.post("/api/v1/decision", json={"source": "synthetic", "scenario": scenario, "seed": 42})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_context_attribution_key_present():
    """context_attribution key must be present in the response (may be null)."""
    snap = _snapshot("B")
    assert "context_attribution" in snap, "context_attribution key missing from API response"


def test_context_attribution_schema_when_present():
    """When not None, all expected fields must be present."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is not None:
        required = {
            "rival_tyre_compound", "compound_baseline_active",
            "observed_sector_delta_s", "context_explained_note",
            "active_aero_mode", "residual_evidence_confidence"
        }
        assert required.issubset(ca.keys()), f"Missing keys: {required - ca.keys()}"


def test_active_aero_mode_valid():
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is not None:
        assert ca["active_aero_mode"] in (
            "OVERTAKE_ELIGIBLE", "STRAIGHT_MODE", "CORNER_MODE", "UNKNOWN"
        )


def test_residual_evidence_confidence_valid():
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is not None:
        assert ca["residual_evidence_confidence"] in ("HIGH", "MEDIUM", "LOW", "UNAVAILABLE")


def test_context_explained_note_is_string():
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is not None:
        assert isinstance(ca["context_explained_note"], str)


def test_compound_baseline_active_is_bool():
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is not None:
        assert isinstance(ca["compound_baseline_active"], bool)


def test_scenario_b_has_overtake_eligible_active_aero():
    """Scenario B: gap=0.6s (< 1.0s) so active_aero_mode should be OVERTAKE_ELIGIBLE."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is not None:
        assert ca["active_aero_mode"] == "OVERTAKE_ELIGIBLE", \
            f"Scenario B gap=0.6s should give OVERTAKE_ELIGIBLE, got {ca['active_aero_mode']}"


def test_cause_attribution_present():
    """cause_attribution dict must be present in context_attribution block."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    assert ca is not None, "context_attribution is None"
    assert "cause_attribution" in ca, f"cause_attribution missing from context_attribution: {list(ca.keys())}"


def test_cause_attribution_sums_to_one():
    """cause_attribution values must sum to 1.0 (± floating point)."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is None or ca.get("cause_attribution") is None:
        return
    d = ca["cause_attribution"]
    total = sum(d.values())
    assert abs(total - 1.0) < 0.01, f"cause_attribution values should sum to 1.0, got {total}: {d}"


def test_cause_attribution_has_correct_keys():
    """cause_attribution must have tyre, energy, traffic_aero, other keys."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is None or ca.get("cause_attribution") is None:
        return
    d = ca["cause_attribution"]
    required_keys = {"tyre", "energy", "traffic_aero", "other"}
    assert required_keys == set(d.keys()), f"Expected keys {required_keys}, got {set(d.keys())}"


def test_cause_attribution_values_non_negative():
    """All cause_attribution values must be >= 0."""
    snap = _snapshot("B")
    ca = snap.get("context_attribution")
    if ca is None or ca.get("cause_attribution") is None:
        return
    d = ca["cause_attribution"]
    for k, v in d.items():
        assert v >= 0, f"cause_attribution[{k}] = {v} is negative"


def test_cause_attribution_all_scenarios():
    """cause_attribution must be present for all scenarios A–E."""
    for scenario in ("A", "B", "C", "D", "E"):
        snap = _snapshot(scenario)
        ca = snap.get("context_attribution")
        assert ca is not None, f"context_attribution is None for scenario {scenario}"
        assert "cause_attribution" in ca, f"cause_attribution missing for scenario {scenario}"
        d = ca["cause_attribution"]
        assert abs(sum(d.values()) - 1.0) < 0.01, f"Scenario {scenario}: values don't sum to 1.0: {d}"
