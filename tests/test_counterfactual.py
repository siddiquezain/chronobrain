"""Tests for the CounterfactualBlock in DecisionSnapshot."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _snapshot(scenario: str = "B", seed: int = 42) -> dict:
    resp = client.post("/api/v1/decision", json={"source": "synthetic", "scenario": scenario, "seed": seed})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_counterfactual_schema_when_present():
    """When counterfactual is not None, it must have all required fields."""
    snap = _snapshot("B")
    cf = snap.get("counterfactual")
    if cf is not None:
        required = {
            "recommended_action", "counterfactual_action",
            "recommended_expected_gain_s", "counterfactual_expected_gain_s",
            "gain_delta_s", "energy_cost_recommended_mj",
            "energy_cost_counterfactual_mj", "future_opportunity_impact", "summary"
        }
        assert required.issubset(cf.keys()), f"Missing keys: {required - cf.keys()}"


def test_counterfactual_gain_delta_non_negative():
    """gain_delta_s must be >= 0 (recommended is at least as good as counterfactual)."""
    snap = _snapshot("B")
    cf = snap.get("counterfactual")
    if cf is not None:
        assert cf["gain_delta_s"] >= -0.01, f"Negative gain delta: {cf['gain_delta_s']}"


def test_counterfactual_actions_differ():
    """recommended_action and counterfactual_action must be different."""
    snap = _snapshot("B")
    cf = snap.get("counterfactual")
    if cf is not None:
        assert cf["recommended_action"] != cf["counterfactual_action"], \
            "Counterfactual must be a different action from the recommendation"


def test_counterfactual_energy_costs_non_negative():
    snap = _snapshot("B")
    cf = snap.get("counterfactual")
    if cf is not None:
        assert cf["energy_cost_recommended_mj"] >= 0.0
        assert cf["energy_cost_counterfactual_mj"] >= 0.0


def test_counterfactual_future_impact_valid():
    snap = _snapshot("B")
    cf = snap.get("counterfactual")
    if cf is not None:
        assert cf["future_opportunity_impact"] in ("BETTER", "SIMILAR", "WORSE")


def test_counterfactual_summary_non_empty_when_present():
    snap = _snapshot("B")
    cf = snap.get("counterfactual")
    if cf is not None:
        assert isinstance(cf["summary"], str) and len(cf["summary"]) > 0


def test_counterfactual_field_in_api_response():
    """'counterfactual' key must be present in the response (may be null)."""
    snap = _snapshot("B")
    assert "counterfactual" in snap, "counterfactual key missing from API response entirely"


def test_counterfactual_none_is_valid():
    """A None counterfactual is valid and must not cause errors."""
    snap = _snapshot("E")  # Scenario E: over deployment cap, all illegal
    assert snap.get("counterfactual") is None or isinstance(snap["counterfactual"], dict)
