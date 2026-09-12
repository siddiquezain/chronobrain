"""
Tests for RivalIntentBlock in DecisionSnapshot.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_INTENTS = {"DEPLOYING", "CONSERVING", "HARVESTING", "DEFENDING", "UNCERTAIN"}
VALID_CAPABILITIES = {"CAN_COUNTER", "CANNOT_COUNTER", "UNCERTAIN"}


def _snap(scenario: str = "B", seed: int = 42) -> dict:
    resp = client.post(
        "/api/v1/decision",
        json={"source": "synthetic", "scenario": scenario, "seed": seed},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_rival_intent_key_present():
    """rival_intent key must exist in the API response."""
    snap = _snap("B")
    assert "rival_intent" in snap, f"rival_intent key missing. Keys: {list(snap.keys())}"


def test_rival_intent_schema():
    """rival_intent block must have all required fields."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    assert ri is not None, "rival_intent is None — expected a block for synthetic scenario B"
    for field in ("intent", "intent_confidence", "response_capability", "trap_probability", "intent_evidence"):
        assert field in ri, f"Missing field: {field}"


def test_intent_is_valid_value():
    """intent must be one of the 5 canonical values."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert ri["intent"] in VALID_INTENTS, f"Invalid intent: {ri['intent']}"


def test_intent_confidence_in_range():
    """intent_confidence must be 0.0 – 1.0."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    conf = ri["intent_confidence"]
    assert 0.0 <= conf <= 1.0, f"intent_confidence out of range: {conf}"


def test_response_capability_is_valid():
    """response_capability must be CAN_COUNTER, CANNOT_COUNTER, or UNCERTAIN."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert ri["response_capability"] in VALID_CAPABILITIES, \
        f"Invalid response_capability: {ri['response_capability']}"


def test_trap_probability_in_range():
    """trap_probability must be 0.0 – 1.0."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    tp = ri["trap_probability"]
    assert 0.0 <= tp <= 1.0, f"trap_probability out of range: {tp}"


def test_intent_evidence_is_non_empty_string():
    """intent_evidence must be a non-empty string."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    ev = ri["intent_evidence"]
    assert isinstance(ev, str) and len(ev) > 10, f"intent_evidence too short: {ev!r}"


def test_intent_evidence_contains_modeled_label():
    """intent_evidence must say MODELED."""
    snap = _snap("B")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert "MODELED" in ri["intent_evidence"].upper(), \
        f"intent_evidence missing 'MODELED' label: {ri['intent_evidence']}"


def test_all_scenarios_produce_rival_intent():
    """All 5 synthetic scenarios must produce a rival_intent block."""
    for scenario in ("A", "B", "C", "D", "E"):
        snap = _snap(scenario)
        ri = snap.get("rival_intent")
        assert ri is not None, f"rival_intent is None for scenario {scenario}"
        assert ri["intent"] in VALID_INTENTS, f"Scenario {scenario}: invalid intent {ri['intent']}"
        assert ri["response_capability"] in VALID_CAPABILITIES
        assert 0.0 <= ri["trap_probability"] <= 1.0
        assert 0.0 <= ri["intent_confidence"] <= 1.0


def test_scenario_c_low_soc_cannot_counter():
    """
    Scenario C has very low rival SoC (~1.8 MJ).
    With high p_low, response_capability should be CANNOT_COUNTER or UNCERTAIN.
    """
    snap = _snap("C")
    ri = snap.get("rival_intent")
    if ri is None:
        return
    assert ri["response_capability"] in ("CANNOT_COUNTER", "UNCERTAIN"), \
        f"Scenario C (low SoC) should give CANNOT_COUNTER or UNCERTAIN, got {ri['response_capability']}"


def test_rival_intent_deterministic():
    """Same scenario + seed must produce identical rival_intent."""
    r1 = _snap("B", seed=42)
    r2 = _snap("B", seed=42)
    ri1 = r1.get("rival_intent")
    ri2 = r2.get("rival_intent")
    assert ri1 is not None and ri2 is not None
    assert ri1["intent"] == ri2["intent"]
    assert ri1["intent_confidence"] == ri2["intent_confidence"]
    assert ri1["response_capability"] == ri2["response_capability"]
    assert ri1["trap_probability"] == ri2["trap_probability"]


def test_rival_intent_not_inside_rival_block():
    """rival_intent must be a top-level snapshot key, not nested inside rival."""
    snap = _snap("B")
    rival = snap.get("rival", {})
    assert "rival_intent" not in rival, "rival_intent should not be inside the rival block"
    assert "intent" not in rival, "intent field should not be inside the rival block"


def test_rival_intent_key_in_openapi():
    """rival_intent must appear in the OpenAPI schema for DecisionSnapshot."""
    from fastapi.testclient import TestClient
    tc = TestClient(app)
    resp = tc.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    schemas = spec.get("components", {}).get("schemas", {})
    assert "RivalIntentBlock" in schemas, "RivalIntentBlock missing from OpenAPI schemas"
    ds_props = schemas.get("DecisionSnapshot", {}).get("properties", {})
    assert "rival_intent" in ds_props, "rival_intent missing from DecisionSnapshot schema"
