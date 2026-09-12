"""POST /api/v1/decision — the one authoritative snapshot endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_decision_returns_full_snapshot():
    r = client.post("/api/v1/decision", json={"scenario": "B", "seed": 42, "total_laps": 50, "lap": 15})
    assert r.status_code == 200
    body = r.json()
    # everything the frontend needs, in one response
    for key in ("meta", "decision", "data_quality", "window", "energy", "rival",
                "opportunity", "monte_carlo", "compliance", "confidence", "constraints",
                "candidate_actions", "feasible_actions", "rejected_alternatives",
                "reason_codes", "reasons", "trace"):
        assert key in body, f"missing snapshot key: {key}"
    assert body["decision"]["mode"] in {
        "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
        "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
    }
    assert body["decision"]["action"] in {
        "ATTACK_NOW", "WAIT_2_LAPS", "WAIT_5_LAPS", "HOLD", "PUSH", "CONSERVE",
    }
    assert body["monte_carlo"]["seed"] == 42
    assert body["meta"]["config_fingerprint"]


def test_legacy_endpoints_are_marked_deprecated():
    """The frontend must not treat Stack B endpoints as authoritative."""
    schema = client.get("/openapi.json").json()
    for path in ("/api/strategy/recommendation", "/api/energy/state",
                 "/api/overtake/current", "/api/race/state"):
        for method in schema["paths"][path].values():
            assert method.get("deprecated") is True, f"{path} not marked deprecated in OpenAPI"
    # and a runtime Deprecation header (404 body, but the header still lands)
    r = client.get("/api/strategy/recommendation")
    assert r.headers.get("Deprecation") == "true"
    assert "/api/v1/decision" in r.headers.get("Link", "")


def test_openapi_json_is_reachable():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["openapi"].startswith("3.")
    assert "/api/v1/decision" in r.json()["paths"]


def test_decision_is_deterministic_over_http():
    payload = {"scenario": "B", "seed": 42, "total_laps": 50, "lap": 15}
    a = client.post("/api/v1/decision", json=payload).json()
    b = client.post("/api/v1/decision", json=payload).json()
    a["meta"]["generated_at"] = b["meta"]["generated_at"] = "x"
    assert a == b


def test_scenario_e_endpoint_reports_illegal():
    r = client.get("/api/v1/scenario/E?lap=30")
    assert r.status_code == 200
    assert r.json()["compliance"]["legal"] is False


def test_unknown_scenario_is_400():
    r = client.post("/api/v1/decision", json={"scenario": "Z"})
    assert r.status_code == 400


def test_fastf1_without_spec_is_422():
    r = client.post("/api/v1/decision", json={"source": "fastf1"})
    assert r.status_code in (400, 422)


def test_v1_health_reports_fastf1_flag():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert "fastf1_available" in r.json()
