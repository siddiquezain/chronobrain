"""API endpoint smoke tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_race_state_not_found():
    r = client.get("/api/race/state")
    assert r.status_code == 404


def test_energy_state_not_found():
    r = client.get("/api/energy/state")
    assert r.status_code == 404


def test_overtake_not_found():
    r = client.get("/api/overtake/current")
    assert r.status_code == 404


def test_strategy_not_found():
    r = client.get("/api/strategy/recommendation")
    assert r.status_code == 404


def test_simulation_status():
    r = client.get("/api/simulation/status")
    assert r.status_code == 200
    data = r.json()
    assert "running" in data
    assert data["running"] is False


def test_simulation_start_unknown_scenario():
    r = client.post("/api/simulation/start", json={"scenario": "Z"})
    assert r.status_code == 400


def test_simulation_stop_when_not_running():
    r = client.post("/api/simulation/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "not_running"
