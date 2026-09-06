"""
Rival Energy Estimator VALIDATION demo — the estimator responds to changing
hidden evidence and keeps honest uncertainty. Not a calibration claim.
"""

import pytest
from fastapi.testclient import TestClient

from app.demo import rival_estimator_trace
from app.main import app

client = TestClient(app)


# --- 2. estimator moves in the correct direction -------------------------
def test_posterior_moves_with_the_hidden_state_same_visible_gap():
    """Same gap (0.6 s), similar speed profile — only the HIDDEN rival SoC differs."""
    hi = rival_estimator_trace(
        scenario="B", seed=42, up_to_lap=20,
        preset_overrides={"rival_initial_soc_mj": 7.5, "gap_to_car_ahead_s": 0.6},
    )
    lo = rival_estimator_trace(
        scenario="B", seed=42, up_to_lap=20,
        preset_overrides={"rival_initial_soc_mj": 2.0, "gap_to_car_ahead_s": 0.6},
    )
    hi_final = hi["steps"][-1]["estimated_reserve_mj"]
    lo_final = lo["steps"][-1]["estimated_reserve_mj"]
    assert hi_final > lo_final + 1.0, (hi_final, lo_final)
    # the estimator never saw the ground truth, but it's tracked for scoring
    assert hi["ground_truth_available"] is True
    assert hi["steps"][-1]["ground_truth_reserve_mj"] > lo["steps"][-1]["ground_truth_reserve_mj"]


# --- 4. maintains uncertainty (does not collapse) ------------------------
def test_estimator_keeps_uncertainty():
    tr = rival_estimator_trace(scenario="B", seed=42, up_to_lap=25)
    stds = [s["estimated_std_mj"] for s in tr["steps"] if s["had_observation"]]
    assert min(stds) >= 0.35            # the reported floor
    assert tr["steps"][-1]["estimated_std_mj"] < 2.6   # but it IS informed
    ess = tr["final_particle_summary"]["effective_sample_size"]
    assert 50 < ess <= 1000             # not degenerate, not all-equal


# --- 5. sequential observations update the posterior ---------------------
def test_sequential_updates_change_the_posterior():
    tr = rival_estimator_trace(scenario="B", seed=42, up_to_lap=15,
                               preset_overrides={"rival_initial_soc_mj": 8.0})
    means = [s["estimated_reserve_mj"] for s in tr["steps"]]
    n_obs = [s["n_observations"] for s in tr["steps"]]
    assert n_obs == list(range(1, 16))          # one update per lap
    assert len(set(round(m, 2) for m in means)) > 5   # the mean actually moves
    # early estimate is closer to the uniform-prior midpoint than the late one
    assert abs(means[0] - 4.5) > abs(means[-1] - tr["steps"][-1]["ground_truth_reserve_mj"]) - 3.0


def test_uncertainty_trend_is_reported():
    tr = rival_estimator_trace(scenario="B", seed=42, up_to_lap=12)
    trends = {s["uncertainty_trend"] for s in tr["steps"]}
    assert trends <= {"flat", "more_certain", "less_certain"}
    assert any(t != "flat" for t in trends)


# --- 6. determinism -----------------------------------------------------
def test_rival_trace_is_deterministic():
    a = rival_estimator_trace(scenario="B", seed=42, up_to_lap=18)
    b = rival_estimator_trace(scenario="B", seed=42, up_to_lap=18)
    assert a == b


# --- compact particle representation ----------------------------------
def test_particle_summary_is_compact():
    tr = rival_estimator_trace(scenario="B", seed=42, up_to_lap=10)
    ps = tr["final_particle_summary"]
    assert set(ps["percentiles"]) == {"p05", "p25", "p50", "p75", "p95"}
    assert len(ps["histogram"]["weights"]) == 12          # bins, not 1000 particles
    assert abs(sum(ps["histogram"]["weights"]) - 1.0) < 1e-3   # rounded to 5dp
    assert ps["percentiles"]["p05"] <= ps["percentiles"]["p50"] <= ps["percentiles"]["p95"]


# --- API endpoint -----------------------------------------------------
def test_rival_trace_endpoint_via_http():
    r = client.post("/api/v1/demo/rival-trace", json={
        "scenario": "B", "seed": 42, "up_to_lap": 8,
        "overrides": {"rival_initial_soc_mj": 3.0},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ground_truth_available"] is True
    assert "GROUND TRUTH — DEMO ONLY" in body["note"]
    assert len(body["steps"]) == 8
    assert all("estimated_reserve_mj" in s and "estimated_std_mj" in s for s in body["steps"])


def test_presets_endpoint_lists_the_five_scenarios():
    keys = {p["key"] for p in client.get("/api/v1/demo/presets").json()["presets"]}
    assert {
        "HIGH_ENERGY_STRONG_OPPORTUNITY",
        "LIMITED_ENERGY_SAME_OPPORTUNITY",
        "BETTER_FUTURE_OPPORTUNITY",
        "LOW_CONFIDENCE_BAD_DATA",
        "RIVAL_ENERGY_HIGH",
        "RIVAL_ENERGY_LOW",
    } <= keys


@pytest.mark.parametrize("key,expect_mode,expect_action", [
    ("HIGH_ENERGY_STRONG_OPPORTUNITY", "USE_OVERTAKE_BONUS_MODE", "ATTACK_NOW"),
    ("LIMITED_ENERGY_SAME_OPPORTUNITY", "CONSERVE_MODE", "CONSERVE"),
    ("BETTER_FUTURE_OPPORTUNITY", "CONSERVE_MODE", "WAIT_5_LAPS"),
    ("LOW_CONFIDENCE_BAD_DATA", "BALANCED_MODE", "HOLD"),
])
def test_presets_produce_engine_computed_decisions(key, expect_mode, expect_action):
    r = client.post(f"/api/v1/demo/preset/{key}").json()
    d = r["snapshot"]["decision"]
    assert (d["mode"], d["action"]) == (expect_mode, expect_action)
    if key == "LOW_CONFIDENCE_BAD_DATA":
        assert d["confidence_overridden"] is True
        assert "DATA_QUALITY" in d["override_reason"]


def test_rival_energy_pair_shows_estimator_response():
    hi = client.post("/api/v1/demo/preset/RIVAL_ENERGY_HIGH").json()["rival_validation"]
    lo = client.post("/api/v1/demo/preset/RIVAL_ENERGY_LOW").json()["rival_validation"]
    assert hi["estimated_reserve_mj"] > lo["estimated_reserve_mj"] + 1.0
    assert hi["ground_truth_reserve_mj"] > lo["ground_truth_reserve_mj"]
