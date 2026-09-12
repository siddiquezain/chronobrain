"""
Interactive demo / manual-control layer tests.

The user changes INPUTS; the deterministic Python pipeline computes the
recommendation. Ground truth is demo-only and never in POST /api/v1/decision.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.data.providers import ReplayProvider, build_provider
from app.data.samples import NormalizedLap
from app.decision import DecisionConfig
from app.decision.engine import assemble_snapshot, run_pipeline
from app.demo import InputOverrides, build_rival_validation
from app.main import app
from telemetry_simulator import RivalObservation, TelemetrySimulator

client = TestClient(app)

_MODES = {
    "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
    "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
}


# --- 9. manual input changes propagate through the whole pipeline --------------
def test_manual_input_change_changes_the_decision():
    hi = client.post("/api/v1/decision", json={"scenario": "B", "seed": 42, "lap": 25}).json()
    lo = client.post("/api/v1/decision", json={
        "scenario": "B", "seed": 42, "lap": 25,
        "overrides": {"initial_soc_mj": 1.4},
    }).json()
    assert hi["decision"]["action"] == "ATTACK_NOW"
    assert lo["decision"]["mode"] in ("CONSERVE_MODE", "BALANCED_MODE")
    assert lo["decision"]["action"] in ("CONSERVE", "HOLD") or lo["decision"]["action"].startswith("WAIT_")
    # the override actually reached the sim
    assert lo["energy"]["soc_mj"] < hi["energy"]["soc_mj"]


def test_manual_gap_override_changes_legality():
    close = client.post("/api/v1/decision", json={
        "scenario": "A", "seed": 42, "lap": 20, "overrides": {"gap_to_car_ahead_s": 0.5},
    }).json()
    far = client.post("/api/v1/decision", json={
        "scenario": "A", "seed": 42, "lap": 20, "overrides": {"gap_to_car_ahead_s": 3.0},
    }).json()
    assert "ARM_OVERTAKE_MODE" in close["compliance"]["legal_modes"]
    assert "ARM_OVERTAKE_MODE" not in far["compliance"]["legal_modes"]


def test_the_frontend_cannot_set_a_deployment_mode():
    """There is no request field that sets decision.mode."""
    from app.api.routes.v1.demo import DemoDecisionRequest
    from app.api.routes.v1.decision import DecisionRequest

    for model in (DemoDecisionRequest, DecisionRequest):
        fields = set(model.model_fields)
        assert not fields & {"mode", "recommended_mode", "decision", "action", "deployment_mode"}


# --- 6. determinism ---------------------------------------------------------
def test_demo_decision_is_deterministic():
    body = {"scenario": "B", "seed": 42, "lap": 20, "overrides": {"rival_initial_soc_mj": 7.5}}
    a = client.post("/api/v1/demo/decision", json=body).json()
    b = client.post("/api/v1/demo/decision", json=body).json()
    a["snapshot"]["meta"]["generated_at"] = b["snapshot"]["meta"]["generated_at"] = "x"
    assert a == b


# --- 7 & 8. ground truth: only in demo mode -------------------------------
def test_ground_truth_never_in_production_decision():
    r = client.post("/api/v1/decision", json={
        "scenario": "B", "seed": 42, "lap": 25, "overrides": {"rival_initial_soc_mj": 8.0},
    })
    blob = json.dumps(r.json()).lower()
    assert "ground_truth" not in blob
    assert "rival_validation" not in blob
    rival = r.json()["rival"]
    # the estimator contract is intact
    assert {
        "mean_reserve_mj", "reserve_std_mj", "n_observations", "confidence",
        "estimate_uncertain", "bucket", "distribution", "freshness_laps", "p_defend",
    } <= set(rival)
    # strategic-rival fields exist but are inert on the synthetic path (no field data)
    assert rival["driver"] is None and rival["role"] is None and rival["relevance_score"] is None
    assert "actual" not in json.dumps(rival).lower() and "hidden" not in json.dumps(rival).lower()


def test_ground_truth_present_only_in_demo_response():
    r = client.post("/api/v1/demo/decision", json={
        "scenario": "B", "seed": 42, "lap": 25, "overrides": {"rival_initial_soc_mj": 8.0},
    }).json()
    rv = r["rival_validation"]
    assert rv["enabled"] is True
    assert rv["ground_truth_reserve_mj"] is not None
    assert rv["error_mj"] == round(rv["estimated_reserve_mj"] - rv["ground_truth_reserve_mj"], 4)
    assert "DEMO ONLY" in rv["label"]


# --- 3. estimator does not receive ground truth ---------------------------
def test_rival_observation_carries_no_soc():
    assert "soc" not in RivalObservation.model_fields
    assert "soc_mj" not in RivalObservation.model_fields
    obs_fields = set(RivalObservation.model_fields)
    assert obs_fields == {
        "terminal_speed_kmh", "clipping_point_fraction",
        "corner_exit_accel_g", "sector_delta_s",
    }


def test_normalized_lap_carries_no_rival_soc():
    assert not any("rival" in f and "soc" in f for f in NormalizedLap.model_fields)


def test_thread_state_never_touches_hidden_rival_soc():
    import inspect
    from app.decision import engine

    src = inspect.getsource(engine._thread_state) + inspect.getsource(engine._rival_features)
    assert "_rival_soc_mj" not in src
    assert "ground_truth" not in src


# --- 1. different hidden rival energy -> different telemetry evidence -----
def test_different_hidden_soc_produces_different_observations():
    def obs_seq(rsoc):
        sim = TelemetrySimulator("B", seed=42, total_laps=12,
                                 preset_overrides={"rival_initial_soc_mj": rsoc})
        return [o for _, o in sim.generate_sequence(12)]

    hi = obs_seq(8.0)
    lo = obs_seq(1.0)
    # higher hidden SoC -> higher terminal speed / later clipping on average
    assert sum(o.terminal_speed_kmh for o in hi) > sum(o.terminal_speed_kmh for o in lo)
    assert sum(o.clipping_point_fraction for o in hi) > sum(o.clipping_point_fraction for o in lo)


# --- 10. synthetic and FastF1 replay share the SAME downstream pipeline ---
def test_replay_and_synthetic_run_the_identical_pipeline():
    """A ReplayProvider (the FastF1 output type) goes through run_pipeline exactly
    as SyntheticProvider does — same stages, same context shape."""
    laps = [
        NormalizedLap(
            lap=i, total_laps=50, data_mode="REPLAY",
            our_speed_kmh=290.0, our_soc_mj=5.0, our_lap_start_soc_mj=5.1,
            our_lap_energy_deployed_mj=1.5, gap_to_car_ahead_s=0.7,
            rival_terminal_speed_kmh=305.0, rival_clipping_point_fraction=0.45,
            rival_corner_exit_accel_g=1.05, rival_sector_delta_s=-0.05,
            energy_is_modeled=True,
        )
        for i in range(1, 13)
    ]
    replay_ctx = run_pipeline(ReplayProvider(laps), lap=12, config=DecisionConfig(seed=42))
    synth_ctx = run_pipeline(
        build_provider("synthetic", scenario="B", seed=42, total_laps=50),
        lap=12, config=DecisionConfig(seed=42),
    )
    for ctx in (replay_ctx, synth_ctx):
        assert ctx.rival is not None and ctx.confidence_result is not None
        assert assemble_snapshot(ctx, "x").decision.mode in _MODES
    # ground truth only exists for the synthetic provider
    rv_replay = build_rival_validation(replay_ctx, ReplayProvider(laps))
    assert rv_replay["ground_truth_reserve_mj"] is None


# --- 12. five deployment modes remain reachable --------------------------
def test_five_modes_still_reachable_after_demo_layer():
    from tests.test_arm_mode import test_all_five_modes_are_reachable
    test_all_five_modes_are_reachable()  # re-run the existing coverage
