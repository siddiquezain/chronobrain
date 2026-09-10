"""
Targeted regressions for the TrackShift 2026 pre-demo audit.

Each test maps to a numbered issue in the audit brief. Cache-gated tests use the
real 2024 Monza replay.
"""

from __future__ import annotations

import json

import pytest

from app.data.fastf1_service import FastF1Unavailable, fastf1_available

_HAVE_FASTF1 = fastf1_available()
_needs = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")


@pytest.fixture(scope="module")
def monza():
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    from app.replay import run_historical_replay
    from app.replay.historical import clear_session_cache
    from app.data.fastf1_service import clear_field_timeline_cache
    clear_session_cache(); clear_field_timeline_cache()
    try:
        return run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
    except FastF1Unavailable as exc:
        pytest.skip(f"session unavailable: {exc}")


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# ISSUE 3 + 4 — rival changes lap by lap, and we can prove WHY
# ---------------------------------------------------------------------------
@_needs
def test_issue3_strategic_rival_is_not_hardcoded(monza):
    tracked = set(monza["strategic_rival"]["drivers_tracked"])
    assert len(tracked) >= 3
    assert monza["strategic_rival"]["tick_level"] is True


@_needs
def test_issue4_debug_timeline_exposes_why_a_switch_happened():
    c = _client()
    tl = c.get("/api/v1/replay/historical/2024_italian_gp/13/timeline?debug=true&max_ticks=200").json()
    assert isinstance(tl["debug"], list) and tl["debug"], "no debug rows"
    reasons = set()
    for row in tl["debug"]:
        assert {"selected", "switch_reason", "candidates"} <= set(row)
        reasons.add(row["switch_reason"])
        for cand in row["candidates"]:
            assert {"driver", "relevance", "gap_s", "ahead", "closing_rate_s_per_lap",
                    "positions_apart", "position"} <= set(cand)
        # the selected driver must be one of the candidates
        assert any(cd["driver"] == row["selected"] for cd in row["candidates"])
    # at least one of the meaningful reasons appears over a full lap
    assert reasons & {"margin_exceeded", "hysteresis_held", "dwell_not_met", "held", "initial"}
    # every number is real (no obviously-hardcoded constant repeated on every tick)
    rels = [cd["relevance"] for row in tl["debug"] for cd in row["candidates"] if cd["relevance"]]
    assert len(set(rels)) > 5


@_needs
def test_issue4_debug_off_by_default_keeps_response_small():
    c = _client()
    tl = c.get("/api/v1/replay/historical/2024_italian_gp/13/timeline").json()
    assert isinstance(tl["debug"], str)   # a hint string, not the payload


# ---------------------------------------------------------------------------
# ISSUE 5 — no future leakage into rival energy inference OR the decision
# ---------------------------------------------------------------------------
@_needs
def test_issue5_future_telemetry_cannot_change_rival_energy_or_decision():
    import fastf1
    from app.replay.field_state import FieldTimeline  # noqa: F401 (ensures module import)
    from app.replay.historical import _load_race_laps, _censored_provider, clear_session_cache
    from app.replay.races import resolve_race
    from app.decision.engine import run_decision
    from app.decision.config import DecisionConfig
    from app.data.fastf1_service import clear_field_timeline_cache

    clear_session_cache(); clear_field_timeline_cache()
    race = resolve_race("2024_italian_gp")
    full = _load_race_laps(race, "LEC", "PIA", ".fastf1_cache", True)
    cfg = DecisionConfig(seed=42)

    def snap_at(n):
        s = run_decision(_censored_provider(full, n, race.name), lap=n, config=cfg).model_dump()
        s["meta"].pop("generated_at", None)
        return s

    before = snap_at(20)
    for nl in full:
        if nl.lap > 20:
            nl.our_soc_mj = 9.0
            nl.rival_terminal_speed_kmh = 999.0
            nl.gap_to_car_ahead_s = 0.001
            if nl.strategic_rival is not None:
                nl.strategic_rival.driver = "XXX"
    after = snap_at(20)

    assert before["rival"] == after["rival"], "future data changed the rival energy inference at lap 20"
    assert before["decision"] == after["decision"], "future data changed the lap-20 decision"
    assert before["energy"] == after["energy"]
    clear_session_cache(); clear_field_timeline_cache()


# ---------------------------------------------------------------------------
# ISSUE 6 — identity-aware estimator bank (PIA -> NOR -> PIA)
# ---------------------------------------------------------------------------
def test_issue6_returning_rival_uses_only_its_own_evidence():
    from app.data.providers import ReplayProvider
    from app.data.samples import NormalizedLap, StrategicRivalInfo
    from app.decision import DecisionConfig
    from app.decision.engine import run_pipeline

    def lap(i, drv, term):
        return NormalizedLap(
            lap=i, total_laps=40, data_mode="REPLAY", our_speed_kmh=330.0,
            our_soc_mj=5.0, our_lap_start_soc_mj=5.0, our_lap_energy_deployed_mj=1.5,
            gap_to_car_ahead_s=0.6,
            strategic_rival=StrategicRivalInfo(driver=drv, role="ATTACK_TARGET",
                                               position=1, gap_s=0.6, ahead=True, relevance_score=0.9),
            rival_terminal_speed_kmh=term, rival_clipping_point_fraction=0.4,
            rival_corner_exit_accel_g=1.6, rival_sector_delta_s=-0.1,
            energy_is_modeled=True, raw_sample_count=200,
        )

    laps = [lap(i, "PIA", 300.0) for i in range(1, 5)]
    laps += [lap(i, "NOR", 340.0) for i in range(5, 9)]
    laps += [lap(i, "PIA", 300.0) for i in range(9, 12)]
    ctx = run_pipeline(ReplayProvider(laps), lap=11, config=DecisionConfig(seed=42))

    assert ctx._estimators["PIA"].observation_count == 7   # 4 early + 3 late, its own only
    assert ctx._estimators["NOR"].observation_count == 4
    # the surfaced estimate is PIA's, built from 7 PIA observations
    assert ctx.rival.estimate.n_observations == 7


# ---------------------------------------------------------------------------
# ISSUE 8 — MGU-K power is modelled + verified, not a hardcoded string
# ---------------------------------------------------------------------------
@_needs
def test_issue8_mgu_k_power_is_modelled_and_compliance_verified():
    c = _client()
    snap = c.get("/api/v1/replay/historical/2024_italian_gp/13?full_snapshot=true").json()["snapshot"]
    e = snap["energy"]
    assert e["mgu_k_power_ceiling_kw"] == 350.0
    assert e["modeled_mgu_k_peak_kw"] is not None
    assert e["modeled_mgu_k_peak_kw"] <= e["mgu_k_power_ceiling_kw"]
    chk = next(x for x in snap["compliance"]["checks"] if "MGU-K" in x["rule"])
    assert chk["status"] in ("pass", "breach")            # a real verdict, not "info"
    assert "modelled from throttle trace" in chk["detail"]
    assert chk["provenance"] == "VERIFIED_FIA"


# ---------------------------------------------------------------------------
# ISSUE 9 — telemetry cadence metadata is honest (~4 Hz, not 128 Hz)
# ---------------------------------------------------------------------------
@_needs
def test_issue9_cadence_metadata_is_real_and_not_128hz(monza):
    t = monza["telemetry"]
    assert t["is_synthetic"] is False and t["is_live"] is False
    assert 1.0 <= t["cadence_hz_measured"] <= 20.0        # FastF1 reality, ~4 Hz
    assert t["cadence_hz_measured"] != 128
    assert "NOT a 128 Hz" in t["cadence_note"]            # the only place "128" may appear


def test_issue9_backend_never_emits_128hz_anywhere():
    # Check that the cadence-rate string "128" (as in "128 Hz") never appears
    # in the production decision response. Use a phrase-level check rather than
    # bare "128" to avoid false positives from unrelated float values like 1.6128.
    c = _client()
    blob = json.dumps(c.post("/api/v1/decision", json={"scenario": "B", "seed": 42, "lap": 20}).json())
    blob_lower = blob.lower()
    assert "128hz" not in blob_lower and "128 hz" not in blob_lower


# ---------------------------------------------------------------------------
# ISSUE 10 — POSITION_BATTLE only for a genuine close fight
# ---------------------------------------------------------------------------
@_needs
def test_issue10_position_battle_never_at_a_large_gap(monza):
    from app.replay.strategic_rival import RivalSelectorConfig
    cap = RivalSelectorConfig().position_battle_max_gap_s
    for L in monza["laps"]:
        s = L["strategic_rival"]
        if s["role"] == "POSITION_BATTLE":
            assert s["gap_s"] is not None and s["gap_s"] <= cap + 1e-6, \
                f"lap {L['lap']}: POSITION_BATTLE at {s['gap_s']}s (> {cap}s)"


# ---------------------------------------------------------------------------
# ISSUE 11 — 2024 telemetry / 2026 model distinction is explicit
# ---------------------------------------------------------------------------
@_needs
def test_issue11_replay_distinguishes_2024_observation_from_2026_model(monza):
    note = monza["provenance"]["note"]
    assert "HISTORICAL TELEMETRY REPLAY" in note and "CHRONOPACE 2026 MODEL" in note
    assert "did NOT run" in note or "did not run" in note.lower()
    lp = monza["laps"][10]
    assert "REAL (2024 FastF1 observation)" in lp["provenance"]
    assert "MODELED (ChronoPace 2026, MODEL_ASSUMPTION)" in lp["provenance"]
    assert "INFERRED (probabilistic, from observable performance)" in lp["provenance"]


# ---------------------------------------------------------------------------
# ISSUE 12 — the five deployment modes are all present
# ---------------------------------------------------------------------------
def test_issue12_five_deployment_modes_available_to_monte_carlo_and_gate():
    from rule_gate import DeploymentMode, RegulatoryGate
    from telemetry_simulator import TelemetryInput
    from planner import MonteCarloPlanner, PlanningContext

    assert {m.value for m in DeploymentMode} == {
        "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
        "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
    }
    ti = TelemetryInput(lap_number=20, current_soc_mj=7.0, lap_start_soc_mj=7.2,
                        lap_energy_deployed_mj=1.0, gap_to_car_ahead_s=0.5,
                        overtake_qualified_last_lap=True, speed_kmh=300.0, total_laps=50)
    gate = RegulatoryGate().evaluate(ti)
    assert set(gate.legal_modes) == set(DeploymentMode)   # all five legal in a clean scenario
    res = MonteCarloPlanner(seed=42).plan(gate, PlanningContext(gap_to_car_ahead_s=0.5))
    assert {p.mode for p in res.ranked_modes} == set(DeploymentMode)


@_needs
def test_issue12_replay_uses_all_five_modes_over_a_race(monza):
    modes = {L["mode"] for L in monza["laps"]}
    # every mode ChronoPace picks must be one of the five
    assert modes <= {"CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
                     "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE"}
    # and Monte Carlo ranks all five every lap
    c = _client()
    snap = c.get("/api/v1/replay/historical/2024_italian_gp/20?full_snapshot=true").json()["snapshot"]
    assert {m["mode"] for m in snap["monte_carlo"]["ranked_modes"]} == {
        "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
        "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
    }


# ---------------------------------------------------------------------------
# ISSUE 13 — decision-engine integrity / LLM narrates verified JSON only
# ---------------------------------------------------------------------------
@_needs
def test_issue13_decision_pipeline_order_and_llm_boundary(monza):
    c = _client()
    snap = c.get("/api/v1/replay/historical/2024_italian_gp/20?full_snapshot=true").json()["snapshot"]
    stages = [s["stage"] for s in snap["trace"]]
    for required in ("Regulatory gate", "Monte Carlo planner", "Confidence gate", "Decision engine"):
        assert required in stages
    assert stages.index("Regulatory gate") < stages.index("Monte Carlo planner") < stages.index("Confidence gate")
    # numeric decision fields come from the deterministic pipeline; narrative is optional prose
    assert isinstance(snap["decision"]["mode"], str)
    assert isinstance(snap["monte_carlo"]["ranked_modes"][0]["mean_laptime_delta_s"], (int, float))
    assert snap.get("narrative") in (None, "") or isinstance(snap["narrative"], str)


# ---------------------------------------------------------------------------
# ISSUE 14 — historical replay uses real FastF1 telemetry (not hardcoded)
# ---------------------------------------------------------------------------
@_needs
def test_issue14_replay_telemetry_is_real_and_varies(monza):
    speeds = [L["our_speed_kmh"] for L in monza["laps"] if L["our_speed_kmh"]]
    throttles = [L["telemetry_real"]["mean_throttle"] for L in monza["laps"]
                 if L["telemetry_real"]["mean_throttle"] is not None]
    assert len(set(round(s, 1) for s in speeds)) >= 8      # real speed traces differ per lap
    assert len(set(round(t, 3) for t in throttles)) >= 8
    assert monza["source"] == "fastf1_historical_replay"
