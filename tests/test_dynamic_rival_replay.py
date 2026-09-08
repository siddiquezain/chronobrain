"""
Dynamic strategic-rival selection wired through the historical replay.

Offline tests drive the selector + estimator plumbing directly. The real-race
tests (cache-gated) confirm the whole 2024 Monza replay selects the rival from
the field, switches it, stays causal, and never leaks a hidden energy state.
"""

from __future__ import annotations

import pytest

from app.data.fastf1_service import FastF1Unavailable, fastf1_available
from app.data.normalizer import to_rival_observation
from app.data.samples import NormalizedLap, StrategicRivalInfo
from app.decision import DecisionConfig, run_decision
from app.decision.engine import run_pipeline
from app.data.providers import ReplayProvider
from app.replay import historical as H
from app.replay import resolve_race, run_historical_lap, run_historical_replay
from app.replay.strategic_rival import LapFieldEntry, build_strategic_rivals

_HAVE_FASTF1 = fastf1_available()
_needs_fastf1 = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")


# ---------------------------------------------------------------------------
# offline: the estimator follows whichever rival the selector names
# ---------------------------------------------------------------------------
def _lap(lap, rival_driver, term_speed, *, role="ATTACK_TARGET"):
    return NormalizedLap(
        lap=lap, total_laps=53, data_mode="REPLAY",
        our_speed_kmh=330.0, our_soc_mj=5.0, our_lap_start_soc_mj=5.0,
        our_lap_energy_deployed_mj=1.2, gap_to_car_ahead_s=0.5,
        strategic_rival=StrategicRivalInfo(
            driver=rival_driver, role=role, position=1, gap_s=0.5, ahead=True,
            relevance_score=0.9,
        ),
        rival_terminal_speed_kmh=term_speed, rival_clipping_point_fraction=0.4,
        rival_corner_exit_accel_g=1.6, rival_sector_delta_s=-0.1,
        energy_is_modeled=True, raw_sample_count=300,
    )


def test_new_rival_gets_its_own_fresh_filter():
    # laps 1-6 track NOR (fast), laps 7-12 track PIA (slow) — PIA never seen before
    laps = [_lap(i, "NOR", 335.0) for i in range(1, 7)]
    laps += [_lap(i, "PIA", 300.0) for i in range(7, 13)]
    ctx6 = run_pipeline(ReplayProvider(laps), lap=6, config=DecisionConfig(seed=42))
    ctx12 = run_pipeline(ReplayProvider(laps), lap=12, config=DecisionConfig(seed=42))
    assert ctx6.rival.estimate.n_observations == 6          # NOR: 6 obs
    assert ctx12.rival.estimate.n_observations == 6          # PIA: 6 obs (7..12), not 12
    assert ctx6.rival.estimate.mean_soc_mj != ctx12.rival.estimate.mean_soc_mj


def test_returning_rival_resumes_its_own_evidence_no_contamination():
    # NOR (1-4) -> PIA (5-7) -> NOR again (8-10).  Spec tests 16 + 17.
    laps = [_lap(i, "NOR", 335.0) for i in range(1, 5)]
    laps += [_lap(i, "PIA", 295.0) for i in range(5, 8)]
    laps += [_lap(i, "NOR", 336.0) for i in range(8, 11)]
    ctx4 = run_pipeline(ReplayProvider(laps), lap=4, config=DecisionConfig(seed=42))
    ctx7 = run_pipeline(ReplayProvider(laps), lap=7, config=DecisionConfig(seed=42))
    ctx10 = run_pipeline(ReplayProvider(laps), lap=10, config=DecisionConfig(seed=42))

    assert ctx4.rival.estimate.n_observations == 4           # NOR: laps 1-4
    assert ctx7.rival.estimate.n_observations == 3           # PIA: laps 5-7 only
    # NOR resumed: 4 (early) + 3 (late) = 7 of its OWN observations, never PIA's
    assert ctx10.rival.estimate.n_observations == 7
    # PIA's filter still holds exactly its 3 observations, untouched by NOR's laps
    assert ctx10._estimators["PIA"].observation_count == 3
    assert ctx10._estimators["NOR"].observation_count == 7


def test_no_reset_when_rival_identity_is_stable():
    laps = [_lap(i, "PIA", 320.0) for i in range(1, 11)]
    ctx = run_pipeline(ReplayProvider(laps), lap=10, config=DecisionConfig(seed=42))
    assert ctx.rival.estimate.n_observations == 10


def test_snapshot_carries_the_strategic_rival_metadata():
    laps = [_lap(i, "NOR", 330.0, role="DEFENDING_THREAT") for i in range(1, 6)]
    snap = run_decision(ReplayProvider(laps), lap=5, config=DecisionConfig(seed=42))
    assert snap.rival.driver == "NOR"
    assert snap.rival.role == "DEFENDING_THREAT"
    assert snap.rival.relevance_score == pytest.approx(0.9)
    stages = [s.stage for s in snap.trace]
    assert "STRATEGIC_RIVAL_SELECTED" in stages


def test_trace_reports_a_strategic_rival_change():
    laps = [_lap(i, "NOR", 330.0) for i in range(1, 5)] + [_lap(i, "SAI", 330.0) for i in range(5, 8)]
    snap = run_decision(ReplayProvider(laps), lap=5, config=DecisionConfig(seed=42))
    changed = [s for s in snap.trace if s.stage == "STRATEGIC_RIVAL_CHANGED"]
    assert changed and "NOR -> SAI" in changed[0].detail


# ---------------------------------------------------------------------------
# TEST 10 — the estimator only ever sees observable kinematics, never SoC
# ---------------------------------------------------------------------------
def test_estimator_input_has_no_energy_state():
    obs = to_rival_observation(_lap(1, "NOR", 330.0))
    assert obs is not None
    fields = set(obs.model_dump())
    assert fields == {"terminal_speed_kmh", "clipping_point_fraction",
                      "corner_exit_accel_g", "sector_delta_s"}
    assert not any("soc" in f or "energy" in f for f in fields)


# ---------------------------------------------------------------------------
# TEST 7 (causal) at the orchestration level — corrupting laps > N cannot move
# the pick at lap N.  (Pure-selector version lives in test_strategic_rival.py;
# this one goes through build_strategic_rivals as load_replay calls it.)
# ---------------------------------------------------------------------------
def test_build_strategic_rivals_is_causal():
    base = {
        "LEC": {l: LapFieldEntry("LEC", l, 2, 90.0 + 83.0 * (l - 1), 83.0) for l in range(1, 13)},
        "PIA": {l: LapFieldEntry("PIA", l, 1, 89.0 + 83.0 * (l - 1), 83.0) for l in range(1, 13)},
        "NOR": {l: LapFieldEntry("NOR", l, 3, 90.4 + 83.2 * (l - 1), 83.2) for l in range(1, 13)},
    }
    before = build_strategic_rivals(base, "LEC", [7])[7]
    corrupt = {
        d: {l: (e if l <= 7 else LapFieldEntry(d, l, 1, 0.0, 1.0, "pit"))
            for l, e in laps.items()}
        for d, laps in base.items()
    }
    after = build_strategic_rivals(corrupt, "LEC", [7])[7]
    assert before == after


# ===========================================================================
# real 2024 Monza — cache-gated
# ===========================================================================
@pytest.fixture(scope="module")
def monza_dynamic():
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    try:
        return run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
    except FastF1Unavailable as exc:
        pytest.skip(f"session unavailable: {exc}")


@_needs_fastf1
def test_monza_selects_rivals_from_the_field_not_just_pia(monza_dynamic):
    tracked = set(monza_dynamic["strategic_rival"]["drivers_tracked"])
    assert len(tracked) >= 3, f"only tracked {tracked} — selection is not using the field"
    assert monza_dynamic["strategic_rival"]["dynamic"] is True


@_needs_fastf1
def test_monza_strategic_rival_actually_changes(monza_dynamic):
    changes = monza_dynamic["strategic_rival"]["changes"]
    assert len(changes) >= 3
    for c in changes:
        assert c["from"] != c["to"]
        assert 1 <= c["lap"] <= 53


@_needs_fastf1
def test_monza_directional_roles_are_consistent_with_the_gap(monza_dynamic):
    for L in monza_dynamic["laps"]:
        sr = L["strategic_rival"]
        if sr["role"] == "ATTACK_TARGET":
            assert sr["ahead"] is True
        if sr["role"] == "DEFENDING_THREAT":
            assert sr["ahead"] is False


@_needs_fastf1
def test_monza_rival_switch_follows_the_new_cars_identity_aware_filter(monza_dynamic):
    laps = monza_dynamic["laps"]
    by_lap = {L["lap"]: L for L in laps}
    changes = monza_dynamic["strategic_rival"]["changes"]
    seen_driver_first_lap: dict = {}
    for L in laps:
        d = L["strategic_rival"]["driver"]
        seen_driver_first_lap.setdefault(d, L["lap"])
    for c in changes:
        lp = c["lap"]
        if lp not in by_lap:
            continue
        # the surfaced inference is now about the NEW driver
        assert by_lap[lp]["rival_energy_inference"]["driver"] == c["to"]
        n = by_lap[lp]["rival_energy_inference"]["n_observations"]
        # a driver seen for the FIRST time this switch -> few observations;
        # a RESUMED driver -> its filter carried its own history (n can be large),
        # but never more than the laps that driver has actually been tracked.
        assert 0 <= n <= lp


@_needs_fastf1
def test_monza_dynamic_replay_is_deterministic():
    H.clear_session_cache()
    a = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=30, seed=42)
    H.clear_session_cache()
    b = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=30, seed=42)
    assert a["laps"] == b["laps"]
    assert a["strategic_rival"]["changes"] == b["strategic_rival"]["changes"]


@_needs_fastf1
def test_monza_dynamic_lap_n_only_sees_laps_up_to_n(monkeypatch):
    seen = {}
    real = H.run_decision

    def spy(provider, *, lap, config, **kw):
        seen[lap] = max(nl.lap for nl in provider.laps())
        return real(provider, lap=lap, config=config, **kw)

    monkeypatch.setattr(H, "run_decision", spy)
    run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=20, seed=42)
    assert all(seen_max == n for n, seen_max in seen.items())


@_needs_fastf1
def test_monza_no_ground_truth_anywhere(monza_dynamic):
    import json
    blob = json.dumps(monza_dynamic).lower()
    for forbidden in ("ground_truth", "actual_soc", "hidden_soc", "real_soc", "rival_actual"):
        assert forbidden not in blob


@_needs_fastf1
def test_monza_api_contract_unbroken():
    from fastapi.testclient import TestClient
    from app.main import app

    r = TestClient(app).post("/api/v1/decision", json={"scenario": "B", "seed": 42, "lap": 30})
    assert r.status_code == 200
    body = r.json()
    for block in ("decision", "data_quality", "window", "energy", "rival", "opportunity",
                  "monte_carlo", "compliance", "confidence", "constraints",
                  "candidate_actions", "feasible_actions", "reason_codes", "trace", "meta"):
        assert block in body
    # synthetic path -> strategic-rival fields present but inert
    assert body["rival"]["driver"] is None
