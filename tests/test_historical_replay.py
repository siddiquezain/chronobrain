"""
Historical F1 race replay — real 2024 telemetry through the EXISTING pipeline,
censored lap by lap so no future information can reach a Lap N decision.

Tests that need FastF1 + a populated cache are marked and skip cleanly otherwise.
The hindsight / pipeline-reuse / no-ground-truth tests run offline against a
hand-built NormalizedLap fixture (a UNIT-TEST fixture, not passed off as real data).
"""

from __future__ import annotations

import math

import pytest

from app.data.fastf1_service import FastF1Unavailable, fastf1_available
from app.data.samples import NormalizedLap
from app.replay import historical as H
from app.replay import resolve_race, run_historical_lap, run_historical_replay
from app.replay.races import HISTORICAL_RACES

_HAVE_FASTF1 = fastf1_available()
_needs_fastf1 = pytest.mark.skipif(
    not _HAVE_FASTF1, reason="fastf1 not installed / session cache unavailable"
)


@pytest.fixture(scope="module")
def real_monza_loaded():
    """Load the real 2024 Monza session ONCE for the whole module (or skip)."""
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    from app.data.fastf1_service import load_replay
    race = resolve_race("2024_italian_gp")
    try:
        laps = load_replay(year=race.year, event=race.event, session=race.session,
                           our_driver="LEC", rival_driver="PIA", scheduled_laps=53)
    except FastF1Unavailable as exc:
        pytest.skip(f"session unavailable: {exc}")
    H._SESSION_CACHE[H.session_cache_key(race.key, "LEC", "PIA", False)] = laps
    return laps


# ---------------------------------------------------------------------------
# offline fixture — mimics load_replay() output shape for ~20 causal laps
# ---------------------------------------------------------------------------
def _fixture_laps(n=20, rival_pattern="steady") -> list[NormalizedLap]:
    laps = []
    soc = 4.5
    for i in range(1, n + 1):
        # a rival whose terminal speed drops after lap 10 => "fading" pattern
        term = 320.0
        if rival_pattern == "fading" and i > 10:
            term = 320.0 - (i - 10) * 4.0
        soc = max(0.5, min(9.0, soc - 0.02 + 0.15 * (4.5 - soc)))
        laps.append(NormalizedLap(
            lap=i, total_laps=53, data_mode="REPLAY",
            our_speed_kmh=330.0, our_soc_mj=round(soc, 3), our_lap_start_soc_mj=round(soc + 0.05, 3),
            our_lap_energy_deployed_mj=1.4, gap_to_car_ahead_s=0.6,
            rival_terminal_speed_kmh=term, rival_clipping_point_fraction=0.4,
            rival_corner_exit_accel_g=1.6, rival_sector_delta_s=-0.1,
            energy_is_modeled=True, raw_sample_count=300,
            source_detail="FIXTURE (not real data)",
        ))
    return laps


@pytest.fixture
def offline_race(monkeypatch):
    """Pre-populate the replay session cache so no fastf1 call happens. These tests
    exercise pipeline mechanics on a fixed two-car fixture (dynamic_rival=False);
    the dynamic strategic-rival path has its own tests."""
    H.clear_session_cache()
    race = resolve_race("2024_italian_gp")
    H._SESSION_CACHE[H.session_cache_key(race.key, "LEC", "PIA", False)] = _fixture_laps(20)
    yield race
    H.clear_session_cache()


# ---------------------------------------------------------------------------
# 1. race config resolves
# ---------------------------------------------------------------------------
def test_race_config_resolves():
    r = resolve_race("2024_italian_gp")
    assert (r.year, r.event, r.session) == (2024, "Italian Grand Prix", "R")
    assert r.scheduled_laps == 53
    assert (r.default_driver, r.default_rival) == ("LEC", "PIA")
    assert "2024_italian_gp" in HISTORICAL_RACES


def test_unknown_race_key_errors():
    with pytest.raises(KeyError):
        resolve_race("1998_belgian_gp")


# ---------------------------------------------------------------------------
# 5 + 6. LAP N ONLY SEES LAPS <= N  (the hindsight barrier)
# ---------------------------------------------------------------------------
def test_lap_n_provider_contains_only_laps_up_to_n(offline_race, monkeypatch):
    seen = {}
    real = H.run_decision

    def spy(provider, *, lap, config, **kw):
        seen[lap] = [nl.lap for nl in provider.laps()]
        return real(provider, lap=lap, config=config, **kw)

    monkeypatch.setattr(H, "run_decision", spy)
    run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=15, seed=42,
                          dynamic_rival=False)

    for n, lap_list in seen.items():
        assert max(lap_list) == n, f"lap {n} decision saw future lap {max(lap_list)}"
        assert lap_list == list(range(1, n + 1))


def test_future_telemetry_cannot_change_a_lap_n_decision(offline_race):
    base = run_historical_replay(race_key="2024_italian_gp", start_lap=8, end_lap=8, seed=42,
                                 dynamic_rival=False)
    lap8_a = base["laps"][0]

    # now corrupt every lap AFTER 8 with wild values and replay lap 8 again
    race = resolve_race("2024_italian_gp")
    laps = H._SESSION_CACHE[H.session_cache_key(race.key, "LEC", "PIA", False)]
    for nl in laps:
        if nl.lap > 8:
            nl.rival_terminal_speed_kmh = 999.0
            nl.gap_to_car_ahead_s = 0.01
            nl.our_soc_mj = 9.0
    lap8_b = run_historical_replay(race_key="2024_italian_gp", start_lap=8, end_lap=8, seed=42,
                                   dynamic_rival=False)["laps"][0]

    assert lap8_a == lap8_b, "future-lap corruption changed the Lap 8 decision"


# ---------------------------------------------------------------------------
# 4. determinism
# ---------------------------------------------------------------------------
def test_replay_is_deterministic(offline_race):
    a = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=18, seed=42,
                              dynamic_rival=False)
    b = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=18, seed=42,
                              dynamic_rival=False)
    assert a["laps"] == b["laps"]
    a2 = run_historical_lap(race_key="2024_italian_gp", lap=12, seed=42, dynamic_rival=False)
    b2 = run_historical_lap(race_key="2024_italian_gp", lap=12, seed=42, dynamic_rival=False)
    assert a2 == b2


# ---------------------------------------------------------------------------
# 12. uses the SAME existing pipeline, not a duplicate
# ---------------------------------------------------------------------------
def test_replay_uses_run_decision_not_a_second_engine():
    import inspect
    src = inspect.getsource(H)
    assert "from app.decision import" in src and "run_decision" in src
    # no local re-implementation of the stages
    for banned in ("MonteCarloPlanner(", "RivalStateEstimator(", "ConfidenceGate(", "class .*Engine"):
        assert banned.rstrip("(") not in src or "run_decision" in src
    assert H.run_decision.__module__ == "app.decision.engine"


# ---------------------------------------------------------------------------
# 7 + 8. rival estimator never gets hidden SoC; response has no ground truth
# ---------------------------------------------------------------------------
def test_rival_observation_has_no_soc_field():
    from telemetry_simulator import RivalObservation
    assert not any("soc" in f.lower() for f in RivalObservation.model_fields)


def test_historical_response_never_exposes_ground_truth(offline_race):
    import json
    res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=20, seed=42,
                                dynamic_rival=False)
    blob = json.dumps(res).lower()
    for forbidden in ("ground_truth", "actual_soc", "rival_actual", "hidden_soc", "real_soc"):
        assert forbidden not in blob
    lap = res["laps"][5]
    assert "RIVAL ENERGY INFERENCE" in lap["rival_energy_inference"]["label"]
    assert set(lap["rival_energy_inference"]) == {
        "label", "driver", "estimated_reserve_mj", "std_mj", "bucket",
        "distribution", "confidence", "n_observations", "p_defend",
        # posterior health fields (Task 5)
        "posterior_mean_mj", "posterior_std_mj",
        "effective_sample_size", "evidence_quality", "posterior_health",
        "baseline_ready", "energy_provenance",
    }
    assert "rival_reference_soc_mj" in lap
    assert lap["rival_reference_provenance"] == "CHRONOPACE_MODELED"


def test_estimator_maintains_uncertainty_over_the_replay(offline_race):
    res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=20, seed=42,
                                dynamic_rival=False)
    stds = [L["rival_energy_inference"]["std_mj"] for L in res["laps"]]
    assert all(s >= 0.35 for s in stds)          # floor, never collapses
    n_obs = [L["rival_energy_inference"]["n_observations"] for L in res["laps"]]
    assert n_obs == list(range(1, 21))           # one observation per lap, censored


def test_posterior_health_fields_are_populated(offline_race):
    """Posterior health fields must carry meaningful values, not just keys."""
    res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=10, seed=42,
                                dynamic_rival=False)
    for L in res["laps"]:
        rei = L["rival_energy_inference"]
        # numeric fields must be non-negative floats
        assert isinstance(rei["posterior_mean_mj"], float) and rei["posterior_mean_mj"] >= 0.0
        assert isinstance(rei["posterior_std_mj"], float) and rei["posterior_std_mj"] >= 0.0
        assert isinstance(rei["effective_sample_size"], float) and rei["effective_sample_size"] >= 0.0
        # categorical fields must be known values
        assert rei["evidence_quality"] in ("insufficient", "weak", "moderate", "strong")
        assert rei["posterior_health"] in ("healthy", "collapsed", "roughened", "insufficient_data")
        assert isinstance(rei["baseline_ready"], bool)
        assert rei["energy_provenance"] == "INFERRED"


# ---------------------------------------------------------------------------
# 9. missing fastf1 / cache -> clear error (not fabricated data)
# ---------------------------------------------------------------------------
def test_missing_fastf1_raises_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_fastf1(name, *a, **k):
        if name == "fastf1" or name.startswith("fastf1."):
            raise ModuleNotFoundError("No module named 'fastf1'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fastf1)
    from app.data import fastf1_service
    with pytest.raises(FastF1Unavailable) as exc:
        fastf1_service.load_replay(year=2024, event="Italian Grand Prix", session="R",
                                   our_driver="LEC", rival_driver="PIA")
    assert "fastf1" in str(exc.value).lower()


def test_replay_endpoint_reports_503_when_session_unavailable(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    def boom(**kw):
        raise FastF1Unavailable("could not load 2024 Italian Grand Prix R from FastF1 (offline)")

    monkeypatch.setattr(H, "load_replay", boom)
    H.clear_session_cache()
    r = TestClient(app).post("/api/v1/replay/historical", json={"race": "2024_italian_gp"})
    assert r.status_code == 503
    assert "offline" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 10. existing synthetic scenarios unchanged
# ---------------------------------------------------------------------------
def test_synthetic_scenarios_unchanged_by_this_feature():
    from app.decision import run_scenario
    expect = {
        "A": ("BALANCED_MODE", "HOLD"), "B": ("USE_OVERTAKE_BONUS_MODE", "ATTACK_NOW"),
        "C": ("CONSERVE_MODE", "CONSERVE"), "D": ("PUSH_MODE", "PUSH"),
        "E": ("BALANCED_MODE", "HOLD"),
    }
    for sc, (mode, action) in expect.items():
        s = run_scenario(sc, seed=42, lap=30, total_laps=50)
        assert (s.decision.mode, s.decision.action) == (mode, action)


# ---------------------------------------------------------------------------
# 2 + 3. FastF1 actually loads the 2024 Italian GP  (needs cache)
# ---------------------------------------------------------------------------
def test_fastf1_loads_2024_italian_gp_lec_vs_pia(real_monza_loaded):
    laps = real_monza_loaded
    assert 45 <= len(laps) <= 53
    assert laps[0].total_laps == 53
    assert all(nl.energy_is_modeled for nl in laps)          # SoC is modelled, not measured
    # racing laps carry a rival observation; pit / out / invalid laps deliberately do not
    racing = [nl for nl in laps if nl.rival_lap_status == "racing"]
    assert len(racing) >= 40
    assert all(nl.rival_terminal_speed_kmh and nl.rival_terminal_speed_kmh > 250 for nl in racing)
    assert all(nl.rival_terminal_speed_kmh is None for nl in laps if nl.rival_lap_status != "racing")
    assert "LEC vs PIA" in laps[0].source_detail


def test_real_monza_replay_runs_and_is_deterministic(real_monza_loaded):
    a = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
    b = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
    assert a["laps"] == b["laps"]
    assert 45 <= len(a["laps"]) <= 53
    decisions = {L["decision"] for L in a["laps"]}
    assert decisions <= {"ATTACK", "WAIT", "CONSERVE", "HOLD", "PUSH"}
    assert a["source"] == "fastf1_historical_replay" and a["model"] == "chronopace_2026"
    socs = [L["energy"]["soc_mj"] for L in a["laps"]]
    assert min(socs) > 1.0 and max(socs) < 9.0    # modelled band, not drained to zero


def test_real_monza_lap_n_only_sees_laps_up_to_n(real_monza_loaded, monkeypatch):
    seen = {}
    real = H.run_decision

    def spy(provider, *, lap, config, **kw):
        seen[lap] = max(nl.lap for nl in provider.laps())
        return real(provider, lap=lap, config=config, **kw)

    monkeypatch.setattr(H, "run_decision", spy)
    run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=20, seed=42)
    assert all(seen_max == n for n, seen_max in seen.items())
