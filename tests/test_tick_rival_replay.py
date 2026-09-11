"""
Telemetry-tick strategic rival, end to end on the real 2024 Monza replay.

All cache-gated (need FastF1 + a populated .fastf1_cache). Covers the tick-level
spec requirements that only real telemetry can exercise: intra-lap rival changes,
no-hindsight against corrupted future samples, identity-aware energy, the timeline
endpoint, and cross-rebuild determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.data.fastf1_service import (
    FastF1Unavailable,
    clear_field_timeline_cache,
    fastf1_available,
    field_timeline_cache_key,
    get_cached_field_timeline,
)
from app.replay import historical as H
from app.replay import resolve_race, run_historical_replay
from app.replay.historical import strategic_rival_timeline
from app.replay.strategic_rival import RivalSelectorConfig

_HAVE_FASTF1 = fastf1_available()
_needs = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")

_RACE = "2024_italian_gp"


@pytest.fixture(scope="module")
def monza_session():
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    import fastf1
    fastf1.set_log_level("ERROR")
    fastf1.Cache.enable_cache(".fastf1_cache")
    try:
        ses = fastf1.get_session(2024, "Italian Grand Prix", "R")
        ses.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception as exc:
        pytest.skip(f"session unavailable: {exc}")
    return ses


@pytest.fixture(scope="module")
def monza_replay():
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    try:
        return run_historical_replay(race_key=_RACE, start_lap=1, end_lap=53, seed=42)
    except FastF1Unavailable as exc:
        pytest.skip(f"session unavailable: {exc}")


# ---------------------------------------------------------------------------
# it is genuinely tick-level, and the rival genuinely changes within laps
# ---------------------------------------------------------------------------
@_needs
def test_replay_is_tick_level(monza_replay):
    sr = monza_replay["strategic_rival"]
    assert sr["tick_level"] is True
    assert sr["intra_lap_changes_total"] >= 5      # real sub-lap changes, not fabricated
    assert len(sr["drivers_tracked"]) >= 3


@_needs
def test_some_laps_have_intra_lap_rival_changes(monza_replay):
    laps_with = [L["lap"] for L in monza_replay["laps"]
                 if L["strategic_rival"]["changes_this_lap"] > 0]
    assert laps_with, "no lap had a sub-lap rival change — selection is still lap-locked"
    # and every one is backed by a real change_events list from the timeline
    lap = laps_with[0]
    tl = strategic_rival_timeline(race_key=_RACE, lap=lap)
    assert tl["summary"]["n_changes"] == len(tl["change_events"]) >= 1
    assert tl["summary"]["n_changes"] == \
        next(L for L in monza_replay["laps"] if L["lap"] == lap)["strategic_rival"]["changes_this_lap"]


@_needs
def test_timeline_ticks_are_real_and_downsampled(monza_replay):
    tl = strategic_rival_timeline(race_key=_RACE, lap=5, max_ticks=100)
    assert 1.0 <= tl["tick_cadence_hz"] <= 30.0        # FastF1 real cadence, ~4 Hz
    assert tl["total_ticks_this_lap"] > tl["returned_ticks"] or tl["downsample_step"] == 1
    ts = [t["t"] for t in tl["ticks"]]
    assert ts == sorted(ts) and ts[0] >= 0.0
    for t in tl["ticks"]:
        assert set(t) >= {"t", "driver", "role", "ahead", "gap_s", "relevance"}
        assert "soc" not in str(t).lower()             # no energy state in the selector output


# ---------------------------------------------------------------------------
# NO HINDSIGHT — corrupting future telemetry cannot move an earlier selection
# ---------------------------------------------------------------------------
@_needs
def test_future_telemetry_cannot_change_tick_selection(monza_session):
    from app.replay.field_state import FieldTimeline
    import fastf1

    base = FieldTimeline.build(monza_session, "LEC", scheduled_laps=53)
    tl_base = base.selection_timeline()

    # cut point ~ 40% through the race
    T = base.ticks[int(len(base.ticks) * 0.4)]

    # corrupt EVERY driver's car_data speed after session-time T, in place
    saved = {}
    for num, cd in monza_session.car_data.items():
        st = cd["SessionTime"].dt.total_seconds().to_numpy()
        m = st > T
        saved[num] = cd["Speed"].to_numpy().copy()
        cd.loc[m, "Speed"] = 999.0
    try:
        corrupt = FieldTimeline.build(monza_session, "LEC", scheduled_laps=53)
        tl_corr = corrupt.selection_timeline()
    finally:
        for num, cd in monza_session.car_data.items():
            cd["Speed"] = saved[num]

    keep = [i for i, s in enumerate(tl_base) if s.t <= T]
    for i in keep:
        a, b = tl_base[i], tl_corr[i]
        assert (a.driver, a.role, a.ahead, a.gap_s, a.relevance_score) == \
               (b.driver, b.role, b.ahead, b.gap_s, b.relevance_score), \
               f"tick {i} (t={a.t:.1f} <= {T:.1f}) changed when the future was corrupted"


@_needs
def test_a_later_overtake_does_not_pre_promote_that_driver(monza_replay):
    # PIA's late-race charge (drivers_tracked includes PIA) must not make PIA the
    # tracked rival during the mid-race stint where SAI is the real neighbour.
    by_lap = {L["lap"]: L for L in monza_replay["laps"]}
    for lp in (39, 40, 41, 42):
        if lp in by_lap:
            assert by_lap[lp]["strategic_rival"]["driver"] != "PIA" or \
                   by_lap[lp]["strategic_rival"]["role"] == "NONE"


# ---------------------------------------------------------------------------
# determinism across a fresh rebuild (proxy for cross-process)
# ---------------------------------------------------------------------------
@_needs
def test_tick_selection_is_stable_across_rebuild(monza_session):
    from app.replay.field_state import FieldTimeline
    a = FieldTimeline.build(monza_session, "LEC", scheduled_laps=53).selection_timeline()
    b = FieldTimeline.build(monza_session, "LEC", scheduled_laps=53).selection_timeline()
    assert a == b


@_needs
def test_replay_deterministic_after_cache_clear():
    H.clear_session_cache(); clear_field_timeline_cache()
    a = run_historical_replay(race_key=_RACE, start_lap=1, end_lap=25, seed=42)
    H.clear_session_cache(); clear_field_timeline_cache()
    b = run_historical_replay(race_key=_RACE, start_lap=1, end_lap=25, seed=42)
    assert a["laps"] == b["laps"]
    assert a["strategic_rival"]["changes"] == b["strategic_rival"]["changes"]


# ---------------------------------------------------------------------------
# identity-aware energy follows the tick-dominant rival, never contaminated
# ---------------------------------------------------------------------------
@_needs
def test_energy_inference_driver_matches_the_lap_rival(monza_replay):
    for L in monza_replay["laps"]:
        assert L["rival_energy_inference"]["driver"] == L["strategic_rival"]["driver"]


@_needs
def test_no_ground_truth_or_soc_leak_anywhere(monza_replay):
    import json
    blob = json.dumps(monza_replay).lower()
    for bad in ("ground_truth", "actual_soc", "hidden_soc", "real_soc", "rival_actual"):
        assert bad not in blob
    tl = strategic_rival_timeline(race_key=_RACE, lap=10)
    assert "soc" not in json.dumps(tl["ticks"]).lower()


# ---------------------------------------------------------------------------
# the FieldTimeline is cached, not rebuilt per request
# ---------------------------------------------------------------------------
@_needs
def test_field_timeline_is_cached(monza_replay):
    race = resolve_race(_RACE)
    key = field_timeline_cache_key(race.year, race.event, race.session, "LEC")
    assert get_cached_field_timeline(key) is not None
