"""
Season -> Grand Prix -> Session discovery on top of the curated race registry.

Schedule lookups need FastF1 (cached or online); the shape / error-path tests
that don't need a schedule run offline.
"""

from __future__ import annotations

import pytest

from app.data.fastf1_service import fastf1_available
from app.replay import discovery as D
from app.replay.historical import resolve_any_race
from app.replay.races import HISTORICAL_RACES

_HAVE_FASTF1 = fastf1_available()
_needs = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")


# ---------------------------------------------------------------------------
# offline
# ---------------------------------------------------------------------------
def test_session_codes():
    assert D._session_code("Race") == "R"
    assert D._session_code("Qualifying") == "Q"
    assert D._session_code("Practice 2") == "FP2"
    assert D._session_code("Sprint") == "S"
    assert D._session_code("Weird Session") == "Weird Session"


def test_supported_window_is_2019_onward():
    assert min(D.SUPPORTED_SEASONS) == 2019
    assert 2024 in D.SUPPORTED_SEASONS


def test_unknown_pre_telemetry_season_raises():
    with pytest.raises(D.SeasonUnavailable):
        D.list_races(2005)


def test_resolve_any_race_registry_key_still_works():
    r = resolve_any_race("2024_italian_gp")
    assert r.year == 2024 and r.default_driver == "LEC"


def test_resolve_any_race_unknown_without_season_is_keyerror():
    with pytest.raises(KeyError):
        resolve_any_race("1998_belgian_gp")


def test_resolve_any_race_adhoc_needs_a_driver():
    # season given but no driver -> ValueError (not a crash)
    with pytest.raises((ValueError, KeyError, D.SeasonUnavailable)):
        resolve_any_race("Italian Grand Prix", season=2024, session="R")


# ---------------------------------------------------------------------------
# schedule-backed (cache-gated)
# ---------------------------------------------------------------------------
@_needs
def test_list_seasons_reports_availability():
    seasons = D.list_seasons()
    yrs = {s["season"] for s in seasons}
    assert D.SUPPORTED_SEASONS[0] in yrs
    got = next(s for s in seasons if s["season"] == 2024)
    assert got["available"] and got["rounds"] >= 20


@_needs
def test_list_races_for_a_season():
    races = D.list_races(2024)
    assert len(races) >= 20
    monza = next(r for r in races if "Italian" in r["name"])
    assert monza["location"].lower() == "monza"
    assert "R" in monza["sessions"] and "Q" in monza["sessions"]
    assert "sessions" in monza and isinstance(monza["sessions"], list)


@_needs
def test_list_sessions_and_resolve():
    ss = D.list_sessions(2024, "Italian Grand Prix")
    assert ss["round"] == 16
    codes = {s["code"] for s in ss["sessions"]}
    assert {"FP1", "Q", "R"} <= codes

    info = D.resolve_session(2024, "italian", "R")
    assert info == {
        "year": 2024, "event": "Italian Grand Prix", "session": "R",
        "name": "2024 Italian Grand Prix", "circuit": "Monza", "scheduled_laps": None,
    }
    # round number and loose match both work
    assert D.resolve_session(2024, "16", "Q")["session"] == "Q"


@_needs
def test_unknown_event_in_a_real_season_is_keyerror():
    with pytest.raises(KeyError):
        D.list_sessions(2024, "Narnian Grand Prix")


@_needs
def test_adhoc_replay_resolves_through_discovery():
    r = resolve_any_race("Italian Grand Prix", season=2024, session="R", driver="LEC", rival="PIA")
    assert r.year == 2024 and r.event == "Italian Grand Prix" and r.session == "R"
    assert r.default_driver == "LEC"
    assert r.key not in HISTORICAL_RACES        # ad-hoc, not registered


@_needs
def test_api_discovery_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)

    assert c.get("/api/v1/replay/seasons").status_code == 200
    r = c.get("/api/v1/replay/races?season=2024")
    assert r.status_code == 200 and len(r.json()["races"]) >= 20
    # no-arg still returns the curated list (backward compatible)
    r0 = c.get("/api/v1/replay/races")
    assert r0.status_code == 200 and any(x["key"] == "2024_italian_gp" for x in r0.json()["races"])
    r2 = c.get("/api/v1/replay/sessions?season=2024&race=Italian Grand Prix")
    assert r2.status_code == 200 and r2.json()["round"] == 16
