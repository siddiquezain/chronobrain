"""
discovery.py — enumerate the historical races FastF1 can actually give us.

The curated registry (`races.py`) stays the source of the *featured* races (with a
default driver/rival and a note). This module adds dynamic
Season -> Grand Prix -> Session discovery on top, so the replay can reach any
telemetry-supported race in the 2018-2025 window without hardcoding each one.

Everything is best-effort and fails loudly:
  * unknown / pre-telemetry season -> SeasonUnavailable
  * unknown event / session        -> KeyError
  * offline + not cached            -> the underlying FastF1 error surfaces

Schedules are cached per season in-process. Nothing here loads telemetry.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# telemetry (car + position data) exists from 2018; 2018 is patchy, so the
# supported window we advertise is 2019 onward. The upper bound is open — a
# season with no schedule yet simply returns SeasonUnavailable.
SUPPORTED_SEASONS = list(range(2019, 2027))

_SCHEDULE_CACHE: Dict[int, list] = {}


class SeasonUnavailable(RuntimeError):
    """FastF1 has no schedule for this season (too old, or offline + uncached)."""


# ---------------------------------------------------------------------------
def _schedule(season: int) -> list:
    if season in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[season]
    try:
        import fastf1
    except Exception as exc:  # pragma: no cover
        raise SeasonUnavailable("the `fastf1` package is not installed") from exc
    try:
        fastf1.set_log_level("ERROR")
        sched = fastf1.get_event_schedule(int(season), include_testing=False)
    except Exception as exc:
        raise SeasonUnavailable(
            f"no schedule for {season} (unsupported season, or offline and not cached): {exc}"
        ) from exc

    rows: List[dict] = []
    for _, r in sched.iterrows():
        try:
            rnd = int(r["RoundNumber"])
        except Exception:
            continue
        if rnd < 1:
            continue
        sessions = []
        for i in range(1, 6):
            name = r.get(f"Session{i}")
            if name is None or (isinstance(name, float) and name != name) or str(name) in ("", "None"):
                continue
            sd = r.get(f"Session{i}DateUtc")
            sessions.append({
                "name": str(name),
                "code": _session_code(str(name)),
                "date_utc": (str(sd)[:19] if sd is not None and sd == sd else None),
            })
        rows.append({
            "round": rnd,
            "event": str(r["EventName"]),                # what get_session() accepts
            "name": str(r["EventName"]),
            "official_name": str(r.get("OfficialEventName") or r["EventName"]),
            "country": str(r.get("Country") or ""),
            "location": str(r.get("Location") or ""),
            "event_date": (str(r.get("EventDate"))[:10] if r.get("EventDate") is not None else None),
            "format": str(r.get("EventFormat") or "conventional"),
            "telemetry_supported": bool(r.get("F1ApiSupport", True)),
            "sessions": sessions,
        })
    rows.sort(key=lambda x: x["round"])
    _SCHEDULE_CACHE[season] = rows
    return rows


def _session_code(name: str) -> str:
    n = name.strip().lower()
    return {
        "practice 1": "FP1", "practice 2": "FP2", "practice 3": "FP3",
        "qualifying": "Q", "sprint": "S", "sprint qualifying": "SQ",
        "sprint shootout": "SS", "race": "R",
    }.get(n, name)


def clear_schedule_cache() -> None:
    _SCHEDULE_CACHE.clear()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def list_seasons() -> List[dict]:
    """The seasons we advertise. `available` reflects whether the schedule loads
    right now (cache/network)."""
    out = []
    for yr in SUPPORTED_SEASONS:
        try:
            n = len(_schedule(yr))
            out.append({"season": yr, "available": True, "rounds": n})
        except SeasonUnavailable:
            out.append({"season": yr, "available": False, "rounds": 0})
    return out


def list_races(season: int) -> List[dict]:
    return [
        {k: v for k, v in row.items() if k != "sessions"}
        | {"sessions": [s["code"] for s in row["sessions"]]}
        for row in _schedule(season)
    ]


def _find_event(season: int, race: str) -> dict:
    rows = _schedule(season)
    key = str(race).strip().lower()
    for row in rows:
        if key == str(row["round"]) or key == row["name"].lower() or key == row["event"].lower():
            return row
    # loose contains-match (e.g. "italian" -> "Italian Grand Prix")
    for row in rows:
        if key in row["name"].lower():
            return row
    raise KeyError(f"no {season} event matching {race!r}; try GET /api/v1/replay/races?season={season}")


def list_sessions(season: int, race: str) -> dict:
    ev = _find_event(season, race)
    return {
        "season": season, "round": ev["round"], "event": ev["event"],
        "name": ev["name"], "circuit": ev["location"], "country": ev["country"],
        "telemetry_supported": ev["telemetry_supported"],
        "sessions": ev["sessions"],
    }


def resolve_session(season: int, race: str, session: str = "R") -> dict:
    """Validate (season, race, session) and return everything `load_replay` needs:
    `{year, event, session, name, circuit, scheduled_laps}` (scheduled_laps unknown
    -> None, load_replay then falls back to the session's own count)."""
    ev = _find_event(season, race)
    want = str(session).strip().lower()
    codes = {s["code"].lower(): s for s in ev["sessions"]}
    names = {s["name"].lower(): s for s in ev["sessions"]}
    chosen = codes.get(want) or names.get(want)
    if chosen is None and want in ("r", "race"):
        chosen = names.get("race")
    if chosen is None:
        raise KeyError(
            f"{ev['name']} {season} has no session {session!r}; "
            f"available: {[s['code'] for s in ev['sessions']]}"
        )
    if not ev["telemetry_supported"]:
        raise SeasonUnavailable(
            f"{ev['name']} {season} predates FastF1 telemetry support"
        )
    return {
        "year": int(season),
        "event": ev["event"],
        "session": chosen["code"],
        "name": f"{season} {ev['name']}",
        "circuit": ev["location"] or ev["country"],
        "scheduled_laps": None,
    }
