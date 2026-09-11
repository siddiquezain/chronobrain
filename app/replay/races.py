"""
Historical race registry. `scheduled_laps` is the pre-race race distance (public
before lights out) — used for `total_laps` so it is never hindsight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HistoricalRace:
    key: str
    name: str
    circuit: str
    year: int
    event: str        # what FastF1's get_session() accepts
    session: str = "R"
    scheduled_laps: int = 0
    default_driver: str = ""
    default_rival: str = ""
    note: str = ""


HISTORICAL_RACES: Dict[str, HistoricalRace] = {
    "2024_italian_gp": HistoricalRace(
        key="2024_italian_gp",
        name="2024 Italian Grand Prix",
        circuit="Autodromo Nazionale Monza",
        year=2024,
        event="Italian Grand Prix",
        session="R",
        scheduled_laps=53,
        default_driver="LEC",
        default_rival="PIA",
        note="Leclerc's one-stop win; Piastri closed the gap in the final laps.",
    ),
}


def resolve_race(key: str) -> HistoricalRace:
    k = key.strip().lower()
    if k not in HISTORICAL_RACES:
        raise KeyError(f"unknown race {key!r}; known: {sorted(HISTORICAL_RACES)}")
    return HISTORICAL_RACES[k]
