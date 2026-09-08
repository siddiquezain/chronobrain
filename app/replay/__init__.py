"""
app.replay — historical F1 race replay through the EXISTING ChronoPace pipeline.

Real 2024 FastF1 telemetry, censored lap by lap, fed to `run_decision` exactly as
the synthetic and demo paths do. No second decision engine, no hindsight.

    FastF1 historical session  (app/data/fastf1_service.load_replay)
        -> list[NormalizedLap]           one causal lap each
        -> for lap N:  laps <= N only    <-- hindsight barrier
        -> ReplayProvider(censored)
        -> run_decision(provider, lap=N) <-- the existing pipeline
        -> DecisionSnapshot -> compact lap summary
"""

from app.replay.races import HISTORICAL_RACES, HistoricalRace, resolve_race
from app.replay.historical import (
    run_historical_lap,
    run_historical_replay,
    strategic_rival_timeline,
)
from app.replay.strategic_rival import (
    RivalSelectorConfig,
    StrategicRival,
    build_strategic_rivals,
    score_matrix,
    select_strategic_rival,
)
from app.replay.field_state import FieldTimeline, TickSelection, select_over_ticks

__all__ = [
    "HISTORICAL_RACES",
    "HistoricalRace",
    "resolve_race",
    "run_historical_replay",
    "run_historical_lap",
    "strategic_rival_timeline",
    "RivalSelectorConfig",
    "StrategicRival",
    "build_strategic_rivals",
    "select_strategic_rival",
    "score_matrix",
    "FieldTimeline",
    "TickSelection",
    "select_over_ticks",
]
