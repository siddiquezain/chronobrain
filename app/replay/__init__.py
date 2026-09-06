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
from app.replay.historical import run_historical_replay, run_historical_lap

__all__ = [
    "HISTORICAL_RACES",
    "HistoricalRace",
    "resolve_race",
    "run_historical_replay",
    "run_historical_lap",
]
