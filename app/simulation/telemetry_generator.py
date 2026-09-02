"""
Adapter-style telemetry source interface.

TelemetrySource is the abstract base. Core engines receive TelemetryState
regardless of source — the core never knows which source produced the data.
This makes a real data source (FastF1, live telemetry) a drop-in replacement.
"""

import abc
from datetime import datetime
from typing import Optional

import numpy as np

from app.models.telemetry import TelemetryState
from app.simulation.scenarios import ScenarioConfig


class TelemetrySource(abc.ABC):
    """Abstract base for all telemetry sources."""

    @abc.abstractmethod
    def next(self) -> TelemetryState:
        ...

    @abc.abstractmethod
    def has_next(self) -> bool:
        ...


class SimulationTelemetrySource(TelemetrySource):
    """Deterministic, seeded simulation telemetry source."""

    def __init__(self, config: ScenarioConfig, seed: int = 42):
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._lap = config.current_lap
        self._soc_mj = config.initial_soc_mj
        self._lap_start_soc = config.initial_soc_mj
        self._deployed = config.lap_energy_deployed_mj
        self._qualified_last = config.overtake_qualified_last_lap

    def has_next(self) -> bool:
        return self._lap <= self.config.total_laps

    def next(self) -> TelemetryState:
        cfg = self.config
        gap_ahead = float(np.clip(
            cfg.gap_to_car_ahead_s + self._rng.normal(0, 0.1), 0.1, 5.0
        ))
        gap_behind = float(np.clip(
            cfg.gap_to_car_behind_s + self._rng.normal(0, 0.1), 0.1, 10.0
        ))
        closing = float(np.clip(
            cfg.closing_speed_mps + self._rng.normal(0, 0.5), -5.0, 20.0
        ))
        speed = float(np.clip(280 + self._rng.normal(0, 10), 150, 350))

        harvested = float(np.clip(1.5 + self._rng.normal(0, 0.2), 0.5, 2.5))
        deployed = float(np.clip(cfg.lap_energy_deployed_mj + self._rng.normal(0, 0.1), 0, 9.0))
        new_soc = float(np.clip(self._soc_mj + harvested - 1.0, 0.0, 9.0))

        qualified_last = self._qualified_last
        self._qualified_last = gap_ahead <= 1.0

        t = TelemetryState(
            timestamp=datetime.utcnow().isoformat(),
            lap=self._lap,
            total_laps=cfg.total_laps,
            position=cfg.initial_position,
            gap_to_car_ahead_s=round(gap_ahead, 3),
            gap_to_car_behind_s=round(gap_behind, 3),
            speed_kmh=round(speed, 1),
            closing_speed_mps=round(closing, 2),
            throttle=float(np.clip(0.7 + self._rng.normal(0, 0.1), 0, 1)),
            brake=float(np.clip(0.1 + self._rng.normal(0, 0.05), 0, 1)),
            braking_point=False,
            sector=1 + (self._lap % 3),
            corner_id=None,
            straight_distance_m=float(cfg.straight_distance_m),
            distance_to_next_corner_m=float(np.clip(200 + self._rng.normal(0, 30), 0, 800)),
            slipstream_factor=float(np.clip(cfg.slipstream_factor + self._rng.normal(0, 0.05), 0, 1)),
            soc_mj=round(new_soc, 3),
            soc_pct=round(new_soc / 9.0 * 100, 1),
            energy_deployment_mj=round(deployed, 3),
            energy_harvest_mj=round(harvested, 3),
            energy_remaining_mj=round(new_soc, 3),
            energy_budget_mj=round(max(0.0, 9.0 - deployed), 3),
            tyre_age_laps=cfg.tyre_age_laps + (self._lap - cfg.current_lap),
            tyre_compound=cfg.tyre_compound,
            drs_available=gap_ahead < 1.0,
            overtake_opportunity=gap_ahead < 1.0 and closing > 3.0,
            track_position=float(np.clip(self._rng.uniform(0, 1), 0, 1)),
            lap_start_soc_mj=round(self._lap_start_soc, 3),
            overtake_qualified_last_lap=qualified_last,
        )

        self._lap += 1
        self._lap_start_soc = new_soc
        self._soc_mj = new_soc
        self._deployed = deployed
        return t


class CSVTelemetrySource(TelemetrySource):
    """
    CSV-backed telemetry source.
    ponytail: stub — wire to real CSV when data is available.
    """

    def __init__(self, csv_path: str):
        import pandas as pd
        self._df = pd.read_csv(csv_path)
        self._idx = 0

    def has_next(self) -> bool:
        return self._idx < len(self._df)

    def next(self) -> TelemetryState:
        row = self._df.iloc[self._idx].to_dict()
        self._idx += 1
        valid = {k: v for k, v in row.items() if k in TelemetryState.model_fields}
        return TelemetryState(**valid)
