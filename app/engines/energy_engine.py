"""
Energy Engine — deterministic and explainable energy state tracking.

Distinguishes three concepts:
  Harvesting: energy gained through regenerative braking
  Deployment: energy consumed during acceleration/deployment phases
  Reserve: energy that must remain for future strategic situations

projected_reserve = current_soc + expected_future_harvest - expected_future_deployment
"""

import logging
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.models.energy import EnergyState
from app.models.telemetry import TelemetryState

logger = logging.getLogger(__name__)

# Per-lap harvest/deployment estimates — engineered, not measured from real PU data
_EXPECTED_HARVEST_PER_LAP_MJ = 1.5
_EXPECTED_DEPLOY_PER_LAP_MJ = 1.8
_AGGRESSIVE_DEPLOY_THRESHOLD_MJ = 3.0  # min SoC to afford aggressive deployment


class EnergyEngine:
    """
    Deterministic energy state calculator.
    All values derived from telemetry fields — no hidden state, fully replayable.
    """

    def update_energy_state(self, telemetry: TelemetryState) -> EnergyState:
        """Compute full EnergyState from current telemetry snapshot."""
        settings = get_settings()

        soc_mj = telemetry.soc_mj
        soc_pct = round(soc_mj / settings.energy_capacity_mj * 100.0, 2)
        deployed = telemetry.energy_deployment_mj
        harvested = telemetry.energy_harvest_mj

        headroom = self.calculate_deployment_headroom(
            deployed_this_lap_mj=deployed,
            has_bonus=telemetry.overtake_qualified_last_lap,
        )

        laps_remaining = max(0, telemetry.total_laps - telemetry.lap)
        projected = self.calculate_projected_reserve(
            soc_mj=soc_mj,
            laps_remaining=laps_remaining,
            deploy_per_lap=_EXPECTED_DEPLOY_PER_LAP_MJ,
            harvest_per_lap=_EXPECTED_HARVEST_PER_LAP_MJ,
        )

        end_of_race = float(
            np.clip(
                soc_mj + laps_remaining * (_EXPECTED_HARVEST_PER_LAP_MJ - _EXPECTED_DEPLOY_PER_LAP_MJ),
                0.0,
                settings.energy_capacity_mj,
            )
        )

        can_afford = (
            soc_mj >= _AGGRESSIVE_DEPLOY_THRESHOLD_MJ
            and headroom >= 1.5
            and projected >= settings.min_energy_reserve_mj
        )

        logger.debug(
            f"Energy state: soc={soc_mj:.2f}MJ headroom={headroom:.2f}MJ "
            f"projected_reserve={projected:.2f}MJ can_afford_aggressive={can_afford}"
        )

        return EnergyState(
            soc_mj=round(soc_mj, 3),
            soc_pct=soc_pct,
            remaining_mj=round(soc_mj, 3),
            deployed_this_lap_mj=round(deployed, 3),
            harvested_this_lap_mj=round(harvested, 3),
            deployment_headroom_mj=round(headroom, 3),
            projected_reserve_mj=round(projected, 3),
            projected_end_of_race_mj=round(end_of_race, 3),
            can_afford_aggressive=can_afford,
        )

    def calculate_deployment_headroom(
        self, deployed_this_lap_mj: float, has_bonus: bool = False
    ) -> float:
        """Remaining deployment budget before the per-lap cap."""
        settings = get_settings()
        cap = settings.max_deployment_per_lap_mj
        if has_bonus:
            cap += settings.overtake_bonus_mj
        return float(np.clip(cap - deployed_this_lap_mj, 0.0, cap))

    def calculate_projected_reserve(
        self,
        soc_mj: float,
        laps_remaining: int,
        deploy_per_lap: float = _EXPECTED_DEPLOY_PER_LAP_MJ,
        harvest_per_lap: float = _EXPECTED_HARVEST_PER_LAP_MJ,
    ) -> float:
        """
        projected_reserve = current_soc + expected_future_harvest - expected_future_deployment

        Assumes constant per-lap rates — a simplification. Real rates vary by circuit,
        fuel load, and driver style. Flagged as illustrative.
        """
        settings = get_settings()
        net_per_lap = harvest_per_lap - deploy_per_lap
        projected = soc_mj + laps_remaining * net_per_lap
        return float(np.clip(projected, 0.0, settings.energy_capacity_mj))

    def calculate_energy_budget(self, soc_mj: float, laps_remaining: int) -> float:
        """Total energy budget available over remaining laps."""
        settings = get_settings()
        projected_harvest = laps_remaining * _EXPECTED_HARVEST_PER_LAP_MJ
        return float(np.clip(soc_mj + projected_harvest, 0.0, settings.energy_capacity_mj * 2))
