"""
Risk Engine — independent deterministic risk assessment.

Risk increases when:
  - gap is large (harder to complete overtake)
  - closing speed is weak
  - braking opportunity is poor
  - energy reserve is low (< 2 MJ → high energy_risk)
  - proposed deployment is excessive
  - track position is unfavorable

All risk values in [0, 1]. Overall risk = weighted average.
"""

import logging
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.models.energy import EnergyState
from app.models.overtake import OvertakeAnalysis
from app.models.telemetry import TelemetryState

logger = logging.getLogger(__name__)

_LOW_ENERGY_THRESHOLD_MJ = 2.0
_HIGH_DEPLOYMENT_THRESHOLD_MJ = 7.5


class RiskEngine:
    """Deterministic and explainable risk assessor."""

    def assess(
        self,
        telemetry: TelemetryState,
        energy: EnergyState,
        overtake: OvertakeAnalysis,
        proposed_mode: str = "BALANCED_MODE",
    ) -> dict:
        """
        Returns dict with overtake_risk, energy_risk, strategic_risk, overall_risk.
        All values in [0, 1].
        """
        settings = get_settings()

        # --- Overtake risk ---
        # High when gap is large, closing speed weak, braking poor
        gap_risk = float(np.clip(1.0 - overtake.gap_score, 0.0, 1.0))
        speed_risk = float(np.clip(1.0 - overtake.closing_speed_score, 0.0, 1.0))
        braking_risk = float(np.clip(1.0 - overtake.braking_zone_score, 0.0, 1.0))
        overtake_risk = float((gap_risk * 0.4 + speed_risk * 0.35 + braking_risk * 0.25))

        # --- Energy risk ---
        # High when SoC is low or deployment is excessive
        soc_risk = float(np.clip(1.0 - energy.soc_mj / 9.0, 0.0, 1.0))
        if energy.soc_mj < _LOW_ENERGY_THRESHOLD_MJ:
            soc_risk = min(1.0, soc_risk * 2.0)  # amplify at low SoC

        deploy_risk = float(
            np.clip(energy.deployed_this_lap_mj / _HIGH_DEPLOYMENT_THRESHOLD_MJ, 0.0, 1.0)
        )
        headroom_risk = float(
            np.clip(1.0 - energy.deployment_headroom_mj / 9.0, 0.0, 1.0)
        )
        energy_risk = float((soc_risk * 0.5 + deploy_risk * 0.3 + headroom_risk * 0.2))

        # --- Strategic risk ---
        # High when track position is unfavorable or mode is aggressive on low reserve
        mode_risk = 0.0
        if proposed_mode in ("USE_OVERTAKE_BONUS_MODE", "ARM_OVERTAKE_MODE", "PUSH_MODE"):
            mode_risk = float(np.clip(1.0 - energy.projected_reserve_mj / 3.0, 0.0, 1.0))
        position_risk = float(np.clip(telemetry.track_position * 0.5, 0.0, 0.5))
        strategic_risk = float((mode_risk * 0.7 + position_risk * 0.3))

        # --- Overall risk (weighted) ---
        overall_risk = float(
            settings.risk_weight_overtake * overtake_risk
            + settings.risk_weight_energy * energy_risk
            + settings.risk_weight_strategic * strategic_risk
        )

        logger.debug(
            f"Risk: overtake={overtake_risk:.3f} energy={energy_risk:.3f} "
            f"strategic={strategic_risk:.3f} overall={overall_risk:.3f}"
        )

        return {
            "overtake_risk": round(overtake_risk, 4),
            "energy_risk": round(energy_risk, 4),
            "strategic_risk": round(strategic_risk, 4),
            "overall_risk": round(float(np.clip(overall_risk, 0.0, 1.0)), 4),
        }
