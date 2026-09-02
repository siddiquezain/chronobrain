"""
Overtake Engine — scored overtake opportunity detection.

Score formula (weights from config, not hardcoded):
    score = w_closing * closing_speed_score
          + w_gap * gap_score
          + w_braking * braking_zone_score
          + w_slipstream * slipstream_score
          + w_corner * corner_exit_score
          + w_energy * energy_advantage_score

All individual component scores are in [0, 1]. Final score is in [0, 1].
Probability is derived from score — not independently computed.
"""

import logging
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.models.energy import EnergyState
from app.models.overtake import OvertakeAnalysis
from app.models.telemetry import TelemetryState

logger = logging.getLogger(__name__)

# Rival SoC estimate used for energy advantage when rival estimator is not wired up
# ponytail: replace with rival_estimator.py output once integrated
_DEFAULT_RIVAL_SOC_MJ = 4.5

_MAX_GAP_S = 3.0  # gaps beyond this score 0
_MAX_CLOSING_SPEED_MPS = 15.0
_SCORE_FACTOR = 0.6  # overtake_factor converts score → probability (sigmoid-like)


class OvertakeEngine:
    """
    Deterministic and explainable overtake opportunity scorer.
    Weights are configurable via Settings — not hardcoded "magic numbers."
    """

    def analyze(
        self, telemetry: TelemetryState, energy: EnergyState
    ) -> OvertakeAnalysis:
        settings = get_settings()

        # --- Component scores (each in [0,1]) ---
        gap_s = telemetry.gap_to_car_ahead_s

        gap_score = (
            float(np.clip(1.0 - gap_s / _MAX_GAP_S, 0.0, 1.0))
            if gap_s is not None
            else 0.0
        )

        closing_score = float(
            np.clip(telemetry.closing_speed_mps / _MAX_CLOSING_SPEED_MPS, 0.0, 1.0)
        )

        braking_score = self._braking_zone_score(telemetry)
        corner_score = self._corner_exit_score(telemetry)
        slipstream_score = float(np.clip(telemetry.slipstream_factor, 0.0, 1.0))

        energy_advantage = energy.soc_mj - _DEFAULT_RIVAL_SOC_MJ
        energy_score = float(np.clip((energy_advantage + 4.5) / 9.0, 0.0, 1.0))

        # --- Weighted sum ---
        total_weight = (
            settings.overtake_weight_closing_speed
            + settings.overtake_weight_gap
            + settings.overtake_weight_braking_zone
            + settings.overtake_weight_slipstream
            + settings.overtake_weight_corner_exit
            + settings.overtake_weight_energy_advantage
        )
        score = (
            settings.overtake_weight_closing_speed * closing_score
            + settings.overtake_weight_gap * gap_score
            + settings.overtake_weight_braking_zone * braking_score
            + settings.overtake_weight_slipstream * slipstream_score
            + settings.overtake_weight_corner_exit * corner_score
            + settings.overtake_weight_energy_advantage * energy_score
        ) / total_weight

        score = float(np.clip(score, 0.0, 1.0))

        # Probability derived from score (sigmoid-like, not a separate model here)
        probability = float(np.clip(score * 1.1 - 0.05, 0.0, 1.0))

        # Human-readable contributing factors
        factors: list[str] = []
        if closing_score > 0.6:
            factors.append(f"HIGH_CLOSING_SPEED ({telemetry.closing_speed_mps:.1f} m/s)")
        if gap_score > 0.7:
            factors.append(f"SMALL_GAP ({gap_s:.2f}s)" if gap_s else "NO_GAP_DATA")
        if slipstream_score > 0.6:
            factors.append(f"STRONG_SLIPSTREAM ({telemetry.slipstream_factor:.2f})")
        if braking_score > 0.7:
            factors.append("FAVORABLE_BRAKING_ZONE")
        if energy_score > 0.6:
            factors.append(f"ENERGY_ADVANTAGE ({energy_advantage:+.1f} MJ)")

        logger.debug(
            f"Overtake analysis: score={score:.3f} prob={probability:.3f} "
            f"factors={factors}"
        )

        return OvertakeAnalysis(
            score=round(score, 4),
            probability=round(probability, 4),
            gap_s=gap_s,
            closing_speed_mps=telemetry.closing_speed_mps,
            slipstream_factor=telemetry.slipstream_factor,
            braking_zone_score=round(braking_score, 4),
            corner_exit_score=round(corner_score, 4),
            energy_advantage_score=round(energy_score, 4),
            contributing_factors=factors,
            closing_speed_score=round(closing_score, 4),
            gap_score=round(gap_score, 4),
        )

    def _braking_zone_score(self, t: TelemetryState) -> float:
        """Estimate braking zone quality from distance to corner and speed."""
        dist_score = float(np.clip(1.0 - t.distance_to_next_corner_m / 500.0, 0.0, 1.0))
        speed_score = float(np.clip(t.speed_kmh / 300.0, 0.0, 1.0))
        return float((dist_score + speed_score) / 2.0)

    def _corner_exit_score(self, t: TelemetryState) -> float:
        """Estimate corner exit quality from throttle and brake inputs."""
        return float(np.clip(t.throttle * 0.8 + (1.0 - t.brake) * 0.2, 0.0, 1.0))
