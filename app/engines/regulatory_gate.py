"""
App-layer regulatory gate wrapper.

Tries to import from root-level rule_gate.py first.
Falls back to a standalone implementation using the same FIA 2026 constants.
"""

import logging
import sys
import os
from typing import Optional

from app.models.regulatory import RegulatoryCheck
from app.models.telemetry import TelemetryState

logger = logging.getLogger(__name__)

# Add repo root to path so we can import root-level modules
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class AppRegulatoryGate:
    """
    Evaluates FIA 2026 legality for each deployment mode.
    Wraps root-level rule_gate.py when available.
    """

    def __init__(self):
        self._gate = None
        try:
            from rule_gate import RegulatoryGate, GateConfig
            self._gate = RegulatoryGate(config=GateConfig())
            logger.debug("Using root-level rule_gate.py for regulatory checks")
        except ImportError:
            logger.warning("root-level rule_gate.py not found; using standalone constraints")

    def evaluate_mode(self, mode: str, telemetry: TelemetryState) -> RegulatoryCheck:
        """Evaluate legality of a single mode for the given telemetry."""
        if self._gate is not None:
            return self._evaluate_with_gate(mode, telemetry)
        return self._evaluate_standalone(mode, telemetry)

    def evaluate_all(self, telemetry: TelemetryState) -> dict[str, RegulatoryCheck]:
        """Evaluate all five deployment modes."""
        modes = [
            "CONSERVE_MODE",
            "BALANCED_MODE",
            "ARM_OVERTAKE_MODE",
            "USE_OVERTAKE_BONUS_MODE",
            "PUSH_MODE",
        ]
        return {mode: self.evaluate_mode(mode, telemetry) for mode in modes}

    def _evaluate_with_gate(self, mode: str, telemetry: TelemetryState) -> RegulatoryCheck:
        """Delegate to root-level RegulatoryGate."""
        from telemetry_simulator import TelemetryInput
        from rule_gate import DeploymentMode

        t_input = TelemetryInput(
            lap_number=telemetry.lap,
            current_soc_mj=telemetry.soc_mj,
            lap_start_soc_mj=telemetry.lap_start_soc_mj,
            lap_energy_deployed_mj=telemetry.energy_deployment_mj,
            gap_to_car_ahead_s=telemetry.gap_to_car_ahead_s,
            overtake_qualified_last_lap=telemetry.overtake_qualified_last_lap,
            speed_kmh=telemetry.speed_kmh,
            total_laps=telemetry.total_laps,
        )

        result = self._gate.evaluate(t_input)
        try:
            dm = DeploymentMode(mode)
            violations = result.violations.get(mode, [])
            legal = dm in result.legal_modes
        except ValueError:
            violations = [f"Unknown mode: {mode}"]
            legal = False

        return RegulatoryCheck(
            legal=legal,
            violations=violations,
            constraints_checked=[
                "Art.5.4.10 (deployment cap)",
                "Art.5.4.9 (SoC swing)",
                "Overtake proximity threshold",
                "Overtake bonus banking",
            ],
            mode=mode,
        )

    def _evaluate_standalone(self, mode: str, telemetry: TelemetryState) -> RegulatoryCheck:
        """Standalone legality check using FIA 2026 constants."""
        from app.core.config import get_settings
        settings = get_settings()

        violations: list[str] = []
        constraints = [
            "Art.5.4.10 (deployment cap)",
            "Art.5.4.9 (SoC swing)",
            "Overtake proximity threshold",
            "Overtake bonus banking",
        ]

        deployed = telemetry.energy_deployment_mj
        base_cap = settings.max_deployment_per_lap_mj
        bonus_cap = base_cap + settings.overtake_bonus_mj

        # Art. 5.4.10
        if mode == "USE_OVERTAKE_BONUS_MODE":
            if deployed > bonus_cap:
                violations.append(f"Deployment {deployed:.2f} MJ exceeds bonus cap {bonus_cap:.1f} MJ")
            if not telemetry.overtake_qualified_last_lap:
                violations.append("USE_OVERTAKE_BONUS_MODE requires overtake_qualified_last_lap=True")
        else:
            if deployed > base_cap:
                violations.append(f"Deployment {deployed:.2f} MJ exceeds Art.5.4.10 cap {base_cap:.1f} MJ")

        # Art. 5.4.9
        if telemetry.lap_start_soc_mj is not None:
            delta = abs(telemetry.lap_start_soc_mj - telemetry.soc_mj)
            if delta > settings.max_delta_soc_mj:
                violations.append(f"SoC swing {delta:.2f} MJ exceeds Art.5.4.9 limit {settings.max_delta_soc_mj:.1f} MJ")

        # ARM_OVERTAKE_MODE proximity
        if mode == "ARM_OVERTAKE_MODE":
            gap = telemetry.gap_to_car_ahead_s
            if gap is None or gap > settings.overtake_detection_gap_threshold_s:
                violations.append(f"Gap {gap}s exceeds proximity threshold {settings.overtake_detection_gap_threshold_s}s")

        return RegulatoryCheck(
            legal=len(violations) == 0,
            violations=violations,
            constraints_checked=constraints,
            mode=mode,
        )
