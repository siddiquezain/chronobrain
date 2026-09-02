"""
rule_gate.py — Stage 1: Regulatory Gate

Implements the FIA 2026 Power Unit Technical Regulations legality check for
ChronoPace's five deployment modes. This gate evaluates legality ONLY — not
viability. A mode with 0.1 MJ of remaining legal budget is still legal;
whether it's worth using is Stage 2's job.

IMPORTANT: This module must NEVER import from the rival estimator.
The legality check is based solely on the car's own verified telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from telemetry_simulator import TelemetryInput


# ---------------------------------------------------------------------------
# FIA 2026 Power Unit Technical Regulations — corrected constants.
# Article numbers cited so a reviewer can check without reading logic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    # Art. 5.4.7 — max instantaneous MGU-K power
    max_ers_k_power_kw: float = 350.0

    # Art. 5.4.10 — max energy from MGU-K HV DC bus per lap (deployment side)
    max_deployment_per_lap_mj: float = 9.0

    # Art. 5.4.9 — max SoC swing per lap
    max_delta_soc_mj: float = 4.0

    # Baseline recovery/harvest figure, event-variable.
    # citation pending — no confirmed FIA article for this specific figure.
    # Informational/config-only; does NOT gate deployment legality.
    recoverable_energy_baseline_mj: float = 8.5

    # F1 Sporting Regulations — proximity eligibility at detection point
    overtake_detection_gap_threshold_s: float = 1.0

    # Banked on the qualifying lap, spendable only the following lap
    overtake_bonus_mj: float = 0.5

    # Normal deployment taper band (km/h) — hardware behavior, all modes
    taper_normal_start_kmh: float = 290.0
    taper_normal_end_kmh: float = 355.0

    # Full 350 kW sustained to this speed under Overtake Mode bonus, then tapers
    taper_overtake_full_power_end_kmh: float = 337.0


class DeploymentMode(str, Enum):
    """
    ChronoPace's five deployment modes.
    These are backend enum values — do not rename without updating all consumers.
    """

    CONSERVE_MODE = "CONSERVE_MODE"
    BALANCED_MODE = "BALANCED_MODE"
    ARM_OVERTAKE_MODE = "ARM_OVERTAKE_MODE"
    USE_OVERTAKE_BONUS_MODE = "USE_OVERTAKE_BONUS_MODE"
    PUSH_MODE = "PUSH_MODE"

    @property
    def display_name(self) -> str:
        """Friendly UI label — not the backend enum value."""
        return {
            "CONSERVE_MODE": "CONSERVE",
            "BALANCED_MODE": "BALANCED",
            "ARM_OVERTAKE_MODE": "ARM",
            "USE_OVERTAKE_BONUS_MODE": "ATTACK",
            "PUSH_MODE": "PUSH",
        }[self.value]


class GateResult(BaseModel):
    """
    Output of RegulatoryGate.evaluate().

    legal_modes: modes that pass all legality checks for this telemetry snapshot.
    violations: dict — ALL 5 mode keys always present, [] for legal modes.
    base_cap_mj: dict — cap is mode-dependent (USE_OVERTAKE_BONUS_MODE gets +0.5 MJ).
    qualifies_for_overtake_bonus_next_lap: threads through to next lap's TelemetryInput.
    """

    legal_modes: list[DeploymentMode]
    violations: dict[str, list[str]]
    base_cap_mj: dict[str, float]
    qualifies_for_overtake_bonus_next_lap: bool


class RegulatoryGate:
    """
    Stage 1: Determines which DeploymentModes are legal given the current TelemetryInput.

    Architecture:
        Candidate Actions → Regulatory Gate → Legal Actions → Strategy Optimizer → Best Legal Action

    Legality is checked FIRST — never after Stage 2 or Stage 3.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(self, telemetry: TelemetryInput) -> GateResult:
        """
        Evaluate legality of all five deployment modes for the given telemetry.
        Returns GateResult with legal_modes, violations dict, per-mode caps, and
        next-lap overtake qualification flag.
        """
        cfg = self.config
        violations: dict[str, list[str]] = {m.value: [] for m in DeploymentMode}

        base_cap = cfg.max_deployment_per_lap_mj
        bonus_cap = base_cap + cfg.overtake_bonus_mj

        base_cap_mj = {m.value: base_cap for m in DeploymentMode}
        base_cap_mj[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value] = bonus_cap

        # --- Art. 5.4.10: per-lap deployment cap ---
        deployed = telemetry.lap_energy_deployed_mj
        if deployed > base_cap:
            for mode in [
                DeploymentMode.CONSERVE_MODE,
                DeploymentMode.BALANCED_MODE,
                DeploymentMode.ARM_OVERTAKE_MODE,
                DeploymentMode.PUSH_MODE,
            ]:
                violations[mode.value].append(
                    f"Lap deployment {deployed:.2f} MJ exceeds Art.5.4.10 cap {base_cap:.1f} MJ"
                )
            if deployed > bonus_cap:
                violations[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value].append(
                    f"Lap deployment {deployed:.2f} MJ exceeds bonus cap {bonus_cap:.1f} MJ"
                )

        # --- Art. 5.4.9: delta-SoC swing ---
        # Fails open when lap_start_soc_mj is None — deliberate demo-usability choice.
        # In a safety-certified build this should fail closed.
        if telemetry.lap_start_soc_mj is not None:
            delta_soc = abs(telemetry.lap_start_soc_mj - telemetry.current_soc_mj)
            if delta_soc > cfg.max_delta_soc_mj:
                for mode in DeploymentMode:
                    violations[mode.value].append(
                        f"SoC swing {delta_soc:.2f} MJ exceeds Art.5.4.9 limit "
                        f"{cfg.max_delta_soc_mj:.1f} MJ"
                    )

        # --- ARM_OVERTAKE_MODE requires proximity at detection point ---
        gap = telemetry.gap_to_car_ahead_s
        if gap is None or gap > cfg.overtake_detection_gap_threshold_s:
            violations[DeploymentMode.ARM_OVERTAKE_MODE.value].append(
                f"Gap {gap}s exceeds overtake detection threshold "
                f"{cfg.overtake_detection_gap_threshold_s}s"
            )

        # --- USE_OVERTAKE_BONUS_MODE requires banked qualification from last lap ---
        # Does NOT re-check the gap — only the banked flag.
        # Spending a banked bonus after the gap has since closed is legal.
        if not telemetry.overtake_qualified_last_lap:
            violations[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value].append(
                "USE_OVERTAKE_BONUS_MODE requires overtake_qualified_last_lap=True "
                "(bonus not banked from previous lap)"
            )

        # --- Qualification for next lap ---
        # Qualification is a telemetry fact, not a mode-choice fact.
        qualifies_next = gap is not None and gap <= cfg.overtake_detection_gap_threshold_s

        legal_modes = [m for m in DeploymentMode if not violations[m.value]]

        return GateResult(
            legal_modes=legal_modes,
            violations=violations,
            base_cap_mj=base_cap_mj,
            qualifies_for_overtake_bonus_next_lap=qualifies_next,
        )
