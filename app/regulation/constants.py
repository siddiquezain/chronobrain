"""
constants.py — provenance catalogue keyed by GateConfig field name.

Each entry says where the number comes from and whether it still needs checking
against the final published 2026 FIA regulations (this was built in 2026 from
public F1 / FIA communications, not the confidential technical regulation text).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Provenance(str, Enum):
    VERIFIED_FIA = "VERIFIED_FIA"
    MODEL_ASSUMPTION = "MODEL_ASSUMPTION"
    DEMO_CONSTANT = "DEMO_CONSTANT"


@dataclass(frozen=True)
class RegConstant:
    gate_config_field: Optional[str]
    value: float
    unit: str
    provenance: Provenance
    needs_verification: bool
    note: str
    source: str = ""


# Grounded against public 2026 F1 / FIA communications — see docs/regulation.md.
REGULATORY_CONSTANTS: dict[str, RegConstant] = {
    "max_ers_k_power_kw": RegConstant(
        "max_ers_k_power_kw", 350.0, "kW",
        Provenance.VERIFIED_FIA, False,
        "MGU-K maximum electrical power for 2026 (up from 120 kW). Widely published.",
        "formula1.com 2026 power unit explainer",
    ),
    "overtake_bonus_mj": RegConstant(
        "overtake_bonus_mj", 0.5, "MJ",
        Provenance.VERIFIED_FIA, False,
        "Additional deployable energy under 2026 Override / Overtake Mode when within "
        "1.0 s of the car ahead. Officially stated as 0.5 MJ.",
        "F1 2026 Overtake Mode communications",
    ),
    "overtake_detection_gap_threshold_s": RegConstant(
        "overtake_detection_gap_threshold_s", 1.0, "s",
        Provenance.VERIFIED_FIA, False,
        "Proximity threshold to become eligible for Override / Overtake Mode "
        "(replaces the DRS 1.0 s detection rule).",
        "F1 2026 Overtake Mode communications",
    ),
    "taper_overtake_full_power_end_kmh": RegConstant(
        "taper_overtake_full_power_end_kmh", 337.0, "km/h",
        Provenance.VERIFIED_FIA, True,
        "Speed to which full 350 kW is sustained under Overtake Mode before taper. "
        "337 km/h is the officially quoted figure; the taper curve shape below it is modelled.",
        "F1 2026 Overtake Mode communications",
    ),
    "max_delta_soc_mj": RegConstant(
        "max_delta_soc_mj", 4.0, "MJ",
        Provenance.MODEL_ASSUMPTION, True,
        "The 4 MJ figure IS verified (usable energy stored in the battery at any one "
        "time is capped at 4 MJ for 2026). Applying it as a per-lap SoC-SWING limit "
        "in the gate is a ChronoPace modelling choice, not a literal Art. 5.4.9 clause.",
        "raceteq / ESPN 2026 energy system explainers",
    ),
    "max_deployment_per_lap_mj": RegConstant(
        "max_deployment_per_lap_mj", 9.0, "MJ",
        Provenance.MODEL_ASSUMPTION, True,
        "Per-lap electrical deployment cap. Public figures put recoverable/deployable "
        "energy at ~8–9 MJ/lap depending on circuit; 9.0 is the top-of-range value used "
        "as a single fixed cap. Not a confirmed universal constant.",
        "raceteq / ESPN 2026 energy system explainers",
    ),
    "recoverable_energy_baseline_mj": RegConstant(
        "recoverable_energy_baseline_mj", 8.5, "MJ",
        Provenance.MODEL_ASSUMPTION, True,
        "Nominal per-lap harvest figure (~8.5 MJ, circuit-variable 5–9 MJ). "
        "Informational only — does NOT gate deployment legality.",
        "raceteq / ESPN 2026 energy system explainers",
    ),
    "taper_normal_start_kmh": RegConstant(
        "taper_normal_start_kmh", 290.0, "km/h",
        Provenance.MODEL_ASSUMPTION, True,
        "Start of the normal-deployment power taper band. Engineered estimate of PU "
        "hardware behaviour, not a published number.",
    ),
    "taper_normal_end_kmh": RegConstant(
        "taper_normal_end_kmh", 355.0, "km/h",
        Provenance.MODEL_ASSUMPTION, True,
        "End of the normal-deployment power taper band. Engineered estimate.",
    ),
}


def provenance_of(gate_config_field: str) -> Provenance:
    entry = REGULATORY_CONSTANTS.get(gate_config_field)
    return entry.provenance if entry else Provenance.MODEL_ASSUMPTION
