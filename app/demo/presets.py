"""
Demo presets — named (scenario, overrides, lap) bundles for the control room.

A preset is INPUTS ONLY. The deterministic Python pipeline produces the
recommendation; `expectation` is a human note about what the preset is meant to
illustrate, not a hardcoded answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class DemoPreset:
    key: str
    label: str
    description: str
    scenario: str
    lap: int
    overrides: dict = field(default_factory=dict)
    rival_obs_dropout: float = 0.0
    telemetry_glitch: bool = False
    seed: int = 42
    total_laps: int = 50
    expectation: str = ""
    pair_group: str = ""  # presets sharing this are meant to be shown side by side


DEMO_PRESETS: Dict[str, DemoPreset] = {
    "HIGH_ENERGY_STRONG_OPPORTUNITY": DemoPreset(
        key="HIGH_ENERGY_STRONG_OPPORTUNITY",
        label="High energy · strong opportunity",
        description="High own SoC, small gap, bonus banked. The clean 'go' case.",
        scenario="B", lap=25,
        expectation="Aggressive action (USE_OVERTAKE_BONUS / ATTACK_NOW) when the "
                    "confidence gate and regulations allow it.",
    ),
    "LIMITED_ENERGY_SAME_OPPORTUNITY": DemoPreset(
        key="LIMITED_ENERGY_SAME_OPPORTUNITY",
        label="Limited energy · same opportunity",
        description="Identical overtake window, but our SoC is scarce.",
        scenario="B", lap=25,
        overrides={"initial_soc_mj": 1.6},
        expectation="The optimal action changes because the energy isn't worth "
                    "spending — CONSERVE / hold instead of attacking.",
    ),
    "BETTER_FUTURE_OPPORTUNITY": DemoPreset(
        key="BETTER_FUTURE_OPPORTUNITY",
        label="Better future opportunity",
        description="Window is open now but energy recovers into a stronger one soon.",
        scenario="C", lap=22,
        expectation="The Opportunity Horizon defers: WAIT_N rather than attacking now.",
    ),
    "LOW_CONFIDENCE_BAD_DATA": DemoPreset(
        key="LOW_CONFIDENCE_BAD_DATA",
        label="Low confidence · bad data",
        description="Telemetry feed is corrupted (missing laps + out-of-range values).",
        scenario="B", lap=24, telemetry_glitch=True,
        expectation="The Data Quality Gate degrades and the Confidence Gate abstains "
                    "-> BALANCED_MODE / HOLD, override reason DATA_QUALITY.",
    ),
    "RIVAL_ENERGY_HIGH": DemoPreset(
        key="RIVAL_ENERGY_HIGH",
        label="Rival energy: HIGH (hidden)",
        description="Same opportunity; hidden rival SoC set HIGH. Estimator must infer it.",
        scenario="B", lap=28, pair_group="RIVAL_ENERGY_CHANGE",
        overrides={"rival_initial_soc_mj": 8.5, "initial_soc_mj": 5.0, "gap_to_car_ahead_s": 0.6},
        expectation="Rival Energy Estimator posterior moves HIGH; bucket/p_defend rise. "
                    "Compare against RIVAL_ENERGY_LOW with the same visible conditions.",
    ),
    "RIVAL_ENERGY_LOW": DemoPreset(
        key="RIVAL_ENERGY_LOW",
        label="Rival energy: LOW (hidden)",
        description="Same opportunity; hidden rival SoC set LOW. Estimator must infer it.",
        scenario="B", lap=28, pair_group="RIVAL_ENERGY_CHANGE",
        overrides={"rival_initial_soc_mj": 1.0, "initial_soc_mj": 5.0, "gap_to_car_ahead_s": 0.6},
        expectation="Rival Energy Estimator posterior moves LOW/MEDIUM from the SAME "
                    "visible gap and a similar speed profile — evidence-driven.",
    ),
}


def resolve_preset(key: str) -> DemoPreset:
    if key not in DEMO_PRESETS:
        raise KeyError(f"unknown preset {key!r}; choose from {sorted(DEMO_PRESETS)}")
    return DEMO_PRESETS[key]
