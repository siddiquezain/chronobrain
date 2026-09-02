"""Scenario configurations for the ChronoPace demo."""

from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class ScenarioConfig:
    scenario: str
    seed: int
    total_laps: int = 50
    current_lap: int = 25
    initial_soc_mj: float = 5.0
    initial_position: int = 5
    tyre_compound: str = "MEDIUM"
    tyre_age_laps: int = 15
    gap_to_car_ahead_s: float = 2.0
    gap_to_car_behind_s: float = 2.0
    closing_speed_mps: float = 2.0
    slipstream_factor: float = 0.3
    lap_energy_deployed_mj: float = 1.0
    overtake_qualified_last_lap: bool = False
    straight_distance_m: float = 500.0
    braking_zone_quality: float = 0.7


SCENARIO_PRESETS: dict[str, ScenarioConfig] = {
    # A — Normal race, balanced, no strong overtake. Expected: BALANCED_MODE
    "A": ScenarioConfig(
        scenario="A", seed=42,
        initial_soc_mj=5.5, gap_to_car_ahead_s=2.5,
        closing_speed_mps=1.5, slipstream_factor=0.2,
        lap_energy_deployed_mj=2.0,
    ),
    # B — Strong overtake window. Expected: ARM or USE_OVERTAKE_BONUS_MODE
    "B": ScenarioConfig(
        scenario="B", seed=42,
        initial_soc_mj=7.2, gap_to_car_ahead_s=0.6,
        closing_speed_mps=8.4, slipstream_factor=0.81,
        lap_energy_deployed_mj=1.0,
        overtake_qualified_last_lap=True,
    ),
    # C — Low energy. Expected: BALANCED_MODE or CONSERVE_MODE
    "C": ScenarioConfig(
        scenario="C", seed=42,
        initial_soc_mj=1.8, gap_to_car_ahead_s=0.7,
        closing_speed_mps=6.0, slipstream_factor=0.65,
        lap_energy_deployed_mj=1.0,
    ),
    # D — Defensive situation. Expected: PUSH_MODE or BALANCED_MODE
    "D": ScenarioConfig(
        scenario="D", seed=42,
        initial_soc_mj=5.0, gap_to_car_ahead_s=3.0,
        gap_to_car_behind_s=0.4,
        closing_speed_mps=0.5, slipstream_factor=0.2,
        lap_energy_deployed_mj=2.5,
    ),
    # E — Illegal candidate (deployment over cap). Expected: REJECTED BY REGULATORY GATE
    "E": ScenarioConfig(
        scenario="E", seed=42,
        initial_soc_mj=4.0, gap_to_car_ahead_s=0.8,
        lap_energy_deployed_mj=9.2,  # OVER ART.5.4.10 CAP
        overtake_qualified_last_lap=True,
    ),
}


def get_scenario_config(scenario: str, seed: Optional[int] = None) -> ScenarioConfig:
    if scenario not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIO_PRESETS)}")
    config = SCENARIO_PRESETS[scenario]
    if seed is not None:
        config = replace(config, seed=seed)
    return config
