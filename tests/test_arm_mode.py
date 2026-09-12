"""
Regression: ARM_OVERTAKE_MODE must be reachable as a final decision.

ARM means "attack this lap while within proximity, which also qualifies next lap's
overtake bonus". It is the right mode when an aggressive move is justified, the
bonus is NOT already banked, ARM is legal, and confidence permits it — the case
where USE_OVERTAKE_BONUS_MODE cannot apply (nothing banked to spend) and PUSH_MODE
would burn energy without the bonus-qualification benefit.

None of this forces ARM; it just stops the planner's raw laptime ranking (which
always prefers PUSH/USE_BONUS) from being the only thing that picks the mode.
"""

import pytest

from app.data.providers import ReplayProvider
from app.data.samples import NormalizedLap
from app.decision import DecisionConfig, run_scenario
from app.decision.engine import run_decision


def _arm_provider() -> ReplayProvider:
    """
    Oscillating gap: on odd laps we are within the 1.0 s proximity threshold, on
    even laps we are not. So on lap 9 we are in proximity but the previous lap
    (10 -> gap 1.4) did NOT qualify => no bonus banked => ARM, not USE_BONUS.
    High SoC throughout so an attack is affordable.
    """
    gaps = [0.5, 1.3, 0.5, 1.3, 0.5, 1.3, 0.5, 1.3, 0.6, 1.4, 0.5, 1.3]
    laps = []
    soc = 6.5
    for i, g in enumerate(gaps, start=1):
        laps.append(NormalizedLap(
            lap=i, total_laps=50, data_mode="SYNTHETIC",
            our_speed_kmh=295.0, our_soc_mj=soc, our_lap_start_soc_mj=soc + 0.1,
            our_lap_energy_deployed_mj=1.6, gap_to_car_ahead_s=g,
            rival_terminal_speed_kmh=302.0, rival_clipping_point_fraction=0.42,
            rival_corner_exit_accel_g=1.05, rival_sector_delta_s=0.03,
            energy_is_modeled=False,
        ))
        soc = max(4.0, soc - 0.03)
    return ReplayProvider(laps, "arm-test")


@pytest.mark.parametrize("lap", [9, 11])
def test_arm_is_selected_when_in_proximity_without_a_banked_bonus(lap):
    s = run_decision(_arm_provider(), lap=lap, config=DecisionConfig(seed=42))
    assert s.decision.mode == "ARM_OVERTAKE_MODE"
    assert s.decision.action == "ATTACK_NOW"
    assert not s.decision.confidence_overridden
    assert "ARMING_OVERTAKE_BONUS" in s.reason_codes
    # ARM is legal, USE_OVERTAKE_BONUS is not (nothing banked)
    assert "ARM_OVERTAKE_MODE" in s.compliance.legal_modes
    assert "USE_OVERTAKE_BONUS_MODE" not in s.compliance.legal_modes


def test_arm_determinism():
    a = run_decision(_arm_provider(), lap=9, config=DecisionConfig(seed=42))
    b = run_decision(_arm_provider(), lap=9, config=DecisionConfig(seed=42))
    assert a.deterministic_dict() == b.deterministic_dict()


def test_scenario_b_lap1_arms_the_bonus():
    """Lap 1: within proximity, no previous lap to have banked a bonus -> ARM."""
    s = run_scenario("B", seed=42, lap=1, total_laps=50)
    assert s.decision.mode == "ARM_OVERTAKE_MODE"
    assert s.decision.action == "ATTACK_NOW"


def test_arm_gives_way_to_use_bonus_once_banked():
    """Lap 2+ of scenario B: bonus banked from lap 1 -> spend it, not ARM."""
    s = run_scenario("B", seed=42, lap=2, total_laps=50)
    assert s.decision.mode == "USE_OVERTAKE_BONUS_MODE"


def test_all_five_modes_are_reachable():
    """Every deployment mode must be selectable by some legitimate situation."""
    seen = set()
    # scenarios A-D across many laps/seeds
    for sc in "ABCD":
        for seed in (1, 42, 7):
            for lap in range(1, 46, 3):
                seen.add(run_scenario(sc, seed=seed, lap=lap, total_laps=50).decision.mode)
    # ARM via the crafted proximity-without-bonus provider
    seen.add(run_decision(_arm_provider(), lap=9, config=DecisionConfig(seed=42)).decision.mode)
    for mode in (
        "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
        "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
    ):
        assert mode in seen, f"{mode} was never selected as a final decision"
