"""
End-to-end determinism: same scenario + same config + same seed => identical
snapshot (excluding the wall-clock meta field). This covers the WHOLE pipeline,
not just the Monte Carlo planner.
"""

import pytest

from app.decision import DecisionConfig, run_scenario
from app.decision.engine import run_decision
from app.data.providers import build_provider

SCENARIOS = ["A", "B", "C", "D", "E"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_seed_same_snapshot(scenario):
    a = run_scenario(scenario, seed=42, total_laps=50, lap=25)
    b = run_scenario(scenario, seed=42, total_laps=50, lap=25)
    assert a.deterministic_dict() == b.deterministic_dict()


@pytest.mark.parametrize("scenario", ["A", "B", "C", "D"])  # E has no legal modes -> no MC samples
def test_different_seed_can_differ(scenario):
    a = run_scenario(scenario, seed=1, total_laps=50, lap=25)
    b = run_scenario(scenario, seed=99999, total_laps=50, lap=25)
    # Monte Carlo std must move with the seed even if the final mode is stable.
    a_std = [m.std_laptime_delta_s for m in a.monte_carlo.ranked_modes]
    b_std = [m.std_laptime_delta_s for m in b.monte_carlo.ranked_modes]
    assert a_std and a_std != b_std


def test_determinism_holds_across_every_lap():
    p1 = build_provider("synthetic", scenario="B", seed=7, total_laps=30)
    p2 = build_provider("synthetic", scenario="B", seed=7, total_laps=30)
    cfg = DecisionConfig(seed=7)
    for lap in range(1, 31):
        s1 = run_decision(p1, lap=lap, config=cfg)
        s2 = run_decision(p2, lap=lap, config=cfg)
        assert s1.deterministic_dict() == s2.deterministic_dict(), f"lap {lap} not deterministic"


def test_seed_is_reported_in_snapshot():
    s = run_scenario("B", seed=123, total_laps=50, lap=10)
    assert s.meta.seed == 123
    assert s.monte_carlo.seed == 123
