"""
Golden snapshots for the five demo scenarios at a fixed lap/seed.

These lock in the *decision-relevant* fields (not the full snapshot, which carries
many Monte Carlo floats). If a change moves one of these, it should be a conscious
choice — update the golden dict in the same commit and say why.
"""

import pytest

from app.decision import run_scenario

# scenario -> (mode, action, compliance.legal, confidence_overridden)
GOLDEN = {
    "A": ("BALANCED_MODE", "HOLD", True, False),
    "B": ("USE_OVERTAKE_BONUS_MODE", "ATTACK_NOW", True, False),
    "C": ("CONSERVE_MODE", "CONSERVE", True, False),
    "D": ("PUSH_MODE", "PUSH", True, False),
    # E: every mode illegal -> gate rejects; confidence gate also reports overridden
    # (it has no ranked modes to establish significance on).
    "E": ("BALANCED_MODE", "HOLD", False, True),
}


@pytest.mark.parametrize("scenario,expected", GOLDEN.items())
def test_scenario_golden_decision(scenario, expected):
    s = run_scenario(scenario, seed=42, total_laps=50, lap=30)
    actual = (
        s.decision.mode,
        s.decision.action,
        s.compliance.legal,
        s.decision.confidence_overridden,
    )
    assert actual == expected


@pytest.mark.parametrize("scenario", list(GOLDEN))
def test_scenario_snapshot_is_well_formed(scenario):
    s = run_scenario(scenario, seed=42, total_laps=50, lap=30)
    assert s.monte_carlo.n_iterations == 10_000
    assert abs(sum(s.rival.distribution.values()) - 1.0) < 1e-6
    assert s.reason_codes and len(s.reason_codes) == len(s.reasons)
    assert s.trace[-1].stage == "FINAL"
    assert s.trace[0].stage == "Telemetry"
