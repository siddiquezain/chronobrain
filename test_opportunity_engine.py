"""Tests for opportunity_engine.py — Opportunity Horizon"""

import pytest
from opportunity_engine import HorizonConfig, HorizonResult, OpportunityEngine, HOLD_STRATEGY
from rival_estimator import RivalSocEstimate
from rule_gate import DeploymentMode, GateConfig, GateResult
from telemetry_simulator import TelemetryInput


def make_all_legal_gate() -> GateResult:
    cfg = GateConfig()
    base_cap = cfg.max_deployment_per_lap_mj
    bonus_cap = base_cap + cfg.overtake_bonus_mj
    base_cap_mj = {m.value: base_cap for m in DeploymentMode}
    base_cap_mj[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value] = bonus_cap
    return GateResult(
        legal_modes=list(DeploymentMode),
        violations={m.value: [] for m in DeploymentMode},
        base_cap_mj=base_cap_mj,
        qualifies_for_overtake_bonus_next_lap=True,
    )


def make_telemetry() -> TelemetryInput:
    return TelemetryInput(
        lap_number=25,
        current_soc_mj=7.0,
        lap_start_soc_mj=7.5,
        lap_energy_deployed_mj=1.0,
        gap_to_car_ahead_s=0.7,
        overtake_qualified_last_lap=True,
        speed_kmh=290.0,
        total_laps=50,
    )


class TestHorizonResult:
    def test_has_ranked_strategies(self):
        engine = OpportunityEngine(seed=42)
        result = engine.evaluate(make_telemetry(), make_all_legal_gate())
        assert isinstance(result.ranked_strategies, list)
        assert len(result.ranked_strategies) > 0

    def test_hold_strategy_present(self):
        engine = OpportunityEngine(seed=42)
        result = engine.evaluate(make_telemetry(), make_all_legal_gate())
        names = {s.strategy_name for s in result.ranked_strategies}
        assert HOLD_STRATEGY in names

    def test_attack_now_present(self):
        engine = OpportunityEngine(seed=42)
        result = engine.evaluate(make_telemetry(), make_all_legal_gate())
        names = {s.strategy_name for s in result.ranked_strategies}
        assert "ATTACK_NOW" in names

    def test_foregone_value_gap_non_negative(self):
        engine = OpportunityEngine(seed=42)
        result = engine.evaluate(make_telemetry(), make_all_legal_gate())
        assert result.foregone_value_gap_s >= 0.0

    def test_uncertainty_note_present(self):
        engine = OpportunityEngine(seed=42)
        result = engine.evaluate(make_telemetry(), make_all_legal_gate())
        assert len(result.uncertainty_note) > 20


class TestUncertaintyWidening:
    def test_wait_5_std_larger_than_wait_2(self):
        """Uncertainty MUST widen with distance — WAIT_5 must have larger std than WAIT_2."""
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=42)
        result = engine.evaluate(make_telemetry(), make_all_legal_gate())

        wait_2 = next((s for s in result.ranked_strategies if s.strategy_name == "WAIT_2"), None)
        wait_5 = next((s for s in result.ranked_strategies if s.strategy_name == "WAIT_5"), None)

        if wait_2 and wait_5:
            assert wait_5.std_horizon_delta_s >= wait_2.std_horizon_delta_s, (
                "WAIT_5 std must be >= WAIT_2 std — uncertainty must widen with distance"
            )

    def test_attack_now_better_than_hold_with_high_soc(self):
        """With high SoC and gap within threshold, ATTACK_NOW should beat HOLD."""
        engine = OpportunityEngine(seed=42)
        gate = make_all_legal_gate()
        result = engine.evaluate(make_telemetry(), gate)

        attack = next((s for s in result.ranked_strategies if s.strategy_name == "ATTACK_NOW"), None)
        hold = next((s for s in result.ranked_strategies if s.strategy_name == HOLD_STRATEGY), None)

        if attack and hold:
            # ATTACK_NOW uses aggressive modes → should have lower (better) mean delta
            assert attack.mean_horizon_delta_s < hold.mean_horizon_delta_s


class TestDeterminism:
    def test_same_seed_same_result(self):
        gate = make_all_legal_gate()
        t = make_telemetry()

        r1 = OpportunityEngine(seed=42).evaluate(t, gate)
        r2 = OpportunityEngine(seed=42).evaluate(t, gate)

        assert r1.recommended_strategy == r2.recommended_strategy
        for s1, s2 in zip(r1.ranked_strategies, r2.ranked_strategies):
            assert s1.mean_horizon_delta_s == s2.mean_horizon_delta_s
