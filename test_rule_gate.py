"""Tests for rule_gate.py — Stage 1: Regulatory Gate"""

import pytest
from rule_gate import RegulatoryGate, GateConfig, DeploymentMode, GateResult
from telemetry_simulator import TelemetryInput


def make_telemetry(**kwargs) -> TelemetryInput:
    defaults = dict(
        lap_number=10,
        current_soc_mj=5.0,
        lap_start_soc_mj=6.0,
        lap_energy_deployed_mj=1.0,
        gap_to_car_ahead_s=2.0,
        overtake_qualified_last_lap=False,
        speed_kmh=280.0,
        total_laps=50,
    )
    defaults.update(kwargs)
    return TelemetryInput(**defaults)


class TestLegalModes:
    def test_baseline_telemetry_legal_modes(self):
        """Normal telemetry: CONSERVE, BALANCED, PUSH legal; ARM/USE_BONUS illegal."""
        gate = RegulatoryGate()
        result = gate.evaluate(make_telemetry())
        assert DeploymentMode.CONSERVE_MODE in result.legal_modes
        assert DeploymentMode.BALANCED_MODE in result.legal_modes
        assert DeploymentMode.PUSH_MODE in result.legal_modes
        assert DeploymentMode.ARM_OVERTAKE_MODE not in result.legal_modes
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE not in result.legal_modes

    def test_all_five_modes_in_violations_dict(self):
        gate = RegulatoryGate()
        result = gate.evaluate(make_telemetry())
        assert set(result.violations.keys()) == {m.value for m in DeploymentMode}

    def test_legal_modes_have_empty_violations(self):
        gate = RegulatoryGate()
        result = gate.evaluate(make_telemetry())
        for mode in result.legal_modes:
            assert result.violations[mode.value] == []


class TestDeploymentCap:
    def test_over_base_cap_rejects_conserve_balanced_push(self):
        gate = RegulatoryGate()
        t = make_telemetry(lap_energy_deployed_mj=9.5)
        result = gate.evaluate(t)
        assert DeploymentMode.CONSERVE_MODE not in result.legal_modes
        assert DeploymentMode.BALANCED_MODE not in result.legal_modes
        assert DeploymentMode.PUSH_MODE not in result.legal_modes

    def test_over_bonus_cap_rejects_use_bonus(self):
        gate = RegulatoryGate()
        t = make_telemetry(
            lap_energy_deployed_mj=9.6,
            gap_to_car_ahead_s=0.5,
            overtake_qualified_last_lap=True,
        )
        result = gate.evaluate(t)
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE not in result.legal_modes

    def test_within_cap_legal(self):
        gate = RegulatoryGate()
        t = make_telemetry(lap_energy_deployed_mj=2.0)
        result = gate.evaluate(t)
        assert DeploymentMode.BALANCED_MODE in result.legal_modes


class TestSoCSwing:
    def test_excessive_soc_swing_rejects_all(self):
        """SoC swing > 4 MJ (Art.5.4.9) should reject all modes."""
        gate = RegulatoryGate()
        t = make_telemetry(current_soc_mj=1.0, lap_start_soc_mj=6.0)  # delta = 5.0 MJ
        result = gate.evaluate(t)
        assert len(result.legal_modes) == 0

    def test_no_lap_start_soc_skips_swing_check(self):
        """When lap_start_soc_mj is None, SoC swing check is skipped (fails open)."""
        gate = RegulatoryGate()
        t = make_telemetry(lap_start_soc_mj=None)
        result = gate.evaluate(t)
        assert len(result.legal_modes) >= 3

    def test_within_soc_swing_legal(self):
        gate = RegulatoryGate()
        t = make_telemetry(current_soc_mj=4.5, lap_start_soc_mj=5.0)  # delta = 0.5
        result = gate.evaluate(t)
        assert DeploymentMode.BALANCED_MODE in result.legal_modes


class TestArmOvertakeProximity:
    def test_within_gap_threshold_allows_arm(self):
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=0.8)
        result = gate.evaluate(t)
        assert DeploymentMode.ARM_OVERTAKE_MODE in result.legal_modes

    def test_over_gap_threshold_rejects_arm(self):
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=1.5)
        result = gate.evaluate(t)
        assert DeploymentMode.ARM_OVERTAKE_MODE not in result.legal_modes

    def test_no_gap_rejects_arm(self):
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=None)
        result = gate.evaluate(t)
        assert DeploymentMode.ARM_OVERTAKE_MODE not in result.legal_modes

    def test_exact_threshold_is_legal(self):
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=1.0)
        result = gate.evaluate(t)
        assert DeploymentMode.ARM_OVERTAKE_MODE in result.legal_modes


class TestUseOvertakeBonusBanked:
    def test_banked_allows_use_bonus(self):
        gate = RegulatoryGate()
        t = make_telemetry(overtake_qualified_last_lap=True, gap_to_car_ahead_s=0.5)
        result = gate.evaluate(t)
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE in result.legal_modes

    def test_not_banked_rejects_use_bonus(self):
        gate = RegulatoryGate()
        t = make_telemetry(overtake_qualified_last_lap=False)
        result = gate.evaluate(t)
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE not in result.legal_modes

    def test_use_bonus_does_not_recheck_gap(self):
        """USE_OVERTAKE_BONUS_MODE only checks banked flag, not current gap."""
        gate = RegulatoryGate()
        t = make_telemetry(overtake_qualified_last_lap=True, gap_to_car_ahead_s=3.0)
        result = gate.evaluate(t)
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE in result.legal_modes

    def test_bonus_cap_larger_than_base_cap(self):
        gate = RegulatoryGate()
        result = gate.evaluate(make_telemetry())
        bonus_cap = result.base_cap_mj[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value]
        base_cap = result.base_cap_mj[DeploymentMode.BALANCED_MODE.value]
        assert bonus_cap == base_cap + 0.5


class TestQualificationForNextLap:
    def test_within_threshold_qualifies_next_lap(self):
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=0.7)
        result = gate.evaluate(t)
        assert result.qualifies_for_overtake_bonus_next_lap is True

    def test_over_threshold_does_not_qualify(self):
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=1.5)
        result = gate.evaluate(t)
        assert result.qualifies_for_overtake_bonus_next_lap is False

    def test_qualification_is_telemetry_fact_not_mode_choice(self):
        """Qualification depends only on gap, regardless of which mode was recommended."""
        gate = RegulatoryGate()
        t = make_telemetry(gap_to_car_ahead_s=0.5, overtake_qualified_last_lap=False)
        result = gate.evaluate(t)
        assert result.qualifies_for_overtake_bonus_next_lap is True
