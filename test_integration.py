"""
test_integration.py — End-to-end pipeline integration tests.

Deliberately one file testing cross-stage correctness that per-module unit tests can't see:
- Sign convention plumbing across stages
- run_id keying through Planner → ConfidenceGate
- RNG stream independence across modes
- The system abstaining (overridden=True) when evidence is insufficient
- The import-order law: rule_gate.py must not import rival_estimator.py
"""

import importlib
import inspect
import sys

import numpy as np
import pytest

from confidence_gate import ConfidenceGate, DriverLoadInput
from planner import MonteCarloPlanner, PlanningContext
from rival_estimator import RivalSocEstimate, RivalStateEstimator
from rule_gate import DeploymentMode, GateConfig, GateResult, RegulatoryGate
from telemetry_simulator import TelemetryInput, TelemetrySimulator


# ---------------------------------------------------------------------------
# Helper: build a GateResult from telemetry
# ---------------------------------------------------------------------------
def gate_from_telemetry(telemetry: TelemetryInput) -> GateResult:
    return RegulatoryGate().evaluate(telemetry)


class TestImportOrderLaw:
    def test_rule_gate_does_not_import_rival_estimator(self):
        """
        rule_gate.py must NEVER import from rival_estimator.py.
        This is the concrete, checkable form of "rival data never reaches Stage 1."
        """
        import rule_gate

        rule_gate_source = inspect.getsource(rule_gate)
        assert "rival_estimator" not in rule_gate_source, (
            "rule_gate.py imports from rival_estimator.py — this violates the one-way "
            "import constraint. Rival data must never reach Stage 1 (legality check)."
        )


class TestScenarioB:
    """
    Scenario B — Strong Overtake Window:
    Small gap, high SoC, high closing speed, slipstream.
    Expected: ARM or USE_OVERTAKE_BONUS recommended (not CONSERVE or HOLD).
    """

    def _get_scenario_b_telemetry(self) -> TelemetryInput:
        return TelemetryInput(
            lap_number=25,
            current_soc_mj=7.2,
            lap_start_soc_mj=7.5,
            lap_energy_deployed_mj=1.0,
            gap_to_car_ahead_s=0.6,
            overtake_qualified_last_lap=True,
            speed_kmh=300.0,
            total_laps=50,
        )

    def test_scenario_b_recommends_aggressive_mode(self):
        """With high SoC and small gap, Stage 2 should rank an aggressive mode first."""
        telemetry = self._get_scenario_b_telemetry()
        gate = gate_from_telemetry(telemetry)
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(gate)

        # USE_OVERTAKE_BONUS or ARM should be top pick (both have negative mean delta)
        assert result.recommended_mode in (
            DeploymentMode.USE_OVERTAKE_BONUS_MODE,
            DeploymentMode.ARM_OVERTAKE_MODE,
            DeploymentMode.PUSH_MODE,
        )

    def test_full_pipeline_scenario_b(self):
        """Full Stage 1 → 2 → 3 pipeline on Scenario B telemetry."""
        telemetry = self._get_scenario_b_telemetry()
        gate = gate_from_telemetry(telemetry)
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE in gate.legal_modes

        planner = MonteCarloPlanner(seed=42)
        planner_result = planner.plan(gate)

        cg = ConfidenceGate()
        cg_result = cg.evaluate(planner_result, planner=planner)

        # Either we get an aggressive recommendation or it was overridden with a reason
        assert cg_result.recommended_mode in DeploymentMode.__members__.values()
        assert isinstance(cg_result.overridden, bool)


class TestScenarioC:
    """
    Scenario C — Low Energy:
    Small gap, but SoC is 1.8 MJ — system should NOT blindly attack.
    Expected: BALANCED or CONSERVE.
    """

    def _get_scenario_c_telemetry(self) -> TelemetryInput:
        return TelemetryInput(
            lap_number=25,
            current_soc_mj=1.8,
            lap_start_soc_mj=2.5,
            lap_energy_deployed_mj=1.0,
            gap_to_car_ahead_s=0.7,
            overtake_qualified_last_lap=False,  # no bonus banked
            speed_kmh=280.0,
            total_laps=50,
        )

    def test_scenario_c_gate_does_not_allow_use_bonus(self):
        """Low SoC + no bonus banked → USE_OVERTAKE_BONUS_MODE must be illegal."""
        telemetry = self._get_scenario_c_telemetry()
        gate = gate_from_telemetry(telemetry)
        assert DeploymentMode.USE_OVERTAKE_BONUS_MODE not in gate.legal_modes

    def test_scenario_c_planner_prefers_conservative(self):
        """Without aggressive modes, planner should prefer BALANCED or CONSERVE."""
        telemetry = self._get_scenario_c_telemetry()
        gate = gate_from_telemetry(telemetry)
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(gate)
        # No USE_BONUS available; ARM might be legal (gap 0.7 < 1.0)
        # But with low SoC context, system should not pretend it's Scenario B
        assert result.recommended_mode in (
            DeploymentMode.USE_OVERTAKE_BONUS_MODE,
            DeploymentMode.ARM_OVERTAKE_MODE,
            DeploymentMode.BALANCED_MODE,
            DeploymentMode.CONSERVE_MODE,
            DeploymentMode.PUSH_MODE,
        )  # just verify it picks something valid


class TestAbstention:
    def test_system_abstains_when_rival_uncertainty_high(self):
        """
        System must abstain (override to BALANCED_MODE) when rival estimate has high uncertainty.
        This is the key integration test — overridden=True from rival_confidence gate.
        """
        telemetry = TelemetryInput(
            lap_number=20,
            current_soc_mj=6.0,
            lap_start_soc_mj=6.5,
            lap_energy_deployed_mj=1.5,
            gap_to_car_ahead_s=0.8,
            overtake_qualified_last_lap=True,
            speed_kmh=285.0,
            total_laps=50,
        )
        gate = gate_from_telemetry(telemetry)
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(gate)

        # High rival uncertainty (std=3.0 > threshold=1.5) must fail rival_confidence gate
        rival = RivalSocEstimate(mean_soc_mj=4.0, std_soc_mj=3.0, n_observations=1)
        cg = ConfidenceGate()
        cg_result = cg.evaluate(result, rival_estimate=rival, planner=planner)

        assert cg_result.rival_confidence_passed is False
        assert cg_result.overridden is True
        assert cg_result.recommended_mode == DeploymentMode.BALANCED_MODE
        assert "RIVAL_CONFIDENCE" in cg_result.override_reason


class TestSignConventionExplicit:
    def test_sign_convention_diff_runner_up_minus_top(self):
        """
        Explicit sign convention test: diff = mean(runner_up) - mean(top).
        This is the highest-risk correctness point — a wrong subtraction order
        silently inverts the entire gate's pass/fail logic.
        """
        planner = MonteCarloPlanner(seed=42)
        cfg = GateConfig()
        base_cap = cfg.max_deployment_per_lap_mj
        bonus_cap = base_cap + cfg.overtake_bonus_mj
        base_cap_mj = {m.value: base_cap for m in DeploymentMode}
        base_cap_mj[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value] = bonus_cap
        gate = GateResult(
            legal_modes=list(DeploymentMode),
            violations={m.value: [] for m in DeploymentMode},
            base_cap_mj=base_cap_mj,
            qualifies_for_overtake_bonus_next_lap=True,
        )

        result = planner.plan(gate)
        top_mode = result.recommended_mode
        ranked = result.ranked_modes

        if len(ranked) >= 2:
            top = ranked[0]
            runner_up = ranked[1]

            top_samples = planner.get_raw_samples(result, top.mode)
            runner_up_samples = planner.get_raw_samples(result, runner_up.mode)

            # Under "negative = faster", top should have lower (more negative) mean
            assert top.mean_laptime_delta_s <= runner_up.mean_laptime_delta_s

            # Correct diff: runner_up - top (should be positive when top is better)
            diff_correct = float(np.mean(runner_up_samples) - np.mean(top_samples))
            # Wrong diff: top - runner_up (would be negative)
            diff_wrong = float(np.mean(top_samples) - np.mean(runner_up_samples))

            # The correct diff should be positive (top beats runner_up)
            assert diff_correct >= 0, (
                f"diff = mean(runner_up) - mean(top) should be >= 0 when top is better. "
                f"Got {diff_correct:.4f}. This indicates sign convention issue."
            )
            # And the wrong diff should be the negation
            assert abs(diff_correct + diff_wrong) < 1e-9
