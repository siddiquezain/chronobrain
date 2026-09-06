"""
Behavioural invariants of the decision engine (the properties the spec calls out):

- changing our energy can change the decision
- a stronger future opportunity can change the decision
- an illegal strategy can never be the final decision
- low confidence forces BALANCED_MODE / HOLD
- missing optional telemetry does not crash the pipeline
"""

import dataclasses

import pytest

from app.data.providers import build_provider
from app.data.samples import NormalizedLap
from app.decision import DecisionConfig, run_scenario
from app.decision.engine import _horizon_prefers_wait, run_decision
from opportunity_engine import HorizonResult, StrategyOutcome
from rule_gate import DeploymentMode


# --- energy changes the decision ------------------------------------------------
def test_high_energy_attacks_low_energy_holds():
    """Same opportunity (scenario B window), different SoC -> different decision."""
    attack = run_scenario("B", seed=42, total_laps=50, lap=20)   # high SoC
    hold = run_scenario("C", seed=42, total_laps=50, lap=20)     # low SoC, same small gap
    assert attack.decision.action == "ATTACK_NOW"
    # low energy: never attack now — either hold, or defer to a window we can afford
    assert hold.decision.mode in ("CONSERVE_MODE", "BALANCED_MODE")
    assert hold.decision.action in ("HOLD", "CONSERVE") or hold.decision.action.startswith("WAIT_")


def test_energy_reserve_low_blocks_aggression_even_when_planner_wants_it():
    s = run_scenario("C", seed=42, total_laps=50, lap=20)
    # planner's raw pick is aggressive; the decision engine refuses it on energy grounds
    assert s.monte_carlo.recommended_mode in (
        "USE_OVERTAKE_BONUS_MODE", "ARM_OVERTAKE_MODE", "PUSH_MODE"
    )
    assert s.decision.mode not in ("USE_OVERTAKE_BONUS_MODE", "ARM_OVERTAKE_MODE", "PUSH_MODE")
    assert "ENERGY_RESERVE_LOW" in s.reason_codes


# --- future opportunity changes the decision ----------------------------------
def test_stronger_future_window_defers_the_attack(monkeypatch):
    """
    If the opportunity horizon says a WAIT_N strategy beats attacking now by the
    decisive margin, an otherwise-aggressive pick is deferred to a WAIT action.
    """
    import app.decision.engine as eng

    real = eng.OpportunityEngine

    class FutureIsBetter(real):
        def evaluate(self, telemetry, gate_result, rival_estimate=None, **kw):
            base = super().evaluate(telemetry, gate_result, rival_estimate, **kw)
            ranked = [
                StrategyOutcome(strategy_name="WAIT_2", delay_laps=2,
                                mean_horizon_delta_s=-5.0, std_horizon_delta_s=1.0,
                                confidence_ci_lower_s=-5.2),
                StrategyOutcome(strategy_name="ATTACK_NOW", delay_laps=0,
                                mean_horizon_delta_s=-1.0, std_horizon_delta_s=1.0,
                                confidence_ci_lower_s=-1.2),
                StrategyOutcome(strategy_name="HOLD", delay_laps=-1,
                                mean_horizon_delta_s=0.0, std_horizon_delta_s=1.0,
                                confidence_ci_lower_s=-0.2),
            ]
            return HorizonResult(
                recommended_strategy="WAIT_2", ranked_strategies=ranked,
                foregone_strategy="ATTACK_NOW", foregone_value_gap_s=4.0,
                uncertainty_note=base.uncertainty_note,
            )

    monkeypatch.setattr(eng, "OpportunityEngine", FutureIsBetter)
    s = run_scenario("B", seed=42, total_laps=50, lap=20)
    assert s.opportunity.prefers_wait is True
    assert s.decision.action.startswith("WAIT_")
    assert s.decision.mode in ("BALANCED_MODE", "CONSERVE_MODE")
    assert "FUTURE_WINDOW_STRONGER" in s.reason_codes


def test_horizon_margin_below_threshold_does_not_defer():
    p = build_provider("synthetic", scenario="B", seed=42, total_laps=50)
    s = run_decision(p, lap=20, config=DecisionConfig(seed=42, horizon_decisive_margin_s=99.0))
    assert s.opportunity.prefers_wait is False


# --- illegal strategies can never be final -----------------------------------
@pytest.mark.parametrize("scenario", ["A", "B", "C", "D", "E"])
def test_final_mode_is_always_legal_or_balanced_fallback(scenario):
    for lap in (5, 15, 30, 45):
        s = run_scenario(scenario, seed=42, total_laps=50, lap=lap)
        legal = set(s.compliance.legal_modes)
        assert (
            s.decision.mode in legal
            or (not legal and s.decision.mode == "BALANCED_MODE")
        ), f"{scenario} lap {lap}: illegal final mode {s.decision.mode}"


def test_scenario_e_is_rejected_by_regulatory_gate():
    s = run_scenario("E", seed=42, total_laps=50, lap=30)
    assert s.compliance.legal is False
    assert s.compliance.legal_modes == []
    assert s.decision.mode == "BALANCED_MODE"
    assert s.decision.action == "HOLD"
    assert "REGULATORY_CONSTRAINT" in s.reason_codes


# --- low confidence forces BALANCED -----------------------------------------
def test_low_confidence_overrides_to_balanced(monkeypatch):
    import app.decision.engine as eng
    from confidence_gate import ConfidenceGate

    real_eval = ConfidenceGate.evaluate

    def forced_override(self, planner_result, **kw):
        res = real_eval(self, planner_result, **kw)
        return res.model_copy(update={
            "recommended_mode": DeploymentMode.BALANCED_MODE,
            "overridden": True,
            "override_reason": "RIVAL_CONFIDENCE",
            "rival_confidence_passed": False,
        })

    monkeypatch.setattr(ConfidenceGate, "evaluate", forced_override)
    s = run_scenario("B", seed=42, total_laps=50, lap=20)
    assert s.decision.confidence_overridden is True
    assert s.decision.mode == "BALANCED_MODE"
    assert s.decision.action == "HOLD"


# --- missing optional telemetry ---------------------------------------------
def test_missing_optional_fields_do_not_crash():
    """A NormalizedLap with only the required fields still produces a snapshot."""
    laps = [
        NormalizedLap(
            lap=i, total_laps=5, data_mode="REPLAY",
            our_speed_kmh=280.0, our_soc_mj=5.0, our_lap_start_soc_mj=5.0,
            our_lap_energy_deployed_mj=1.5,
            gap_to_car_ahead_s=None, position=None, sector=None, drs_available=None,
            energy_is_modeled=True,
        )
        for i in range(1, 6)
    ]
    from app.data.providers import ReplayProvider

    s = run_decision(ReplayProvider(laps), lap=5, config=DecisionConfig(seed=42))
    assert s.decision.mode in {m for m in _MODES}
    assert s.rival.n_observations == 0  # no rival observations available
    assert isinstance(s.narrative, type(None))


_MODES = {
    "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
    "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
}
