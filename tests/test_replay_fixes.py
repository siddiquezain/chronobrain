"""
Regression tests for the three targeted replay fixes:

  1. ahead/behind direction — the relative gap is placed on the correct side using
     real running position; an unsigned gap is never treated as "car ahead".
  2. pit-stop awareness — pit / out / invalid laps are real telemetry but not
     clean racing evidence: they don't collapse the rival filter, don't create an
     overtake window, and don't pass Data Quality as perfect.
  3. Monte Carlo causal relevance — the same planner produces different simulated
     outcomes as live race state changes, and a static prior can't let an
     infeasible-in-context mode dominate the final decision.
"""

from __future__ import annotations

import pytest

from app.data.fastf1_service import FastF1Unavailable, _we_lead, fastf1_available
from app.data.normalizer import condense_lap
from app.data.providers import ReplayProvider
from app.data.quality import assess_quality
from app.data.samples import NormalizedLap, TelemetrySample
from app.decision import DecisionConfig, run_decision
from app.decision.engine import run_pipeline
from app.replay import historical as H
from app.replay import resolve_race, run_historical_replay
from planner import DeploymentMode, MonteCarloPlanner, PlanningContext
from rival_estimator import RivalSocEstimate
from rule_gate import GateConfig, GateResult

_HAVE_FASTF1 = fastf1_available()
_needs_fastf1 = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")


# ===========================================================================
# FIX 1 — ahead / behind direction
# ===========================================================================
class TestAheadBehindDirection:
    def test_we_lead_from_position_lower_is_ahead(self):
        # LEC P1 / PIA P2 -> we lead (PIA is behind)
        assert _we_lead(1, 2, None, None) is True
        # LEC P2 / PIA P1 -> we do not lead (PIA is ahead)
        assert _we_lead(2, 1, None, None) is False

    def test_we_lead_falls_back_to_cumulative_time(self):
        # positions missing -> less elapsed race time = ahead
        assert _we_lead(None, None, 5000.0, 5003.0) is True
        assert _we_lead(None, None, 5003.0, 5000.0) is False
        assert _we_lead(None, None, None, None) is False

    def _lap(self, *, ahead=None, behind=None, soc=4.5):
        return NormalizedLap(
            lap=1, total_laps=53, data_mode="REPLAY",
            our_speed_kmh=330.0, our_soc_mj=soc, our_lap_start_soc_mj=soc,
            our_lap_energy_deployed_mj=1.2,
            gap_to_car_ahead_s=ahead, gap_to_car_behind_s=behind,
            rival_terminal_speed_kmh=320.0, rival_clipping_point_fraction=0.4,
            rival_corner_exit_accel_g=1.6, rival_sector_delta_s=-0.1,
            energy_is_modeled=True, raw_sample_count=300,
        )

    def test_gap_lands_in_ahead_slot_when_trailing(self):
        nl = self._lap(ahead=0.4)
        assert nl.gap_to_car_ahead_s == 0.4 and nl.gap_to_car_behind_s is None

    def test_gap_lands_in_behind_slot_when_leading(self):
        nl = self._lap(behind=0.4)
        assert nl.gap_to_car_behind_s == 0.4 and nl.gap_to_car_ahead_s is None

    def test_push_mode_becomes_the_decision_when_a_car_is_within_defending_range(self):
        """A real rival 0.5 s behind -> PUSH_MODE is a live, selectable outcome
        (previously impossible: gap_to_car_behind_s was always None in replay)."""
        laps = []
        soc = 6.5
        for i in range(1, 9):
            laps.append(NormalizedLap(
                lap=i, total_laps=53, data_mode="REPLAY",
                our_speed_kmh=330.0, our_soc_mj=soc, our_lap_start_soc_mj=soc,
                our_lap_energy_deployed_mj=1.2,
                gap_to_car_ahead_s=None, gap_to_car_behind_s=0.5,
                position=1,
                rival_terminal_speed_kmh=322.0, rival_clipping_point_fraction=0.42,
                rival_corner_exit_accel_g=1.7, rival_sector_delta_s=-0.2,
                energy_is_modeled=True, raw_sample_count=300,
            ))
        snap = run_decision(ReplayProvider(laps), lap=8, config=DecisionConfig(seed=42))
        assert snap.decision.mode == "PUSH_MODE"
        assert snap.decision.action == "PUSH"

    @_needs_fastf1
    def test_real_monza_direction_flips_when_leclerc_takes_the_lead(self, _monza):
        by_lap = {nl.lap: nl for nl in _monza}
        # early: LEC P2, PIA P1 -> PIA is the car AHEAD
        early = by_lap[2]
        assert early.gap_to_car_ahead_s is not None
        assert early.gap_to_car_behind_s is None
        # late: LEC P1, PIA P2/P3 -> PIA is the car BEHIND
        late = by_lap[45]
        assert late.gap_to_car_behind_s is not None
        assert late.gap_to_car_ahead_s is None

    @_needs_fastf1
    def test_late_race_is_framed_as_defending_not_attacking(self):
        res = run_historical_replay(race_key="2024_italian_gp", start_lap=40, end_lap=53, seed=42)
        roles = {L["rival_role"] for L in res["laps"]}
        assert roles == {"defending"}
        assert all(L["decision"] in ("HOLD", "PUSH", "CONSERVE") for L in res["laps"])
        assert all("ATTACK" not in L["reason_codes"] for L in res["laps"])


# ===========================================================================
# FIX 2 — pit-stop awareness
# ===========================================================================
def _samples(n=60, speed=300.0, throttle=0.9, brake=0.05):
    return [
        TelemetrySample(
            timestamp_s=float(i), lap=1, speed_kmh=speed,
            throttle=throttle, brake=brake, distance_m=float(i) * 90.0,
        )
        for i in range(n)
    ]


class TestPitStopAwareness:
    def test_pit_lap_status_propagates_to_normalized_lap(self):
        nl = condense_lap(
            _samples(), _samples(speed=310.0), lap=5, total_laps=53, data_mode="REPLAY",
            prev_soc_mj=4.5, gap_to_car_ahead_s=None,
            lap_status="pit", rival_lap_status="out_lap",
        )
        assert nl.lap_status == "pit"
        assert nl.rival_lap_status == "out_lap"

    def test_pit_lap_carries_soc_forward_unchanged(self):
        nl = condense_lap(
            _samples(throttle=0.1, brake=0.0), _samples(), lap=5, total_laps=53,
            data_mode="REPLAY", prev_soc_mj=4.2, gap_to_car_ahead_s=None,
            throttle_baseline=0.9, brake_baseline=0.05, lap_status="pit",
        )
        assert nl.our_soc_mj == pytest.approx(4.2)

    def test_non_racing_rival_lap_drops_the_observation(self):
        nl = condense_lap(
            _samples(), _samples(), lap=5, total_laps=53, data_mode="REPLAY",
            prev_soc_mj=4.5, gap_to_car_ahead_s=0.5, rival_lap_status="pit",
        )
        assert nl.rival_terminal_speed_kmh is None
        assert nl.rival_sector_delta_s is None
        assert nl.has_rival_observation is False

    def test_huge_sector_delta_is_reclassified_even_without_a_pit_flag(self):
        """A +18 s lap time vs baseline is a pit lap / SC lap — never 'battery empty'."""
        slow = [
            TelemetrySample(timestamp_s=float(i), lap=1, speed_kmh=120.0, throttle=0.2, brake=0.1)
            for i in range(60)
        ]
        # baseline ~ 60 s implied; this "lap" runs ~ 90 s -> +30 s delta
        long_samples = [
            TelemetrySample(timestamp_s=float(i) * 1.5, lap=1, speed_kmh=120.0, throttle=0.2, brake=0.1)
            for i in range(60)
        ]
        nl = condense_lap(
            _samples(), long_samples, lap=10, total_laps=53, data_mode="REPLAY",
            prev_soc_mj=4.5, gap_to_car_ahead_s=0.5,
            rival_sector_baseline_s=59.0, rival_lap_status="racing",
        )
        assert nl.rival_lap_status == "invalid_for_energy_inference"
        assert nl.rival_terminal_speed_kmh is None

    def test_data_quality_degrades_on_a_pit_target_lap(self):
        racing = NormalizedLap(
            lap=1, total_laps=53, data_mode="REPLAY", our_speed_kmh=330.0,
            our_soc_mj=4.5, our_lap_start_soc_mj=4.5, our_lap_energy_deployed_mj=1.0,
            rival_terminal_speed_kmh=320.0, rival_clipping_point_fraction=0.4,
            rival_corner_exit_accel_g=1.6, rival_sector_delta_s=-0.1,
            raw_sample_count=300,
        )
        pit = racing.model_copy(update={"lap": 2, "lap_status": "pit"})
        good = assess_quality([racing], racing, "REPLAY")
        degraded = assess_quality([racing, pit], pit, "REPLAY")
        assert good.status == "GOOD"
        assert degraded.status == "DEGRADED"
        assert degraded.quality_score < good.quality_score

    @_needs_fastf1
    def test_real_monza_no_lap_collapses_the_rival_estimator(self):
        """No lap of the whole race (pit laps and rival-switch laps included) may
        collapse the posterior to a confident LOW."""
        res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
        for L in res["laps"]:
            rv = L["rival_energy_inference"]
            if rv["n_observations"] == 0:
                continue                       # no estimate yet (lap 1, standing start)
            assert rv["std_mj"] >= 0.35
            assert not (rv["bucket"] == "LOW" and rv["distribution"]["low"] >= 0.9), (
                f"lap {L['lap']}: rival estimate collapsed to {rv}"
            )

    @_needs_fastf1
    def test_real_monza_our_pit_laps_degrade_quality_and_never_attack(self):
        res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
        by_lap = {L["lap"]: L for L in res["laps"]}
        # LEC's own stop: lap 15 (in) / lap 16 (out) are not clean racing evidence
        for lp in (15, 16):
            L = by_lap[lp]
            assert L["lap_status"] in ("pit", "out_lap")
            assert L["data_quality"]["status"] != "GOOD"
            assert L["decision"] != "ATTACK"

    @_needs_fastf1
    def test_real_monza_selector_never_gives_a_pitting_car_a_directional_role(self):
        res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
        for L in res["laps"]:
            sr = L["strategic_rival"]
            if sr["role"] in ("ATTACK_TARGET", "DEFENDING_THREAT"):
                assert L["rival_lap_status"] == "racing", (
                    f"lap {L['lap']}: {sr['role']} assigned to a {L['rival_lap_status']} car"
                )


# ===========================================================================
# FIX 3 — Monte Carlo causal relevance
# ===========================================================================
def _all_legal_gate() -> GateResult:
    cfg = GateConfig()
    legal = list(DeploymentMode)
    return GateResult(
        legal_modes=legal,
        violations={m.value: [] for m in DeploymentMode},
        base_cap_mj={m.value: cfg.max_deployment_per_lap_mj for m in DeploymentMode},
        qualifies_for_overtake_bonus_next_lap=True,
    )


def _mean_for(planner: MonteCarloPlanner, ctx: PlanningContext, mode: DeploymentMode) -> float:
    result = planner.plan(_all_legal_gate(), ctx)
    return next(p for p in result.ranked_modes if p.mode == mode).mean_laptime_delta_s


class TestMonteCarloResponds:
    def test_gap_to_car_ahead_changes_an_attack_outcome(self):
        near = _mean_for(MonteCarloPlanner(seed=42),
                         PlanningContext(gap_to_car_ahead_s=0.3),
                         DeploymentMode.USE_OVERTAKE_BONUS_MODE)
        far = _mean_for(MonteCarloPlanner(seed=42),
                        PlanningContext(gap_to_car_ahead_s=5.0),
                        DeploymentMode.USE_OVERTAKE_BONUS_MODE)
        assert near != far
        assert near < far    # a car in range makes attacking genuinely better

    def test_own_soc_changes_a_deployment_mode_outcome(self):
        healthy = _mean_for(MonteCarloPlanner(seed=42),
                            PlanningContext(gap_to_car_ahead_s=0.5, own_soc_mj=6.0, low_reserve_mj=2.0),
                            DeploymentMode.USE_OVERTAKE_BONUS_MODE)
        depleted = _mean_for(MonteCarloPlanner(seed=42),
                             PlanningContext(gap_to_car_ahead_s=0.5, own_soc_mj=0.5, low_reserve_mj=2.0),
                             DeploymentMode.USE_OVERTAKE_BONUS_MODE)
        assert depleted > healthy    # low SoC erodes the attack's pace

    def test_opportunity_strength_changes_an_attack_outcome(self):
        weak = _mean_for(MonteCarloPlanner(seed=42),
                         PlanningContext(gap_to_car_ahead_s=0.5, opportunity_strength=0.05),
                         DeploymentMode.ARM_OVERTAKE_MODE)
        strong = _mean_for(MonteCarloPlanner(seed=42),
                           PlanningContext(gap_to_car_ahead_s=0.5, opportunity_strength=0.95),
                           DeploymentMode.ARM_OVERTAKE_MODE)
        assert weak != strong

    def test_rival_energy_changes_a_push_outcome(self):
        low = _mean_for(MonteCarloPlanner(seed=42),
                        PlanningContext(gap_to_car_ahead_s=0.5,
                                        rival_soc_estimate=RivalSocEstimate(mean_soc_mj=1.0, std_soc_mj=0.3, n_observations=8)),
                        DeploymentMode.PUSH_MODE)
        high = _mean_for(MonteCarloPlanner(seed=42),
                         PlanningContext(gap_to_car_ahead_s=0.5,
                                         rival_soc_estimate=RivalSocEstimate(mean_soc_mj=8.0, std_soc_mj=0.3, n_observations=8)),
                         DeploymentMode.PUSH_MODE)
        assert low != high

    def test_defending_makes_push_valuable_but_only_when_a_car_is_behind(self):
        defending = _mean_for(MonteCarloPlanner(seed=42),
                              PlanningContext(gap_to_car_behind_s=0.4),
                              DeploymentMode.PUSH_MODE)
        clear = _mean_for(MonteCarloPlanner(seed=42),
                          PlanningContext(gap_to_car_behind_s=None),
                          DeploymentMode.PUSH_MODE)
        assert defending < clear

    def test_infeasible_context_stops_a_static_prior_from_dominating(self):
        """No car anywhere near: an attack/PUSH mode must NOT rank first purely on
        its (better) static prior."""
        result = MonteCarloPlanner(seed=42).plan(
            _all_legal_gate(),
            PlanningContext(gap_to_car_ahead_s=None, gap_to_car_behind_s=None),
        )
        assert result.recommended_mode in (
            DeploymentMode.BALANCED_MODE, DeploymentMode.CONSERVE_MODE
        )

    def test_null_context_is_still_deterministic(self):
        a = MonteCarloPlanner(seed=42).plan(_all_legal_gate(), PlanningContext())
        b = MonteCarloPlanner(seed=42).plan(_all_legal_gate(), PlanningContext())
        assert [p.mean_laptime_delta_s for p in a.ranked_modes] == \
               [p.mean_laptime_delta_s for p in b.ranked_modes]


# ===========================================================================
# Task 5 — posterior health + rival_reference_soc_mj in lap summaries
# ===========================================================================
class TestPosteriorHealthInLapSummary:
    """lap_summary() must include posterior health fields and rival_reference_soc_mj."""

    def _snap(self):
        laps = []
        soc = 6.5
        for i in range(1, 9):
            laps.append(NormalizedLap(
                lap=i, total_laps=53, data_mode="REPLAY",
                our_speed_kmh=330.0, our_soc_mj=soc, our_lap_start_soc_mj=soc,
                our_lap_energy_deployed_mj=1.2,
                gap_to_car_ahead_s=0.5,
                rival_terminal_speed_kmh=320.0, rival_clipping_point_fraction=0.4,
                rival_corner_exit_accel_g=1.6, rival_sector_delta_s=-0.1,
                energy_is_modeled=True, raw_sample_count=300,
            ))
        return run_decision(ReplayProvider(laps), lap=8, config=DecisionConfig(seed=42))

    def test_rival_energy_inference_has_posterior_health_fields(self):
        snap = self._snap()
        summary = H.lap_summary(snap)
        rei = summary["rival_energy_inference"]
        assert "effective_sample_size" in rei
        assert "evidence_quality" in rei
        assert "posterior_health" in rei
        assert "baseline_ready" in rei
        assert rei["energy_provenance"] == "INFERRED"
        assert "posterior_mean_mj" in rei
        assert "posterior_std_mj" in rei
        assert rei["posterior_mean_mj"] == rei["estimated_reserve_mj"]

    def test_rival_reference_soc_mj_present(self):
        snap = self._snap()
        summary = H.lap_summary(snap)
        assert "rival_reference_soc_mj" in summary
        assert summary["rival_reference_soc_mj"] is None   # not co-loaded in ego replay
        assert summary["rival_reference_provenance"] == "CHRONOPACE_MODELED"

    @_needs_fastf1
    def test_real_monza_replay_has_posterior_health_fields(self):
        res = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=10, seed=42)
        for L in res["laps"]:
            rei = L["rival_energy_inference"]
            assert "effective_sample_size" in rei, f"missing ESS on lap {L['lap']}"
            assert "evidence_quality" in rei
            assert "posterior_health" in rei
            assert rei["energy_provenance"] == "INFERRED"
            assert "rival_reference_soc_mj" in L
            assert L["rival_reference_provenance"] == "CHRONOPACE_MODELED"


# ===========================================================================
# fixture
# ===========================================================================
@pytest.fixture(scope="module")
def _monza():
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    from app.data.fastf1_service import load_replay
    race = resolve_race("2024_italian_gp")
    try:
        laps = load_replay(year=race.year, event=race.event, session=race.session,
                           our_driver="LEC", rival_driver="PIA", scheduled_laps=53)
    except FastF1Unavailable as exc:
        pytest.skip(f"session unavailable: {exc}")
    H._SESSION_CACHE[(race.key, "LEC", "PIA")] = laps
    return laps
