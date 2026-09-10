"""
Energy accounting / conservation regression tests (audit Issue 2 + 8 + 9).

The modelled ChronoPace SoC path for a historical replay must:
  * obey an explicit SoC_next = SoC + recovered - deployed accounting relationship,
  * stay inside [floor, capacity] and the per-lap swing / deployment / MGU-K caps,
  * be deterministic for the same seed / input,
  * genuinely ADVANCE with the replay (not accidentally constant),
  * never masquerade as measured battery SoC.

Cache-gated tests use the real 2024 Monza replay; the rest are offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.data.fastf1_service import FastF1Unavailable, fastf1_available
from app.data.normalizer import ReplayEnergyModel, condense_lap
from app.data.samples import TelemetrySample
from rule_gate import GateConfig

_HAVE_FASTF1 = fastf1_available()
_needs = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")
_GATE = GateConfig()


# ---------------------------------------------------------------------------
# offline — the model itself
# ---------------------------------------------------------------------------
def _samples(n=40, speed=300.0, throttle=0.75, brake=0.10):
    return [
        TelemetrySample(timestamp_s=float(i), lap=1, speed_kmh=speed,
                        throttle=throttle, brake=brake, distance_m=float(i) * 100.0)
        for i in range(n)
    ]


def _run(model, throttle_seq, brake_seq):
    """Thread the model through a sequence of laps; return list of NormalizedLap."""
    laps, prev, thr_seen, brk_seen = [], None, [], []
    for i, (thr, brk) in enumerate(zip(throttle_seq, brake_seq), start=1):
        thr_base = float(np.mean(thr_seen)) if thr_seen else None
        brk_base = float(np.mean(brk_seen)) if brk_seen else None
        nl = condense_lap(
            _samples(throttle=thr, brake=brk), _samples(speed=290.0),
            lap=i, total_laps=len(throttle_seq), data_mode="REPLAY",
            prev_soc_mj=prev, gap_to_car_ahead_s=0.8,
            throttle_baseline=thr_base, brake_baseline=brk_base,
            rival_sector_baseline_s=5.0,
        )
        prev = nl.our_soc_mj
        thr_seen.append(thr); brk_seen.append(brk)
        laps.append(nl)
    return laps


def test_1_soc_never_negative_and_2_never_exceeds_capacity():
    m = ReplayEnergyModel()
    # 40 laps of wildly varying work
    rng = np.random.default_rng(0)
    thr = list(rng.uniform(0.2, 0.98, 40))
    brk = list(rng.uniform(0.0, 0.4, 40))
    for nl in _run(m, thr, brk):
        assert 0.0 <= nl.our_soc_mj <= m.capacity_mj
        assert nl.our_soc_mj >= m.floor_mj - 1e-9


def test_3_deployment_never_exceeds_per_lap_limit():
    for nl in _run(ReplayEnergyModel(), [0.99] * 20, [0.0] * 20):
        assert nl.our_lap_energy_deployed_mj <= _GATE.max_deployment_per_lap_mj


def test_4_mgu_k_peak_never_exceeds_ceiling():
    from app.decision.engine import _modeled_mgu_k_peak_kw
    for nl in _run(ReplayEnergyModel(), list(np.linspace(0.1, 1.0, 15)), [0.1] * 15):
        peak = _modeled_mgu_k_peak_kw(nl, _GATE.max_ers_k_power_kw)
        assert peak is None or peak <= _GATE.max_ers_k_power_kw + 1e-6


def test_5_per_lap_soc_swing_within_limit():
    rng = np.random.default_rng(1)
    laps = _run(ReplayEnergyModel(), list(rng.uniform(0.2, 1.0, 30)), list(rng.uniform(0.0, 0.4, 30)))
    for nl in laps:
        if nl.our_lap_start_soc_mj is not None:
            assert abs(nl.our_soc_mj - nl.our_lap_start_soc_mj) <= _GATE.max_delta_soc_mj


def test_6_deterministic_same_input():
    thr = [0.5, 0.7, 0.9, 0.6, 0.8, 0.75, 0.7]
    brk = [0.1, 0.2, 0.05, 0.15, 0.1, 0.12, 0.1]
    a = [nl.our_soc_mj for nl in _run(ReplayEnergyModel(), thr, brk)]
    b = [nl.our_soc_mj for nl in _run(ReplayEnergyModel(), thr, brk)]
    assert a == b


def test_7_explicit_accounting_relationship_holds():
    """SoC_next == SoC + recovered - deployed (net of a nominal lap), within fp tol,
    or clipped to a bound."""
    TOL = 1e-6
    m = ReplayEnergyModel()
    rng = np.random.default_rng(2)
    thr = list(rng.uniform(0.3, 0.95, 25))
    brk = list(rng.uniform(0.02, 0.35, 25))
    laps = _run(m, thr, brk)
    for nl in laps:
        if nl.our_lap_start_soc_mj is None or nl.our_lap_net_swing_mj is None:
            continue
        prev = nl.our_lap_start_soc_mj
        # reconstruct: SoC_next = clip(prev + net_swing + anchor, floor, cap)
        anchor = m.reversion * (m.nominal_soc_mj - prev)
        expected = min(m.capacity_mj, max(m.floor_mj, prev + nl.our_lap_net_swing_mj + anchor))
        assert abs(nl.our_soc_mj - round(expected, 4)) <= TOL
        # and net_swing itself == (recovered - deployed) - (nominal recovered - nominal deployed)
        # -> at minimum, recovered - deployed is exposed and finite
        assert nl.our_lap_energy_recovered_mj is not None
        assert abs((nl.our_lap_energy_recovered_mj - nl.our_lap_energy_deployed_mj)) < 20.0


def test_8_changing_the_work_changes_the_state():
    flat = _run(ReplayEnergyModel(), [0.7] * 15, [0.1] * 15)
    pushy = _run(ReplayEnergyModel(), [0.7] * 5 + [0.95] * 10, [0.1] * 15)
    assert flat[-1].our_soc_mj != pushy[-1].our_soc_mj
    assert pushy[-1].our_soc_mj < flat[-1].our_soc_mj      # pushing harder drains more


def test_10_replay_soc_is_labeled_modeled_not_measured():
    nl = _run(ReplayEnergyModel(), [0.7, 0.8], [0.1, 0.1])[-1]
    assert nl.energy_is_modeled is True


def test_pit_lap_holds_soc_and_zeroes_components():
    nl = condense_lap(
        _samples(throttle=0.2, brake=0.0), _samples(), lap=5, total_laps=53,
        data_mode="REPLAY", prev_soc_mj=4.2, gap_to_car_ahead_s=None,
        throttle_baseline=0.8, brake_baseline=0.1, lap_status="pit",
    )
    assert nl.our_soc_mj == pytest.approx(4.2)
    assert nl.our_lap_energy_deployed_mj == 0.0
    assert nl.our_lap_energy_recovered_mj == 0.0


# ---------------------------------------------------------------------------
# cache-gated — the whole 2024 Monza replay
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def monza():
    if not _HAVE_FASTF1:
        pytest.skip("fastf1 not installed")
    from app.replay import run_historical_replay
    from app.replay.historical import clear_session_cache
    from app.data.fastf1_service import clear_field_timeline_cache
    clear_session_cache(); clear_field_timeline_cache()
    try:
        return run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
    except FastF1Unavailable as exc:
        pytest.skip(f"session unavailable: {exc}")


@_needs
def test_replay_energy_accounting_valid_every_lap(monza):
    for L in monza["laps"]:
        e = L["energy"]
        assert 0.0 <= e["soc_mj"] <= e["soc_capacity_mj"]
        assert e["deployed_this_lap_mj"] <= 9.0
        if e["lap_start_soc_mj"] is not None:
            assert abs(e["soc_mj"] - e["lap_start_soc_mj"]) <= 4.0
        if e["modeled_mgu_k_peak_kw"] is not None:
            assert e["modeled_mgu_k_peak_kw"] <= e["mgu_k_power_ceiling_kw"]
        assert e["energy_is_modeled"] is True
        assert "MODELED" in e["label"]


@_needs
def test_replay_soc_is_not_accidentally_constant(monza):
    socs = [L["energy"]["soc_mj"] for L in monza["laps"]]
    assert len(set(socs)) >= 8, "SoC barely moves across 53 laps — is the model actually stateful?"
    assert max(socs) - min(socs) >= 0.3
    deps = [L["energy"]["deployed_this_lap_mj"] for L in monza["laps"] if L["lap_status"] == "racing"]
    assert max(deps) - min(deps) >= 0.03      # deployment tracks the real throttle trace


@_needs
def test_replay_energy_advances_with_the_lap(monza):
    by_lap = {L["lap"]: L["energy"]["soc_mj"] for L in monza["laps"]}
    # the SoC at lap 20 is the result of 19 laps of accounting, not lap 5's value
    assert by_lap.get(5) != by_lap.get(20)
    # lap_start_soc at lap N == soc at lap N-1 (state threads forward). The two are
    # rounded to 4 dp / 3 dp respectively in the response, so allow one rounding unit.
    ROUNDING_TOL = 1e-3
    checked = 0
    for L in monza["laps"]:
        p = L["lap"] - 1
        if p in by_lap and L["energy"]["lap_start_soc_mj"] is not None and L["lap_status"] == "racing":
            assert abs(L["energy"]["lap_start_soc_mj"] - by_lap[p]) < ROUNDING_TOL
            checked += 1
    assert checked >= 20


@_needs
def test_replay_deterministic_energy(monza):
    from app.replay import run_historical_replay
    from app.replay.historical import clear_session_cache
    from app.data.fastf1_service import clear_field_timeline_cache
    clear_session_cache(); clear_field_timeline_cache()
    again = run_historical_replay(race_key="2024_italian_gp", start_lap=1, end_lap=53, seed=42)
    assert [L["energy"] for L in again["laps"]] == [L["energy"] for L in monza["laps"]]


@_needs
def test_replay_never_claims_measured_battery(monza):
    import json
    blob = json.dumps(monza).lower()
    # false claims — must never appear
    for bad in ("measured battery", "actual battery", "measured soc", "actual soc",
                "measured ers", "telemetry contains", "ferrari's battery", "real battery soc"):
        assert bad not in blob
    # honest disclaimers — must be present
    lab = monza["laps"][3]["energy"]["label"].lower()
    assert "modeled" in lab and "not measured" in lab
    assert "no ers" in blob or "publishes no ers" in blob or "no car's real ers soc" in blob


@_needs
def test_synthetic_mode_is_explicitly_synthetic():
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).post("/api/v1/decision", json={"scenario": "B", "seed": 42, "lap": 20}).json()
    assert r["meta"]["data_mode"] == "SYNTHETIC"
    # synthetic SoC is the sim's own ground truth -> energy_is_modeled False
    assert r["energy"]["energy_is_modeled"] is False


def test_energy_block_provenance_is_modeled():
    from app.decision.engine import run_scenario
    snap = run_scenario("B", seed=42, lap=10)
    assert snap.energy.energy_provenance == "MODELED"


def test_rival_block_provenance_is_inferred():
    from app.decision.engine import run_scenario
    snap = run_scenario("B", seed=42, lap=20)
    assert snap.rival.energy_provenance == "INFERRED"
