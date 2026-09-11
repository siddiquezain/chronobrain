"""
FastF1 abstraction tests — no `fastf1` install required.

`dataframe_to_samples()` is a pure function of a DataFrame with FastF1-shaped
columns, so we drive the whole normalize -> NormalizedLap -> decision-engine path
with a synthetic frame. This proves the synthetic simulator and a FastF1-style
replay feed the *same* downstream engine.
"""

import numpy as np
import pandas as pd
import pytest

from app.data.fastf1_service import dataframe_to_samples
from app.data.normalizer import ReplayEnergyModel, condense_lap, to_rival_observation, to_telemetry_input
from app.data.providers import ReplayProvider, build_provider
from app.decision import DecisionConfig
from app.decision.engine import run_decision


def _fake_lap_frame(peak_speed=320.0, n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 5.0, n)
    # speed ramps up then plateaus/dips — a plausible straight + braking zone
    speed = np.concatenate([
        np.linspace(120, peak_speed, n // 2),
        np.linspace(peak_speed, peak_speed - 60, n - n // 2),
    ])
    throttle = np.concatenate([np.full(n // 2, 100.0), np.linspace(100, 10, n - n // 2)])
    brake = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(bool)
    return pd.DataFrame({
        "Time": pd.to_timedelta(t, unit="s"),
        "Speed": speed + rng.normal(0, 1, n),
        "Throttle": throttle,
        "Brake": brake,
        "RPM": np.full(n, 10500.0),
        "nGear": np.clip((speed / 45).astype(int), 1, 8),
        "DRS": np.where(np.arange(n) > n // 3, 12, 0),
        "Distance": np.cumsum(speed) * 0.1,
    })


def test_dataframe_to_samples_maps_channels():
    df = _fake_lap_frame()
    samples = dataframe_to_samples(df, lap=12)
    assert len(samples) == len(df)
    assert all(s.lap == 12 for s in samples)
    assert 0.0 <= samples[0].throttle <= 1.0
    assert samples[-1].brake == 1.0          # bool -> 1.0
    assert samples[0].timestamp_s == 0.0     # rebased to lap start
    assert any(s.drs for s in samples)       # DRS code 12 -> open


def test_missing_columns_are_tolerated():
    df = pd.DataFrame({"Speed": [100.0, 200.0, 300.0]})  # only speed
    samples = dataframe_to_samples(df, lap=1)
    assert len(samples) == 3
    assert samples[0].throttle is None and samples[0].brake is None and samples[0].drs is None


def test_replay_energy_model_is_deterministic_and_modeled():
    df = _fake_lap_frame()
    s = dataframe_to_samples(df, lap=1)
    a = condense_lap(s, s, lap=1, total_laps=50, data_mode="REPLAY",
                     prev_soc_mj=5.0, gap_to_car_ahead_s=0.7,
                     energy_model=ReplayEnergyModel())
    b = condense_lap(s, s, lap=1, total_laps=50, data_mode="REPLAY",
                     prev_soc_mj=5.0, gap_to_car_ahead_s=0.7,
                     energy_model=ReplayEnergyModel())
    assert a.model_dump() == b.model_dump()
    assert a.energy_is_modeled is True          # F1 publishes no SoC
    assert a.our_soc_mj is not None
    assert a.has_rival_observation is True


def test_replay_provider_feeds_the_same_engine_as_synthetic():
    # build ~15 replayed laps from synthetic frames
    laps = []
    prev = None
    for ln in range(1, 16):
        our = dataframe_to_samples(_fake_lap_frame(peak_speed=318, seed=ln), lap=ln)
        rival = dataframe_to_samples(_fake_lap_frame(peak_speed=300, seed=100 + ln), lap=ln)
        nl = condense_lap(our, rival, lap=ln, total_laps=50, data_mode="REPLAY",
                          prev_soc_mj=prev, gap_to_car_ahead_s=0.8,
                          rival_sector_baseline_s=5.0)
        prev = nl.our_soc_mj
        laps.append(nl)

    replay_snapshot = run_decision(ReplayProvider(laps, "test replay"), lap=15,
                                   config=DecisionConfig(seed=42))
    synth_snapshot = run_decision(build_provider("synthetic", scenario="B", seed=42, total_laps=50),
                                  lap=15, config=DecisionConfig(seed=42))

    # both produce a well-formed snapshot with the same contract
    for snap in (replay_snapshot, synth_snapshot):
        assert snap.decision.mode in _MODES
        assert 0.0 <= snap.decision.confidence <= 1.0
        assert len(snap.trace) >= 6
    assert replay_snapshot.meta.data_mode == "REPLAY"
    assert replay_snapshot.energy.energy_is_modeled is True
    assert synth_snapshot.energy.energy_is_modeled is False


def test_adapters_round_trip_to_stack_a_contracts():
    df = _fake_lap_frame()
    s = dataframe_to_samples(df, lap=3)
    nl = condense_lap(s, s, lap=3, total_laps=50, data_mode="REPLAY",
                      prev_soc_mj=6.0, gap_to_car_ahead_s=0.6, rival_sector_baseline_s=5.0)
    ti = to_telemetry_input(nl, overtake_qualified_last_lap=True)
    ro = to_rival_observation(nl)
    assert ti.lap_number == 3 and ti.total_laps == 50
    assert 0.0 <= ti.current_soc_mj <= 9.0
    assert ro is not None and ro.terminal_speed_kmh > 0


_MODES = {
    "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
    "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
}


def test_condense_lap_rival_compound_round_trip():
    """rival_compound kwarg on condense_lap ends up on NormalizedLap."""
    from app.data.normalizer import condense_lap
    from app.data.samples import TelemetrySample

    sample = TelemetrySample(timestamp_s=0.0, lap=1, speed_kmh=280.0, throttle=0.8, brake=0.1)
    nl = condense_lap(
        [sample], [],
        lap=1, total_laps=50, data_mode="REPLAY",
        prev_soc_mj=None,
        gap_to_car_ahead_s=None,
        rival_compound="SOFT",
    )
    assert nl.rival_compound == "SOFT"


def test_condense_lap_rival_compound_defaults_none():
    """rival_compound defaults to None when not provided."""
    from app.data.normalizer import condense_lap
    from app.data.samples import TelemetrySample

    sample = TelemetrySample(timestamp_s=0.0, lap=1, speed_kmh=280.0, throttle=0.8, brake=0.1)
    nl = condense_lap(
        [sample], [],
        lap=1, total_laps=50, data_mode="REPLAY",
        prev_soc_mj=None,
        gap_to_car_ahead_s=None,
    )
    assert nl.rival_compound is None
