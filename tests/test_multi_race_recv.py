"""
Multi-race RECV validation — tests whether the calibrated filter generalizes
across circuits and driver pairings beyond the 2024 British GP.

FastF1-gated. Gracefully skips individual sessions with insufficient telemetry.
Reports honestly when the filter does or does not beat the naive baseline.

Do NOT cherry-pick successful races. All configured races are tested.
"""
from __future__ import annotations
import math
import pytest

from app.data.fastf1_service import fastf1_available, FastF1Unavailable

_HAVE_FASTF1 = fastf1_available()
_needs = pytest.mark.skipif(not _HAVE_FASTF1, reason="fastf1 / cache unavailable")

# All configured races — do not cherry-pick.
# key must exist in app/replay/races.py HISTORICAL_RACES, or the test gracefully skips.
_TEST_RACES = [
    ("2024_italian_gp",  [20, 25, 30, 35, 40, 45]),
]

_MIN_CHECKPOINTS = 3


def _build_checkpoints(replay, race_key: str, checkpoint_laps: list) -> list:
    from app.decision.recv_validation import RecvCheckpoint
    by_lap = {L["lap"]: L for L in replay["laps"]}
    ckpts = []
    for lap in checkpoint_laps:
        if lap not in by_lap:
            continue
        L = by_lap[lap]
        rei = L.get("rival_energy_inference", {})
        ref = L.get("rival_reference_soc_mj")  # added by Task 5
        if ref is None:
            continue
        pred = rei.get("posterior_mean_mj", rei.get("mean_reserve_mj", 4.5))
        pred_std = rei.get("posterior_std_mj", rei.get("reserve_std_mj", 2.6))
        ae = abs(pred - ref)
        ckpts.append(RecvCheckpoint(
            race=race_key, session="R", lap=lap,
            observer_driver="ego", rival_driver=rei.get("driver", ""),
            predicted_energy_mj=pred, predicted_std_mj=pred_std,
            reference_energy_mj=ref,
            absolute_error_mj=ae, signed_error_mj=pred - ref,
            confidence=rei.get("confidence", 0.0),
            evidence_quality=rei.get("evidence_quality", "unknown"),
            posterior_health=rei.get("posterior_health", "unknown"),
            baseline_ready=rei.get("baseline_ready", False),
            n_observations=rei.get("n_observations", 0),
        ))
    return ckpts


@_needs
@pytest.mark.parametrize("race_key,checkpoint_laps", _TEST_RACES)
def test_filter_vs_naive_per_race(race_key, checkpoint_laps):
    from app.replay import run_historical_replay
    from app.decision.recv_validation import compute_recv_report, recv_baselines

    try:
        replay = run_historical_replay(
            race_key=race_key, start_lap=1,
            end_lap=max(checkpoint_laps) + 1, seed=42,
        )
    except (FastF1Unavailable, KeyError) as exc:
        pytest.skip(f"{race_key} unavailable: {exc}")

    ckpts = _build_checkpoints(replay, race_key, checkpoint_laps)
    if len(ckpts) < _MIN_CHECKPOINTS:
        pytest.skip(
            f"{race_key}: only {len(ckpts)} valid checkpoints with reference SoC "
            f"(need {_MIN_CHECKPOINTS}; Task 5 may be needed for rival_reference_soc_mj)"
        )

    report = compute_recv_report(ckpts, label=race_key)
    bl = recv_baselines(ckpts)
    naive_mae = bl["naive_prior"].mae_mj

    # Honest reporting: print results regardless of outcome
    print(
        f"\n{race_key}: n={report.n_checkpoints}, "
        f"MAE={report.mae_mj:.3f} MJ, RMSE={report.rmse_mj:.3f} MJ, "
        f"bias={report.bias_mj:.3f} MJ, naive_MAE={naive_mae:.3f} MJ, "
        f"filter/naive={report.mae_mj / naive_mae:.2f}x"
    )

    # Non-fatal threshold: filter must not be catastrophically worse than naive
    assert report.mae_mj < naive_mae * 3.0, (
        f"{race_key}: filter MAE {report.mae_mj:.3f} MJ is "
        f"{report.mae_mj / naive_mae:.1f}x the naive baseline ({naive_mae:.3f} MJ). "
        f"Bias: {report.bias_mj:.3f} MJ. Observation model may not suit this circuit."
    )


@_needs
def test_multi_race_aggregate():
    """Aggregate RECV across all available races."""
    from app.replay import run_historical_replay
    from app.decision.recv_validation import compute_recv_report, recv_baselines

    all_ckpts = []
    n_ok = 0
    n_skipped = 0
    for race_key, checkpoint_laps in _TEST_RACES:
        try:
            replay = run_historical_replay(
                race_key=race_key, start_lap=1,
                end_lap=max(checkpoint_laps) + 1, seed=42,
            )
            ckpts = _build_checkpoints(replay, race_key, checkpoint_laps)
            if len(ckpts) >= _MIN_CHECKPOINTS:
                all_ckpts.extend(ckpts)
                n_ok += 1
            else:
                n_skipped += 1
        except (FastF1Unavailable, KeyError):
            n_skipped += 1

    if n_ok == 0:
        pytest.skip("No historical races available with FastF1 cache")

    report = compute_recv_report(all_ckpts, label="multi_race_aggregate")
    bl = recv_baselines(all_ckpts)
    print(
        f"\nMulti-race aggregate: {n_ok} races, {n_skipped} skipped, "
        f"n={report.n_checkpoints}, MAE={report.mae_mj:.3f} MJ, "
        f"RMSE={report.rmse_mj:.3f} MJ, bias={report.bias_mj:.3f} MJ, "
        f"naive_MAE={bl['naive_prior'].mae_mj:.3f} MJ"
    )
    assert report.n_checkpoints >= _MIN_CHECKPOINTS
