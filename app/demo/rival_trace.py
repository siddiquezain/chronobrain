"""
rival_trace.py — sequential replay of the Rival Energy Estimator.

Runs the SAME `RivalStateEstimator` (no second estimator) lap by lap over a
provider's telemetry, capturing the posterior after every observation so a
frontend can animate it:

    observation 1 -> estimate
    observation 2 -> update
    ...
    observation N -> final posterior

The estimator state genuinely updates each step — nothing is faked. In synthetic
mode the hidden ground truth is attached to each step for scoring; for FastF1
replay there is no ground truth (F1 publishes no rival SoC) and it is null.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from rival_estimator import RivalStateEstimator

from app.data.normalizer import to_rival_observation
from app.data.providers import SyntheticProvider, TelemetryProvider, build_provider
from app.decision.config import DecisionConfig


def _walk(
    provider: TelemetryProvider,
    cfg: DecisionConfig,
    up_to_lap: Optional[int],
) -> tuple[list[dict], RivalStateEstimator]:
    laps = provider.laps()
    if up_to_lap is not None:
        laps = [nl for nl in laps if nl.lap <= up_to_lap]
    if not laps:
        raise ValueError("no laps to trace")

    gt_by_lap = getattr(provider, "ground_truth_rival_soc_by_lap", {}) or {}
    lo, hi = cfg.rival_low_soc_mj, cfg.rival_high_soc_mj

    est = RivalStateEstimator(config=cfg.rival_config(), seed=cfg.seed)
    prev_std: Optional[float] = None
    steps: list[dict] = []

    for nl in laps:
        obs = to_rival_observation(nl)
        had_obs = obs is not None
        if had_obs:
            est.predict()
            est.update(obs)

        e = est.estimate()
        ess = float(1.0 / np.sum(est._weights**2))
        gt = gt_by_lap.get(nl.lap)
        trend = "flat"
        if prev_std is not None:
            if e.std_soc_mj < prev_std - 0.02:
                trend = "more_certain"
            elif e.std_soc_mj > prev_std + 0.02:
                trend = "less_certain"
        prev_std = e.std_soc_mj

        steps.append({
            "lap": nl.lap,
            "had_observation": had_obs,
            "observation": est._last_observation if had_obs else None,
            "estimated_reserve_mj": e.mean_soc_mj,
            "estimated_std_mj": e.std_soc_mj,
            "n_observations": e.n_observations,
            "bucket": e.bucket(lo, hi),
            "distribution": {k: round(v, 4) for k, v in e.bucket_distribution(lo, hi).items()},
            "effective_sample_size": round(ess, 1),
            "uncertainty_trend": trend,
            "ground_truth_reserve_mj": None if gt is None else round(float(gt), 4),
            "error_mj": None if gt is None else round(e.mean_soc_mj - float(gt), 4),
        })

    return steps, est


def rival_estimator_trace(
    *,
    source: str = "synthetic",
    scenario: str = "B",
    seed: int = 42,
    total_laps: int = 50,
    up_to_lap: Optional[int] = None,
    preset_overrides: Optional[dict] = None,
    rival_obs_dropout: float = 0.0,
    telemetry_glitch: bool = False,
    fastf1_kwargs: Optional[dict] = None,
) -> dict:
    """Full sequential trace + a compact final particle summary."""
    cfg = DecisionConfig(seed=seed)
    provider = build_provider(
        source, scenario=scenario, seed=seed, total_laps=total_laps,
        preset_overrides=preset_overrides, rival_obs_dropout=rival_obs_dropout,
        telemetry_glitch=telemetry_glitch, fastf1_kwargs=fastf1_kwargs,
    )
    steps, est = _walk(provider, cfg, up_to_lap)
    ground_truth_available = isinstance(provider, SyntheticProvider)
    return {
        "source": source,
        "ground_truth_available": ground_truth_available,
        "note": (
            "GROUND TRUTH — DEMO ONLY. The estimator never receives it; it only "
            "sees the four kinematic observables per lap."
            if ground_truth_available else
            "No ground truth: FastF1 does not publish rival ERS SoC."
        ),
        "steps": steps,
        "final_particle_summary": est.posterior_summary(),
    }
