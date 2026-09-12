"""
validation.py — score the Rival Energy Estimator against the simulator's hidden
ground truth. DEMO/VALIDATION ONLY.

The block this builds is returned by `/api/v1/demo/*` and NEVER by
`POST /api/v1/decision`. Its `ground_truth_*` fields exist purely so a judge can
see that the estimator (which only saw the four kinematic observables) tracks a
hidden number it was never told.
"""

from __future__ import annotations

from typing import Optional

from app.data.providers import SyntheticProvider, TelemetryProvider
from app.decision.context import DecisionContext


def build_rival_validation(
    ctx: DecisionContext,
    provider: TelemetryProvider,
) -> dict:
    r = ctx.rival
    est_obj = ctx._estimator  # the RivalStateEstimator instance the pipeline used
    target_lap = ctx.target_lap.lap

    if r is None or est_obj is None:
        return {
            "enabled": False,
            "reason": "no rival observations in the window — estimator not run",
        }

    gt: Optional[float] = None
    if isinstance(provider, SyntheticProvider):
        gt = provider.ground_truth_rival_soc_by_lap.get(target_lap)

    est = r.estimate
    block = {
        "enabled": True,
        "label": "ESTIMATED RIVAL ENERGY vs GROUND TRUTH — DEMO ONLY",
        "estimated_reserve_mj": est.mean_soc_mj,
        "estimated_std_mj": est.std_soc_mj,
        "bucket": r.bucket,
        "distribution": {k: round(v, 4) for k, v in r.distribution.items()},
        "confidence": r.confidence,
        "n_observations": est.n_observations,
        "estimate_uncertain": r.uncertain,
        "latest_observation": getattr(est_obj, "_last_observation", None),
        "particle_summary": est_obj.posterior_summary(),
        "ground_truth_reserve_mj": None,
        "error_mj": None,
        "abs_error_mj": None,
        "within_1_sigma": None,
        "ground_truth_note": (
            "GROUND TRUTH is the simulator's hidden rival SoC. It is NOT telemetry, "
            "the estimator never receives it, and it is absent from POST /api/v1/decision."
        ),
    }
    if gt is not None:
        err = round(est.mean_soc_mj - gt, 4)
        block.update({
            "ground_truth_reserve_mj": round(float(gt), 4),
            "error_mj": err,
            "abs_error_mj": round(abs(err), 4),
            "within_1_sigma": bool(abs(err) <= est.std_soc_mj),
        })
    else:
        block["ground_truth_note"] = "No ground truth: FastF1 does not publish rival ERS SoC."
    return block
