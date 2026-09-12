"""
rival_soc.py — aggregate Rival SoC estimation metrics from per-lap trace steps.

Input: list of trace step dicts (from app.demo.rival_trace._walk()).
       Each step has "error_mj" (float or None) and "ground_truth_reserve_mj" (float or None).

Output: RivalSocValidation with MAE, RMSE, median AE, P90 AE.

Ground truth isolation: this module only reads "error_mj" from trace steps.
The trace is produced by the validation runner AFTER inference completes.
Ground truth never flows back into the estimator.
"""

from __future__ import annotations

from typing import List

import numpy as np

from app.validation.models import RivalSocValidation


def compute_rival_soc_metrics(trace_steps: List[dict]) -> RivalSocValidation:
    """
    Aggregate Rival SoC MAE/RMSE/median/P90 from per-lap trace steps.

    Only steps with error_mj != None count. Steps where ground truth was
    unavailable (FastF1, early laps before baseline ready) are excluded.

    The 'error_mj' in each step = estimated_reserve_mj − ground_truth_reserve_mj.
    Absolute error = abs(error_mj).
    """
    abs_errors = [
        abs(float(step["error_mj"]))
        for step in trace_steps
        if step.get("error_mj") is not None
    ]

    if not abs_errors:
        return RivalSocValidation(
            ground_truth_available=False,
            validation_type="unavailable",
            mae_mj=None,
            rmse_mj=None,
            median_ae_mj=None,
            p90_ae_mj=None,
            sample_count=0,
            evaluation_note=(
                "GROUND TRUTH UNAVAILABLE. FastF1 does not publish rival ERS SoC. "
                "Estimator quality assessed through uncertainty calibration and temporal consistency only."
            ),
        )

    arr = np.array(abs_errors, dtype=float)
    mae = float(np.mean(arr))
    rmse = float(np.sqrt(np.mean(arr**2)))
    median_ae = float(np.median(arr))
    p90_ae = float(np.percentile(arr, 90))

    return RivalSocValidation(
        ground_truth_available=True,
        validation_type="controlled_hidden_state",
        mae_mj=round(mae, 4),
        rmse_mj=round(rmse, 4),
        median_ae_mj=round(median_ae, 4),
        p90_ae_mj=round(p90_ae, 4),
        sample_count=len(abs_errors),
        evaluation_note=(
            f"Controlled hidden-state validation: {len(abs_errors)} laps. "
            "MODELED — NOT MEASURED. "
            "Synthetic simulator generated hidden rival SoC; estimator saw only kinematic observables. "
            "Not a claim of real-world F1 rival battery prediction accuracy."
        ),
    )
