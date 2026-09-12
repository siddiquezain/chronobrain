"""
Synthetic training data for the overtake-success classifier.

Synthetic on purpose — real per-attempt F1 overtake outcome data is not public.
The label encodes uncontroversial domain structure (small closing gap + energy +
Overtake Mode eligibility + a slower car ahead => more likely to complete the pass). It is NOT
calibrated against real outcomes.
"""

from __future__ import annotations

import numpy as np

from app.ml.features import FEATURE_NAMES

_RNG = np.random.default_rng(42)
N_SAMPLES = 6_000


def generate() -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, 6) and y (N,) in FEATURE_NAMES order."""
    gap = _RNG.uniform(0.1, 3.5, N_SAMPLES)
    gap_trend = _RNG.uniform(-0.6, 0.4, N_SAMPLES)        # negative = closing
    soc = _RNG.uniform(0.5, 9.0, N_SAMPLES)
    speed = _RNG.uniform(200.0, 350.0, N_SAMPLES)
    overtake_eligible = _RNG.integers(0, 2, N_SAMPLES).astype(float)
    rival_speed = _RNG.uniform(290.0, 345.0, N_SAMPLES)

    X = np.column_stack([gap, gap_trend, soc, speed, overtake_eligible, rival_speed])
    assert X.shape[1] == len(FEATURE_NAMES)

    # structural score in [0, ~1]
    score = (
        np.clip(1.0 - gap / 3.0, 0.0, 1.0) * 0.30           # close
        + np.clip(-gap_trend / 0.6, 0.0, 1.0) * 0.25        # closing fast
        + (soc / 9.0) * 0.15                                # have energy
        + overtake_eligible * 0.15                          # Overtake Mode eligibility (gap <= 1.0 s)
        + np.clip((325.0 - rival_speed) / 35.0, 0.0, 1.0) * 0.15  # slower car ahead
    )
    y = (score + _RNG.normal(0.0, 0.06, N_SAMPLES) > 0.55).astype(int)
    return X, y
