"""Generate synthetic training data for the overtake success classifier."""

import numpy as np

from app.ml.features import FEATURE_NAMES

_RNG = np.random.default_rng(42)

N_SAMPLES = 5_000


def generate() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns X (N, 8) and y (N,) for the overtake success classifier.

    Labels are heuristic: high score on gap, closing speed, slipstream, SoC,
    and DRS → success = 1.
    """
    gap = _RNG.uniform(0.1, 3.0, N_SAMPLES)
    closing = _RNG.uniform(-2.0, 15.0, N_SAMPLES)
    slipstream = _RNG.uniform(0.0, 1.0, N_SAMPLES)
    soc = _RNG.uniform(0.5, 9.0, N_SAMPLES)
    tyre_age = _RNG.integers(0, 40, N_SAMPLES).astype(float)
    straight = _RNG.uniform(100.0, 800.0, N_SAMPLES)
    speed = _RNG.uniform(200.0, 350.0, N_SAMPLES)
    drs = _RNG.integers(0, 2, N_SAMPLES).astype(float)

    X = np.column_stack([gap, closing, slipstream, soc, tyre_age, straight, speed, drs])

    # Heuristic label
    score = (
        (1.0 - gap / 3.0) * 0.3
        + np.clip(closing / 15.0, 0, 1) * 0.25
        + slipstream * 0.15
        + (soc / 9.0) * 0.15
        + drs * 0.15
    )
    y = (score + _RNG.normal(0, 0.05, N_SAMPLES) > 0.5).astype(int)
    return X, y
