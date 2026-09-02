"""Shared math helpers used across engines."""

import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0.0:
        return default
    return numerator / denominator


def sharpe(mean: float, std: float, cap: float = 999.0) -> float:
    """Return mean/std, capped at ±cap when std ≈ 0."""
    if std < 1e-9:
        return cap if mean >= 0 else -cap
    return float(np.clip(mean / std, -cap, cap))
