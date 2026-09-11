"""
Feature vector for the overtake-success classifier.

Every feature is genuinely computed from a `NormalizedLap` and the event-time
window — no hardcoded placeholders. Order here is the contract: training,
inference, and the metadata sidecar all use it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Canonical feature order. Change here => retrain (train.py writes it to metadata,
# predict.py checks the width).
FEATURE_NAMES = [
    "gap_to_car_ahead_s",       # smaller = better
    "gap_trend_s_per_lap",      # negative = closing on the car ahead
    "our_soc_mj",               # energy available to spend
    "our_speed_kmh",            # pace
    "drs_available",            # 0 / 1 (Override / Overtake Mode eligibility proxy)
    "rival_terminal_speed_kmh", # lower = a slower / fading car ahead
]

_GAP_MISSING = 3.5
_RIVAL_SPEED_MISSING = 320.0


def build_features(
    gap_to_car_ahead_s: Optional[float],
    gap_trend_s_per_lap: Optional[float],
    our_soc_mj: float,
    our_speed_kmh: float,
    drs_available: Optional[bool],
    rival_terminal_speed_kmh: Optional[float],
) -> np.ndarray:
    """Return a (1, 6) float array in FEATURE_NAMES order. Unknowns -> neutral."""
    return np.array([[
        _GAP_MISSING if gap_to_car_ahead_s is None else float(gap_to_car_ahead_s),
        0.0 if gap_trend_s_per_lap is None else float(gap_trend_s_per_lap),
        float(our_soc_mj),
        float(our_speed_kmh),
        1.0 if drs_available else 0.0,
        _RIVAL_SPEED_MISSING if rival_terminal_speed_kmh is None else float(rival_terminal_speed_kmh),
    ]], dtype=np.float64)
