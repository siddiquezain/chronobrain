"""Feature extraction for the overtake success classifier."""

import numpy as np

from app.models.telemetry import TelemetryState

# Canonical 8 features — order must match training
FEATURE_NAMES = [
    "gap_to_car_ahead_s",
    "closing_speed_mps",
    "slipstream_factor",
    "soc_mj",
    "tyre_age_laps",
    "straight_distance_m",
    "speed_kmh",
    "drs_available",
]


def extract(telemetry: TelemetryState) -> np.ndarray:
    """Return shape (1, 8) feature array from a TelemetryState."""
    return np.array(
        [[
            telemetry.gap_to_car_ahead_s or 3.0,
            telemetry.closing_speed_mps,
            telemetry.slipstream_factor,
            telemetry.soc_mj,
            telemetry.tyre_age_laps,
            telemetry.straight_distance_m,
            telemetry.speed_kmh,
            float(telemetry.drs_available),
        ]],
        dtype=np.float64,
    )
