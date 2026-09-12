"""
window.py — lightweight event-time / sliding-window feature layer.

ChronoPace must not commit finite energy off a single telemetry point. This
computes short-horizon TRENDS over the last few laps so the decision engine and
the opportunity horizon can see whether the situation is improving or decaying.

Deterministic: sorts + de-duplicates by lap number (tolerating out-of-order
input), then linear-fits each channel. No RNG, no streaming machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

from app.data.samples import NormalizedLap

OpportunityTrend = Literal["IMPROVING", "STABLE", "DECAYING"]


@dataclass
class WindowFeatures:
    n_laps: int
    speed_trend_kmh_per_lap: float
    gap_ahead_trend_s_per_lap: Optional[float]   # negative = closing on the car ahead
    soc_trend_mj_per_lap: Optional[float]
    rival_terminal_speed_trend: Optional[float]  # negative = rival slowing (fading)
    rival_sector_delta_trend: Optional[float]    # positive = rival losing time
    closing: bool                                # gap to car ahead is shrinking
    opportunity_trend: OpportunityTrend
    stale_laps: int                              # laps at the end of the window with no rival obs


def _slope(xs: List[float], ys: List[Optional[float]]) -> Optional[float]:
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def compute_window(lap_history: List[NormalizedLap], window: int = 5) -> WindowFeatures:
    # sort + dedupe by lap (keep the last sample seen for a given lap number)
    by_lap = {nl.lap: nl for nl in sorted(lap_history, key=lambda n: n.lap)}
    laps = sorted(by_lap)[-window:]
    rows = [by_lap[l] for l in laps]
    xs = [float(l) for l in laps]

    speed_slope = _slope(xs, [r.our_speed_kmh for r in rows]) or 0.0
    gap_slope = _slope(xs, [r.gap_to_car_ahead_s for r in rows])
    soc_slope = _slope(xs, [r.our_soc_mj for r in rows])
    rival_speed_slope = _slope(xs, [r.rival_terminal_speed_kmh for r in rows])
    rival_sector_slope = _slope(xs, [r.rival_sector_delta_s for r in rows])

    closing = gap_slope is not None and gap_slope < -0.01

    stale = 0
    for r in reversed(rows):
        if r.has_rival_observation:
            break
        stale += 1

    # opportunity trend: closing gap and/or a fading rival => improving.
    signal = 0.0
    if gap_slope is not None:
        signal += -gap_slope * 3.0            # closing gap pushes positive
    if rival_speed_slope is not None:
        signal += -rival_speed_slope * 0.05   # rival slowing pushes positive
    if rival_sector_slope is not None:
        signal += rival_sector_slope * 2.0    # rival losing time pushes positive
    if signal > 0.15:
        trend: OpportunityTrend = "IMPROVING"
    elif signal < -0.15:
        trend = "DECAYING"
    else:
        trend = "STABLE"

    return WindowFeatures(
        n_laps=len(rows),
        speed_trend_kmh_per_lap=round(speed_slope, 4),
        gap_ahead_trend_s_per_lap=None if gap_slope is None else round(gap_slope, 4),
        soc_trend_mj_per_lap=None if soc_slope is None else round(soc_slope, 4),
        rival_terminal_speed_trend=None if rival_speed_slope is None else round(rival_speed_slope, 4),
        rival_sector_delta_trend=None if rival_sector_slope is None else round(rival_sector_slope, 4),
        closing=bool(closing),
        opportunity_trend=trend,
        stale_laps=stale,
    )
