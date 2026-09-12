"""
quality.py — deterministic Data Quality Gate.

Runs BEFORE feature extraction. It does not fix or invent telemetry and it does
not choose a strategy. It produces a structured verdict that flows into the
confidence gate: bad or stale telemetry -> lower quality_score -> the confidence
gate becomes more conservative -> BALANCED_MODE if it has to.

Pure function of the lap list — no RNG, no clock.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from app.data.samples import NormalizedLap

QualityStatus = Literal["GOOD", "DEGRADED", "INVALID"]

# range sanity bounds (generous — these catch corruption, not tuning issues)
_SOC_MAX_MJ = 9.5
_SPEED_MAX_KMH = 400.0
_MIN_REPLAY_SAMPLES = 5  # a REPLAY lap condensed from fewer sub-lap samples is thin


class DataQuality(BaseModel):
    status: QualityStatus
    quality_score: float = Field(..., ge=0.0, le=1.0)
    freshness_laps: int = Field(
        ..., ge=0, description="Laps since the most recent usable rival observation (0 = fresh)"
    )
    dropped_samples: int = Field(..., ge=0, description="Missing lap numbers in the 1..N sequence")
    out_of_order: bool = Field(..., description="Lap history was not monotonically increasing")
    missing_fields: List[str]
    checks: List[str] = Field(..., description="Human-readable notes, one per issue found")

    @property
    def usable(self) -> bool:
        return self.status != "INVALID"


def assess_quality(
    lap_history: List[NormalizedLap],
    target_lap: NormalizedLap,
    data_mode: str,
) -> DataQuality:
    checks: List[str] = []
    missing: List[str] = []
    score = 1.0

    # --- sequence integrity ---
    lap_nums = [nl.lap for nl in lap_history]
    out_of_order = any(b < a for a, b in zip(lap_nums, lap_nums[1:]))
    if out_of_order:
        checks.append("lap history not monotonically increasing")
        score -= 0.15

    span = (max(lap_nums) - min(lap_nums) + 1) if lap_nums else 0
    dropped = max(0, span - len(set(lap_nums)))
    if dropped:
        checks.append(f"{dropped} lap(s) missing from the sequence")
        score -= min(0.30, 0.06 * dropped)

    # --- required fields on the target lap ---
    if target_lap.our_soc_mj is None:
        missing.append("our_soc_mj")
    if target_lap.our_speed_kmh is None or target_lap.our_speed_kmh <= 0:
        missing.append("our_speed_kmh")
    if target_lap.total_laps < target_lap.lap:
        missing.append("total_laps")
    if missing:
        checks.append("required field(s) missing/invalid: " + ", ".join(missing))
        score -= 0.5

    # --- range sanity on the target lap ---
    if target_lap.our_soc_mj is not None and not (0.0 <= target_lap.our_soc_mj <= _SOC_MAX_MJ):
        checks.append(f"SoC out of range: {target_lap.our_soc_mj}")
        score -= 0.4
    if target_lap.our_speed_kmh is not None and target_lap.our_speed_kmh > _SPEED_MAX_KMH:
        checks.append(f"implausible speed: {target_lap.our_speed_kmh} km/h")
        score -= 0.3
    if target_lap.gap_to_car_ahead_s is not None and target_lap.gap_to_car_ahead_s < 0:
        checks.append("negative gap ahead")
        score -= 0.2

    # --- thin replayed lap ---
    if data_mode == "REPLAY" and 0 < target_lap.raw_sample_count < _MIN_REPLAY_SAMPLES:
        checks.append(f"only {target_lap.raw_sample_count} sub-lap samples backed this lap")
        score -= 0.15

    # --- pit / out / invalid lap: real telemetry, but not clean racing evidence ---
    if getattr(target_lap, "lap_status", "racing") != "racing":
        checks.append(
            f"target lap is a '{target_lap.lap_status}' lap — not representative racing evidence"
        )
        score -= 0.3
    if getattr(target_lap, "rival_lap_status", "racing") != "racing":
        checks.append(
            f"rival on a '{target_lap.rival_lap_status}' lap — no valid rival observation this lap"
        )
        score -= 0.15

    # --- rival-observation freshness ---
    freshness = 0
    for nl in reversed(lap_history):
        if nl.has_rival_observation:
            break
        freshness += 1
    if freshness == len(lap_history):
        checks.append("no rival observation anywhere in the window")
        score -= 0.15
    elif freshness >= 3:
        checks.append(f"rival estimate is {freshness} laps stale")
        score -= min(0.20, 0.05 * freshness)

    score = max(0.0, min(1.0, score))

    if missing or (target_lap.our_soc_mj is not None and not (0.0 <= target_lap.our_soc_mj <= _SOC_MAX_MJ)):
        status: QualityStatus = "INVALID"
    elif score < 0.75 or checks:
        status = "DEGRADED"
    else:
        status = "GOOD"

    if not checks:
        checks.append("all data-quality checks passed")

    return DataQuality(
        status=status,
        quality_score=round(score, 4),
        freshness_laps=freshness,
        dropped_samples=dropped,
        out_of_order=out_of_order,
        missing_fields=missing,
        checks=checks,
    )
