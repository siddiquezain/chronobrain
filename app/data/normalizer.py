"""
normalizer.py — TelemetrySample[] -> NormalizedLap, and NormalizedLap -> Stack A.

Two jobs:

1. `condense_lap()` reduces a burst of sub-lap `TelemetrySample`s (from FastF1 or
   any future feed) into one `NormalizedLap`, including a small deterministic
   energy model because real F1 timing carries no ERS SoC.

2. `to_telemetry_input()` / `to_rival_observation()` adapt a `NormalizedLap` to
   the Stack A single-lap contracts. The decision engine calls these; it does not
   touch `TelemetrySample` or `fastf1`.

Everything here is a pure function of its inputs — no RNG, no clocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from app.data.samples import DataMode, NormalizedLap, TelemetrySample

# Stack A contracts
from rule_gate import GateConfig
from telemetry_simulator import RivalObservation, TelemetryInput

_GATE = GateConfig()
_SOC_CEIL_MJ = _GATE.max_deployment_per_lap_mj  # 9.0 — Stack A clamps SoC to this


# ---------------------------------------------------------------------------
# Deterministic energy model for replay (F1 publishes no ERS SoC)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayEnergyModel:
    """
    Turns a lap's throttle/brake trace into a modelled SoC path. **MODEL_ASSUMPTION**,
    not a regulation and NOT a measurement — real F1 telemetry carries no ERS SoC,
    and the 2024 cars did not run ChronoPace's 2026 energy budget. Deterministic:
    same samples in, same numbers out.

    Two modes:
      * `lap_deploy_mj` / `lap_harvest_mj` — the original absolute model (kept for
        callers that use it directly).
      * `next_soc` — a causal, mean-reverting model: SoC pulls toward a nominal
        level and dips/recovers with how hard THIS lap was worked relative to the
        driver's OWN rolling baseline (which only ever contains earlier laps). This
        is what the historical replay uses so SoC stays in a plausible band instead
        of draining to zero on a flat-out circuit like Monza.
    """

    start_soc_mj: float = 4.5
    nominal_soc_mj: float = 4.5
    reversion: float = 0.12           # per-lap pull toward nominal
    intensity_gain_mj: float = 3.0    # SoC swing per unit of relative work intensity
    floor_mj: float = 0.3
    # Full-throttle lap would spend this; scaled by mean throttle fraction.
    max_deploy_per_lap_mj: float = 2.6
    # Full-braking-share lap would recover this; scaled by mean brake fraction.
    max_harvest_per_lap_mj: float = 2.3

    def lap_deploy_mj(self, mean_throttle: float) -> float:
        return round(self.max_deploy_per_lap_mj * _clip01(mean_throttle), 4)

    def lap_harvest_mj(self, mean_brake: float) -> float:
        return round(self.max_harvest_per_lap_mj * _clip01(mean_brake), 4)

    def next_soc(
        self,
        prev_soc: float,
        mean_throttle: float,
        mean_brake: float,
        throttle_baseline: float,
        brake_baseline: float,
    ) -> float:
        rel_intensity = (mean_throttle - throttle_baseline) - 0.5 * (mean_brake - brake_baseline)
        pull = self.reversion * (self.nominal_soc_mj - prev_soc)
        soc = prev_soc + pull - self.intensity_gain_mj * rel_intensity
        return _clip(soc, self.floor_mj, 9.0)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in vals if v is not None]
    return sum(present) / len(present) if present else None


# ---------------------------------------------------------------------------
# condense: samples -> NormalizedLap
# ---------------------------------------------------------------------------
def condense_lap(
    our_samples: Sequence[TelemetrySample],
    rival_samples: Sequence[TelemetrySample],
    *,
    lap: int,
    total_laps: int,
    data_mode: DataMode,
    prev_soc_mj: Optional[float],
    gap_to_car_ahead_s: Optional[float],
    gap_to_car_behind_s: Optional[float] = None,
    energy_model: Optional[ReplayEnergyModel] = None,
    rival_sector_baseline_s: Optional[float] = None,
    throttle_baseline: Optional[float] = None,
    brake_baseline: Optional[float] = None,
    source_detail: str = "",
) -> NormalizedLap:
    """
    Reduce one lap's sub-lap samples (our car + rival) to a NormalizedLap.

    `prev_soc_mj` threads the modelled SoC forward lap-to-lap (None on the first
    lap -> energy_model.start_soc_mj). When `throttle_baseline` / `brake_baseline`
    are supplied (rolling means of EARLIER laps only), the mean-reverting
    `next_soc` model is used instead of the absolute one.
    """
    if not our_samples:
        raise ValueError(f"condense_lap: no samples for our car on lap {lap}")

    model = energy_model or ReplayEnergyModel()

    mean_throttle = _mean([s.throttle for s in our_samples]) or 0.0
    mean_brake = _mean([s.brake for s in our_samples]) or 0.0

    lap_start_soc = model.start_soc_mj if prev_soc_mj is None else prev_soc_mj
    deployed = model.lap_deploy_mj(mean_throttle)
    harvested = model.lap_harvest_mj(mean_brake)
    if throttle_baseline is not None and brake_baseline is not None:
        end_soc = round(model.next_soc(
            lap_start_soc, mean_throttle, mean_brake, throttle_baseline, brake_baseline
        ), 4)
    else:
        end_soc = round(_clip(lap_start_soc + harvested - deployed, 0.0, _SOC_CEIL_MJ), 4)

    our_speed = max(s.speed_kmh for s in our_samples)
    drs_open = any(bool(s.drs) for s in our_samples if s.drs is not None) or None
    position = next((s.position for s in reversed(our_samples) if s.position is not None), None)
    sector = next((s.sector for s in reversed(our_samples) if s.sector is not None), None)

    rival_fields = _rival_observables(rival_samples, rival_sector_baseline_s)

    return NormalizedLap(
        lap=lap,
        total_laps=total_laps,
        data_mode=data_mode,
        our_speed_kmh=round(our_speed, 3),
        our_soc_mj=end_soc,
        our_lap_start_soc_mj=round(lap_start_soc, 4),
        our_lap_energy_deployed_mj=deployed,
        gap_to_car_ahead_s=gap_to_car_ahead_s,
        gap_to_car_behind_s=gap_to_car_behind_s,
        position=position,
        sector=sector,
        drs_available=drs_open,
        energy_is_modeled=True,
        raw_sample_count=len(our_samples),
        source_detail=source_detail,
        **rival_fields,
    )


def _rival_observables(
    rival_samples: Sequence[TelemetrySample],
    sector_baseline_s: Optional[float],
) -> dict:
    """Derive the 4 particle-filter observables from a rival's sub-lap samples."""
    if not rival_samples:
        return dict(
            rival_terminal_speed_kmh=None,
            rival_clipping_point_fraction=None,
            rival_corner_exit_accel_g=None,
            rival_sector_delta_s=None,
        )

    terminal_speed = max(s.speed_kmh for s in rival_samples)

    # clipping point: fraction along the lap where speed stops rising materially.
    ordered = sorted(rival_samples, key=lambda s: (s.distance_m or s.timestamp_s))
    peak_i = max(range(len(ordered)), key=lambda i: ordered[i].speed_kmh)
    clip_fraction = peak_i / max(1, len(ordered) - 1)

    # corner-exit accel proxy: largest positive dv/dt across consecutive samples.
    max_accel_g = 0.0
    for a, b in zip(ordered, ordered[1:]):
        dt = b.timestamp_s - a.timestamp_s
        if dt <= 0:
            continue
        dv_ms = (b.speed_kmh - a.speed_kmh) / 3.6
        max_accel_g = max(max_accel_g, dv_ms / dt / 9.81)

    # sector delta: this lap's implied lap time vs a rolling baseline.
    lap_time_s = ordered[-1].timestamp_s - ordered[0].timestamp_s
    sector_delta = (
        round(lap_time_s - sector_baseline_s, 4) if sector_baseline_s is not None else 0.0
    )

    return dict(
        rival_terminal_speed_kmh=round(terminal_speed, 3),
        rival_clipping_point_fraction=round(_clip01(clip_fraction), 4),
        rival_corner_exit_accel_g=round(_clip(max_accel_g, 0.0, 3.0), 4),
        rival_sector_delta_s=sector_delta,
    )


# ---------------------------------------------------------------------------
# adapt: NormalizedLap -> Stack A
# ---------------------------------------------------------------------------
def to_telemetry_input(
    lap: NormalizedLap,
    *,
    overtake_qualified_last_lap: bool,
) -> TelemetryInput:
    """
    NormalizedLap -> rule_gate.TelemetryInput. `our_soc_mj` must be populated
    (synthetic supplies it; replay models it). Raises rather than inventing a SoC.
    """
    if lap.our_soc_mj is None:
        raise ValueError(
            f"NormalizedLap(lap={lap.lap}) has no our_soc_mj — a provider must "
            "supply or model SoC before the regulatory gate can run."
        )
    return TelemetryInput(
        lap_number=lap.lap,
        current_soc_mj=round(_clip(lap.our_soc_mj, 0.0, _SOC_CEIL_MJ), 4),
        lap_start_soc_mj=(
            round(_clip(lap.our_lap_start_soc_mj, 0.0, _SOC_CEIL_MJ), 4)
            if lap.our_lap_start_soc_mj is not None
            else None
        ),
        lap_energy_deployed_mj=lap.our_lap_energy_deployed_mj,
        gap_to_car_ahead_s=lap.gap_to_car_ahead_s,
        gap_to_car_behind_s=lap.gap_to_car_behind_s,
        overtake_qualified_last_lap=overtake_qualified_last_lap,
        speed_kmh=lap.our_speed_kmh,
        total_laps=lap.total_laps,
    )


def to_rival_observation(lap: NormalizedLap) -> Optional[RivalObservation]:
    """NormalizedLap -> RivalObservation, or None when the source lacks rival data."""
    if not lap.has_rival_observation:
        return None
    return RivalObservation(
        terminal_speed_kmh=lap.rival_terminal_speed_kmh,
        clipping_point_fraction=lap.rival_clipping_point_fraction,
        corner_exit_accel_g=lap.rival_corner_exit_accel_g,
        sector_delta_s=lap.rival_sector_delta_s,
    )
