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

from app.data.samples import (
    DataMode,
    LapStatus,
    NormalizedLap,
    StrategicRivalInfo,
    TelemetrySample,
)

# Stack A contracts
from rule_gate import GateConfig
from telemetry_simulator import RivalObservation, TelemetryInput

_GATE = GateConfig()
_SOC_CEIL_MJ = _GATE.max_deployment_per_lap_mj  # 9.0 — Stack A clamps SoC to this

# A rival lap whose implied lap time is this far off its own rolling baseline is a
# pit lap / safety-car lap / grossly compromised lap — not racing evidence. This
# is the backstop that keeps a +15/+20 s pit-lap sector delta from ever being read
# as "rival battery is empty" even if an upstream pit flag were missing.
# MODEL_ASSUMPTION — an engineered cutoff, not a measured quantity.
_RIVAL_RACING_MAX_ABS_SECTOR_DELTA_S = 5.0


# ---------------------------------------------------------------------------
# Deterministic energy model for replay (F1 publishes no ERS SoC)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayEnergyModel:
    """
    Turns a lap's throttle/brake trace into a modelled ERS energy path.
    **MODEL_ASSUMPTION** — not a regulation and NOT a measurement: real F1 telemetry
    carries no ERS state of charge, and the 2024 cars did not run ChronoPace's 2026
    energy budget. Deterministic: same samples in, same numbers out.

    `next_soc` / `next_soc_and_components` implement an EXPLICIT per-lap accounting
    relationship, causal (baselines only ever contain earlier laps):

        deployed(lap)  = max_deploy_per_lap_mj  * throttle_fraction
        recovered(lap) = max_harvest_per_lap_mj * brake_fraction
                       + coast_recovery_mj      * (1 - throttle_fraction)
        net_swing      = (recovered - deployed) - (recovered_base - deployed_base)
        SoC_next       = clip( SoC + net_swing + soft_anchor, floor_mj, capacity_mj )

    `net_swing` is measured against a NOMINAL lap (the driver's rolling
    throttle/brake baseline) so a normal lap is roughly net-neutral and the state
    genuinely accumulates instead of being pinned to a fixed value. `soft_anchor`
    is a gentle pull toward `nominal_soc_mj` (`reversion` is small — it stops slow
    drift over 50+ laps, it does not lock the value).

    `lap_deploy_mj` / `lap_harvest_mj` remain for the absolute legacy branch.
    """

    start_soc_mj: float = 4.5
    nominal_soc_mj: float = 4.5
    reversion: float = 0.05           # small per-lap pull toward nominal (anti-drift, not a lock)
    intensity_gain_mj: float = 3.0    # legacy — kept for callers of the old formula
    floor_mj: float = 0.3
    capacity_mj: float = 9.0          # SoC ceiling in ChronoPace's 0-9 MJ SoC scale (MODEL_ASSUMPTION)
    # Full-throttle lap would spend this; scaled by mean throttle fraction.
    max_deploy_per_lap_mj: float = 2.6
    # Full-braking-share lap would recover this; scaled by mean brake fraction.
    max_harvest_per_lap_mj: float = 2.3
    # Off-throttle harvesting (lift & coast, engine braking) — scaled by how far
    # below flat-out the lap ran. Tuned so a nominal racing lap is ~net-neutral.
    coast_recovery_mj: float = 1.9

    def lap_deploy_mj(self, mean_throttle: float) -> float:
        return round(self.max_deploy_per_lap_mj * _clip01(mean_throttle), 4)

    def lap_harvest_mj(self, mean_brake: float) -> float:
        return round(self.max_harvest_per_lap_mj * _clip01(mean_brake), 4)

    def lap_energy(self, mean_throttle: float, mean_brake: float) -> tuple:
        """Modeled (deployed_mj, recovered_mj) for one lap of work. MODEL_ASSUMPTION."""
        thr = _clip01(mean_throttle)
        brk = _clip01(mean_brake)
        deployed = self.max_deploy_per_lap_mj * thr
        recovered = self.max_harvest_per_lap_mj * brk + self.coast_recovery_mj * (1.0 - thr)
        return round(deployed, 4), round(recovered, 4)

    def next_soc_and_components(
        self,
        prev_soc: float,
        mean_throttle: float,
        mean_brake: float,
        throttle_baseline: float,
        brake_baseline: float,
    ) -> tuple:
        """Returns (soc_next, deployed_mj, recovered_mj, net_swing_mj)."""
        dep, rec = self.lap_energy(mean_throttle, mean_brake)
        base_dep, base_rec = self.lap_energy(throttle_baseline, brake_baseline)
        net = (rec - dep) - (base_rec - base_dep)
        anchor = self.reversion * (self.nominal_soc_mj - prev_soc)
        soc = _clip(prev_soc + net + anchor, self.floor_mj, self.capacity_mj)
        return round(soc, 4), dep, rec, round(net, 4)

    def next_soc(
        self,
        prev_soc: float,
        mean_throttle: float,
        mean_brake: float,
        throttle_baseline: float,
        brake_baseline: float,
    ) -> float:
        return self.next_soc_and_components(
            prev_soc, mean_throttle, mean_brake, throttle_baseline, brake_baseline
        )[0]


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
    lap_status: LapStatus = "racing",
    rival_lap_status: LapStatus = "racing",
    strategic_rival: Optional[StrategicRivalInfo] = None,
    rival_compound: Optional[str] = None,
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
    recovered = harvested
    net_swing: Optional[float] = None
    if lap_status != "racing":
        # A pit in/out lap's throttle/brake trace (pit-lane limiter, box stop) is
        # not representative work — do not model an energy swing through it.
        end_soc = round(lap_start_soc, 4)
        deployed, recovered, net_swing = 0.0, 0.0, 0.0
    elif throttle_baseline is not None and brake_baseline is not None:
        end_soc, deployed, recovered, net_swing = model.next_soc_and_components(
            lap_start_soc, mean_throttle, mean_brake, throttle_baseline, brake_baseline
        )
    else:
        end_soc = round(_clip(lap_start_soc + harvested - deployed, 0.0, _SOC_CEIL_MJ), 4)
        recovered = harvested
        net_swing = round(recovered - deployed, 4)

    our_speed = max(s.speed_kmh for s in our_samples)
    drs_open = any(bool(s.drs) for s in our_samples if s.drs is not None) or None
    position = next((s.position for s in reversed(our_samples) if s.position is not None), None)
    sector = next((s.sector for s in reversed(our_samples) if s.sector is not None), None)

    rival_fields = _rival_observables(rival_samples, rival_sector_baseline_s)

    # Backstop: a rival lap whose implied time is grossly off its own baseline is
    # not racing evidence, regardless of any upstream pit flag.
    if (
        rival_lap_status == "racing"
        and rival_fields["rival_sector_delta_s"] is not None
        and abs(rival_fields["rival_sector_delta_s"]) > _RIVAL_RACING_MAX_ABS_SECTOR_DELTA_S
    ):
        rival_lap_status = "invalid_for_energy_inference"

    # When the rival lap is not clean racing, drop the four derived observables so
    # the particle filter simply gets no observation that lap (it predicts, it does
    # not update — no collapse) and the event-time window skips the point.
    if rival_lap_status != "racing":
        rival_fields = dict(
            rival_terminal_speed_kmh=None,
            rival_clipping_point_fraction=None,
            rival_corner_exit_accel_g=None,
            rival_sector_delta_s=None,
        )

    return NormalizedLap(
        lap=lap,
        total_laps=total_laps,
        data_mode=data_mode,
        our_speed_kmh=round(our_speed, 3),
        our_soc_mj=end_soc,
        our_soc_capacity_mj=model.capacity_mj,
        our_lap_start_soc_mj=round(lap_start_soc, 4),
        our_lap_energy_deployed_mj=deployed,
        our_lap_energy_recovered_mj=recovered,
        our_lap_net_swing_mj=net_swing,
        our_mean_throttle=round(_clip01(mean_throttle), 4),
        our_mean_brake=round(_clip01(mean_brake), 4),
        gap_to_car_ahead_s=gap_to_car_ahead_s,
        gap_to_car_behind_s=gap_to_car_behind_s,
        position=position,
        sector=sector,
        overtake_mode_eligible=drs_open,
        lap_status=lap_status,
        rival_lap_status=rival_lap_status,
        strategic_rival=strategic_rival,
        rival_compound=rival_compound,
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
