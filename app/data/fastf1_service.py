"""
fastf1_service.py — the only module that knows FastF1 exists.

Everything FastF1-specific is contained here: cache setup, `get_session`, the
`session.load()` call, per-lap car-data DataFrames, DRS/throttle/brake encoding,
driver and session selection, gap computation. The output is
`list[NormalizedLap]` — the decision engine never imports this module.

`fastf1` is imported lazily inside `load_replay()`, so importing this module (for
`dataframe_to_samples`, which tests use) never requires the package.

HONESTY NOTE: FastF1 exposes car motion and FIA timing only. Formula 1 does not
publish ERS state of charge for any car, so SoC and per-lap deployment on a
replayed lap are produced by `normalizer.ReplayEnergyModel` (a MODEL_ASSUMPTION),
and every replayed `NormalizedLap` carries `energy_is_modeled=True`. This is
historical replay of public data — not a live feed and not team telemetry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.data.normalizer import ReplayEnergyModel, condense_lap
from app.data.samples import NormalizedLap, TelemetrySample

_DEFAULT_CACHE = ".fastf1_cache"


# ---------------------------------------------------------------------------
# Pure helper — no fastf1 required (tests drive this with a synthetic DataFrame)
# ---------------------------------------------------------------------------
def dataframe_to_samples(df, lap: int) -> List[TelemetrySample]:
    """
    A FastF1 car-data DataFrame (or any frame with the same columns) -> samples.

    Recognised columns (all optional except a speed column):
      Time (timedelta|seconds), SessionTime, Speed, Throttle (0-100), Brake (bool|0-1),
      RPM, nGear|Gear, DRS (FastF1 code), Distance, Sector, Position.
    """
    cols = {c.lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    speed_c = col("speed")
    if speed_c is None:
        raise ValueError("dataframe_to_samples: need a 'Speed' column")
    time_c = col("time", "sessiontime", "date")
    thr_c = col("throttle")
    brk_c = col("brake")
    rpm_c = col("rpm")
    gear_c = col("ngear", "gear")
    drs_c = col("drs")
    dist_c = col("distance")
    sec_c = col("sector")
    pos_c = col("position")

    t0 = None
    out: List[TelemetrySample] = []
    for _, row in df.iterrows():
        ts = _to_seconds(row[time_c]) if time_c else float(len(out))
        if t0 is None:
            t0 = ts
        out.append(
            TelemetrySample(
                timestamp_s=round(ts - t0, 4),
                lap=lap,
                speed_kmh=float(row[speed_c]),
                distance_m=_opt_float(row, dist_c),
                throttle=_opt_frac(row, thr_c, scale=100.0),
                brake=_brake_to_frac(row, brk_c),
                rpm=_opt_float(row, rpm_c),
                gear=_opt_int(row, gear_c),
                drs=_drs_open(row, drs_c),
                position=_opt_int(row, pos_c),
                sector=_opt_int(row, sec_c),
            )
        )
    return out


# ---------------------------------------------------------------------------
# FastF1-backed loader (lazy import)
# ---------------------------------------------------------------------------
def load_replay(
    *,
    year: int,
    event,
    session: str = "R",
    our_driver: str,
    rival_driver: str,
    laps: Optional[Sequence[int]] = None,
    cache_dir: str = _DEFAULT_CACHE,
    energy_model: Optional[ReplayEnergyModel] = None,
    label: Optional[str] = None,
) -> List[NormalizedLap]:
    """
    Load a historical session and return NormalizedLaps for `our_driver`, with
    `rival_driver`'s kinematics attached as particle-filter observations.

    `event` is anything FastF1 accepts (round number or GP name). `laps` limits
    which lap numbers to return (default: all laps both drivers completed).
    """
    import fastf1  # lazy — see module docstring

    import os

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    ses = fastf1.get_session(year, event, session)
    ses.load(telemetry=True, laps=True, weather=False, messages=False)

    our_samples = _driver_lap_samples(ses, our_driver)
    rival_samples = _driver_lap_samples(ses, rival_driver)
    gap = _gap_to_rival_s(ses, our_driver, rival_driver)

    lap_numbers = sorted(set(our_samples) & set(rival_samples))
    if laps is not None:
        wanted = set(laps)
        lap_numbers = [ln for ln in lap_numbers if ln in wanted]
    if not lap_numbers:
        raise RuntimeError(
            f"No overlapping laps for {our_driver} / {rival_driver} in {year} {event} {session}"
        )
    total_laps = int(ses.total_laps or lap_numbers[-1])

    model = energy_model or ReplayEnergyModel()
    src = label or f"{year} {getattr(ses, 'event', {}).get('EventName', event)} {session}, {our_driver} vs {rival_driver}"

    # rolling baseline for the rival sector-delta signal
    rival_lap_times = _rival_lap_time_baseline(rival_samples)

    out: List[NormalizedLap] = []
    prev_soc: Optional[float] = None
    for ln in lap_numbers:
        nl = condense_lap(
            our_samples[ln],
            rival_samples.get(ln, []),
            lap=ln,
            total_laps=total_laps,
            data_mode="REPLAY",
            prev_soc_mj=prev_soc,
            gap_to_car_ahead_s=gap.get(ln),
            energy_model=model,
            rival_sector_baseline_s=rival_lap_times.get(ln),
            source_detail=src,
        )
        prev_soc = nl.our_soc_mj
        out.append(nl)
    return out


# ---------------------------------------------------------------------------
# fastf1 internals
# ---------------------------------------------------------------------------
def _driver_lap_samples(session, driver: str) -> Dict[int, List[TelemetrySample]]:
    laps = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
    by_lap: Dict[int, List[TelemetrySample]] = {}
    for _, lap in laps.iterlaps():
        ln = int(lap["LapNumber"])
        try:
            car = lap.get_car_data().add_distance()
        except Exception:
            continue
        df = car.reset_index()
        pos_val = lap.get("Position")
        samples = dataframe_to_samples(df, ln)
        if pos_val is not None and not _is_nan(pos_val):
            for s in samples:
                s.position = int(pos_val)
        if samples:
            by_lap[ln] = samples
    return by_lap


def _gap_to_rival_s(session, our_driver: str, rival_driver: str) -> Dict[int, float]:
    """Approximate per-lap gap as |cumulative lap-time difference| at each lap end."""
    def cum(driver):
        d = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
        total = 0.0
        acc = {}
        for _, lap in d.iterlaps():
            lt = lap["LapTime"]
            secs = _to_seconds(lt) if lt is not None and not _is_nan(lt) else None
            if secs is None:
                continue
            total += secs
            acc[int(lap["LapNumber"])] = total
        return acc

    ours, theirs = cum(our_driver), cum(rival_driver)
    return {ln: round(abs(ours[ln] - theirs[ln]), 3) for ln in set(ours) & set(theirs)}


def _rival_lap_time_baseline(rival_samples: Dict[int, List[TelemetrySample]]) -> Dict[int, float]:
    """Rolling mean of the rival's implied lap time (last-sample timestamp) up to each lap."""
    times = {ln: s[-1].timestamp_s for ln, s in rival_samples.items() if s}
    baseline: Dict[int, float] = {}
    seen: List[float] = []
    for ln in sorted(times):
        if seen:
            baseline[ln] = sum(seen) / len(seen)
        seen.append(times[ln])
    return baseline


# ---------------------------------------------------------------------------
# column coercion
# ---------------------------------------------------------------------------
def _to_seconds(v) -> float:
    if hasattr(v, "total_seconds"):
        return float(v.total_seconds())
    try:
        import numpy as np

        if isinstance(v, np.timedelta64):
            return float(v / np.timedelta64(1, "s"))
    except Exception:
        pass
    return float(v)


def _is_nan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


def _opt_float(row, c):
    if not c or _is_nan(row[c]):
        return None
    return float(row[c])


def _opt_int(row, c):
    if not c or row[c] is None or _is_nan(row[c]):
        return None
    return int(row[c])


def _opt_frac(row, c, scale=1.0):
    v = _opt_float(row, c)
    return None if v is None else max(0.0, min(1.0, v / scale))


def _brake_to_frac(row, c):
    if not c or row[c] is None or _is_nan(row[c]):
        return None
    v = row[c]
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return max(0.0, min(1.0, float(v)))


def _drs_open(row, c):
    """FastF1 DRS codes: 0/1/2/3 closed or eligible, 8/10/12/14 open/active."""
    if not c or row[c] is None or _is_nan(row[c]):
        return None
    return int(row[c]) >= 10
