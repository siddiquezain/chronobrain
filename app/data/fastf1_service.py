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


class FastF1Unavailable(RuntimeError):
    """`fastf1` is not installed, or the historical session could not be loaded
    (offline / cache miss / unknown session). Raised so callers fail clearly
    instead of fabricating data."""


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
def fastf1_available() -> bool:
    try:
        import fastf1  # noqa: F401
        return True
    except Exception:
        return False


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
    scheduled_laps: Optional[int] = None,
) -> List[NormalizedLap]:
    """
    Load a historical session and return causal per-lap `NormalizedLap`s for
    `our_driver`, with `rival_driver`'s kinematics attached as particle-filter
    observations.

    ANTI-HINDSIGHT: every field on lap N is derived from lap N alone plus rolling
    statistics of laps < N (throttle/brake baselines, rival sector-delta baseline,
    cumulative gap). `scheduled_laps` (the pre-race race distance, a constant from
    the race registry) is used for `total_laps` rather than the actually-completed
    count. Downstream, `run_decision(provider, lap=N)` censors again to laps <= N.

    Raises `FastF1Unavailable` if `fastf1` is missing or the session cannot load.
    """
    try:
        import fastf1  # lazy — see module docstring
    except Exception as exc:  # ModuleNotFoundError, or a broken install
        raise FastF1Unavailable(
            "the `fastf1` package is not installed. Run `pip install -r requirements-fastf1.txt`."
        ) from exc

    import os

    try:
        try:
            fastf1.set_log_level("WARNING")  # quiet the per-request INFO spam
        except Exception:
            pass
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        ses = fastf1.get_session(year, event, session)
        ses.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception as exc:
        raise FastF1Unavailable(
            f"could not load {year} {event} {session} from FastF1 "
            f"(offline, cache miss, or unknown session): {exc}"
        ) from exc

    our_samples = _driver_lap_samples(ses, our_driver)
    rival_samples = _driver_lap_samples(ses, rival_driver)
    if not our_samples or not rival_samples:
        raise FastF1Unavailable(
            f"no telemetry for {our_driver!r} and/or {rival_driver!r} in {year} {event} {session}"
        )
    gap = _gap_to_rival_s(ses, our_driver, rival_driver)

    # per-lap running position and lap-fitness status (pit / out / invalid), and a
    # cumulative-lap-time fallback for who-leads when Position is missing.
    our_status = _lap_status_by_lap(ses, our_driver)
    rival_status = _lap_status_by_lap(ses, rival_driver)
    our_pos = _positions_by_lap(ses, our_driver)
    rival_pos = _positions_by_lap(ses, rival_driver)
    our_cum = _cumulative_laptime(ses, our_driver)
    rival_cum = _cumulative_laptime(ses, rival_driver)

    lap_numbers = sorted(set(our_samples) & set(rival_samples))
    if laps is not None:
        wanted = set(laps)
        lap_numbers = [ln for ln in lap_numbers if ln in wanted]
    if not lap_numbers:
        raise FastF1Unavailable(
            f"no overlapping laps for {our_driver} / {rival_driver} in {year} {event} {session}"
        )
    total_laps = int(scheduled_laps or ses.total_laps or lap_numbers[-1])

    model = energy_model or ReplayEnergyModel()
    ev_name = event
    try:
        ev_name = ses.event["EventName"]
    except Exception:
        pass
    src = label or f"{year} {ev_name} {session}, {our_driver} vs {rival_driver}"

    # rolling (causal) baselines — built from RACING laps only so a pit lap never
    # poisons the baseline that later laps are measured against.
    rival_lap_times = _rival_lap_time_baseline(rival_samples, rival_status)
    our_thr = {
        ln: _mean_channel(s, "throttle")
        for ln, s in our_samples.items()
        if our_status.get(ln, "racing") == "racing"
    }
    our_brk = {
        ln: _mean_channel(s, "brake")
        for ln, s in our_samples.items()
        if our_status.get(ln, "racing") == "racing"
    }
    thr_base = _rolling_mean(our_thr)
    brk_base = _rolling_mean(our_brk)
    all_thr = {ln: _mean_channel(s, "throttle") for ln, s in our_samples.items()}
    all_brk = {ln: _mean_channel(s, "brake") for ln, s in our_samples.items()}

    out: List[NormalizedLap] = []
    prev_soc: Optional[float] = None
    for ln in lap_numbers:
        o_status = our_status.get(ln, "racing")
        r_status = rival_status.get(ln, "racing")

        # who is ahead on track? Position when both are known, else the sign of the
        # cumulative lap-time difference (both causal — through lap N only).
        we_lead = _we_lead(
            our_pos.get(ln), rival_pos.get(ln), our_cum.get(ln), rival_cum.get(ln)
        )

        # a pit cycle on EITHER car makes the relative gap meaningless — do not let
        # it become an overtake window or a defensive signal.
        g = gap.get(ln)
        gap_usable = g if (o_status == "racing" and r_status == "racing") else None

        nl = condense_lap(
            our_samples[ln],
            rival_samples.get(ln, []),
            lap=ln,
            total_laps=total_laps,
            data_mode="REPLAY",
            prev_soc_mj=prev_soc,
            gap_to_car_ahead_s=(None if we_lead else gap_usable),
            gap_to_car_behind_s=(gap_usable if we_lead else None),
            energy_model=model,
            rival_sector_baseline_s=rival_lap_times.get(ln),
            throttle_baseline=thr_base.get(ln, all_thr.get(ln)),
            brake_baseline=brk_base.get(ln, all_brk.get(ln)),
            lap_status=o_status,
            rival_lap_status=r_status,
            source_detail=src,
        )
        prev_soc = nl.our_soc_mj
        out.append(nl)
    return out


def _mean_channel(samples: List[TelemetrySample], attr: str) -> float:
    vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _rolling_mean(by_lap: Dict[int, float]) -> Dict[int, float]:
    """lap -> mean of ALL EARLIER laps' values (causal; no value for the first lap)."""
    out: Dict[int, float] = {}
    seen: List[float] = []
    for ln in sorted(by_lap):
        if seen:
            out[ln] = sum(seen) / len(seen)
        seen.append(by_lap[ln])
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


def _rival_lap_time_baseline(
    rival_samples: Dict[int, List[TelemetrySample]],
    rival_status: Optional[Dict[int, str]] = None,
) -> Dict[int, float]:
    """Rolling mean of the rival's implied lap time (last-sample timestamp) up to
    each lap, over RACING laps only (a pit lap would otherwise skew the baseline
    every subsequent lap)."""
    status = rival_status or {}
    times = {
        ln: s[-1].timestamp_s
        for ln, s in rival_samples.items()
        if s and status.get(ln, "racing") == "racing"
    }
    baseline: Dict[int, float] = {}
    seen: List[float] = []
    for ln in sorted(times):
        if seen:
            baseline[ln] = sum(seen) / len(seen)
        seen.append(times[ln])
    return baseline


def _lap_status_by_lap(session, driver: str) -> Dict[int, str]:
    """
    Per-lap fitness as clean racing evidence, from data known at that lap's end:
      * `PitInTime` present   -> 'pit'      (in-lap)
      * `PitOutTime` present  -> 'out_lap'
      * FastF1 `IsAccurate` False (standing start, lap deleted, timing glitch)
                              -> 'invalid_for_energy_inference'
      * otherwise             -> 'racing'
    Causal: every column here is a property of that lap, not a future one.
    """
    laps = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
    out: Dict[int, str] = {}
    for _, lap in laps.iterlaps():
        ln = int(lap["LapNumber"])
        pit_in = lap.get("PitInTime")
        pit_out = lap.get("PitOutTime")
        if pit_in is not None and not _is_nan(pit_in):
            out[ln] = "pit"
        elif pit_out is not None and not _is_nan(pit_out):
            out[ln] = "out_lap"
        elif not bool(lap.get("IsAccurate", True)):
            out[ln] = "invalid_for_energy_inference"
        else:
            out[ln] = "racing"
    return out


def _positions_by_lap(session, driver: str) -> Dict[int, Optional[int]]:
    laps = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
    out: Dict[int, Optional[int]] = {}
    for _, lap in laps.iterlaps():
        ln = int(lap["LapNumber"])
        pos = lap.get("Position")
        out[ln] = int(pos) if pos is not None and not _is_nan(pos) else None
    return out


def _we_lead(
    our_pos: Optional[int],
    rival_pos: Optional[int],
    our_cum: Optional[float],
    rival_cum: Optional[float],
) -> bool:
    """Is our driver ahead of the rival on this lap? Running position when both are
    known (lower number = ahead), else the cumulative-lap-time sign (less elapsed
    time = ahead). Both signals are causal (through lap N only)."""
    if our_pos is not None and rival_pos is not None:
        return our_pos < rival_pos
    return our_cum is not None and rival_cum is not None and our_cum < rival_cum


def _cumulative_laptime(session, driver: str) -> Dict[int, float]:
    """Cumulative sum of completed lap times through each lap — the fallback signal
    for who-leads when running Position is missing."""
    d = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
    total = 0.0
    acc: Dict[int, float] = {}
    for _, lap in d.iterlaps():
        lt = lap["LapTime"]
        secs = _to_seconds(lt) if lt is not None and not _is_nan(lt) else None
        if secs is None:
            continue
        total += secs
        acc[int(lap["LapNumber"])] = total
    return acc


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
