"""
historical.py — replay a real F1 race, lap by lap, through the existing pipeline.

The only ChronoPace-specific work here is:
  1. slicing the loaded lap list to laps <= N   (the hindsight barrier), and
  2. compacting each DecisionSnapshot into a per-lap summary.

Everything else is `app.data.fastf1_service.load_replay` + `app.decision.run_decision`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.data.fastf1_service import FastF1Unavailable, load_replay
from app.data.providers import ReplayProvider
from app.data.samples import NormalizedLap
from app.decision import DecisionConfig, DecisionSnapshot, run_decision
from app.replay.races import HISTORICAL_RACES, HistoricalRace, resolve_race

_DEFAULT_CACHE = ".fastf1_cache"

# In-process cache of the *parsed* session (FastF1 caches the raw download itself).
# Keyed so different driver pairings / races don't collide.
_SESSION_CACHE: Dict[Tuple[str, str, str], List[NormalizedLap]] = {}


def resolve_any_race(
    race_key: Optional[str] = None,
    *,
    season: Optional[int] = None,
    event: Optional[str] = None,
    session: Optional[str] = None,
    driver: Optional[str] = None,
    rival: Optional[str] = None,
) -> HistoricalRace:
    """A curated registry key, OR an on-the-fly race from FastF1's schedule
    (`season` + `event`). Featured races keep their default driver/rival + note;
    ad-hoc races need `driver` supplied by the caller."""
    if race_key and str(race_key).strip().lower() in HISTORICAL_RACES:
        return resolve_race(race_key)
    if season is None:
        raise KeyError(
            f"unknown race {race_key!r}; pass a known key ({sorted(HISTORICAL_RACES)}) "
            f"or a `season` + `event` from GET /api/v1/replay/races?season=..."
        )
    from app.replay.discovery import resolve_session
    ev_name = event or race_key
    if not ev_name:
        raise KeyError("a non-registry race needs an `event` (name or round number)")
    info = resolve_session(int(season), str(ev_name), session or "R")
    if not driver:
        raise ValueError("a non-registry race needs `driver` (3-letter code)")
    return HistoricalRace(
        key=f"{season}_{info['event'].lower().replace(' ', '_')}_{info['session'].lower()}",
        name=info["name"], circuit=info["circuit"], year=info["year"],
        event=info["event"], session=info["session"],
        scheduled_laps=info["scheduled_laps"] or 0,
        default_driver=(driver or "").upper(), default_rival=(rival or "").upper(),
        note="ad-hoc replay from the FastF1 schedule",
    )


def session_cache_key(race_key: str, driver: str, rival: str, dynamic_rival: bool) -> Tuple:
    """The `_SESSION_CACHE` key for a (race, pairing, mode). Public so tests can
    pre-seed the cache and stay offline."""
    return (race_key, driver.upper(), rival.upper(), f"dyn={bool(dynamic_rival)}")


def _load_race_laps(
    race: HistoricalRace, driver: str, rival: str, cache_dir: str,
    dynamic_rival: bool = True,
) -> List[NormalizedLap]:
    key = session_cache_key(race.key, driver, rival, dynamic_rival)
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = load_replay(
            year=race.year,
            event=race.event,
            session=race.session,
            our_driver=driver.upper(),
            rival_driver=rival.upper(),   # focus / fallback rival when dynamic
            cache_dir=cache_dir,
            label=race.name,
            scheduled_laps=race.scheduled_laps,
            dynamic_rival=dynamic_rival,
        )
    return _SESSION_CACHE[key]


def clear_session_cache() -> None:
    _SESSION_CACHE.clear()


# ---------------------------------------------------------------------------
# per-lap summary — NO ground truth, NO future data
# ---------------------------------------------------------------------------
_ACTION_TO_LABEL = {
    "ATTACK_NOW": "ATTACK",
    "WAIT_2_LAPS": "WAIT",
    "WAIT_5_LAPS": "WAIT",
    "PUSH": "PUSH",
    "HOLD": "HOLD",
    "CONSERVE": "CONSERVE",
}


def lap_summary(snap: DecisionSnapshot, nl: Optional[NormalizedLap] = None) -> dict:
    d = snap.decision
    o = snap.opportunity
    mc = snap.monte_carlo
    r = snap.rival
    return {
        "lap": snap.meta.lap,
        "decision": _ACTION_TO_LABEL.get(d.action, d.action),
        "action": d.action,
        "mode": d.mode,
        "confidence": d.confidence,
        "confidence_overridden": d.confidence_overridden,
        "override_reason": d.override_reason or None,
        "reason_codes": snap.reason_codes,
        "strategic_rival": {   # who matters this lap and why (dynamic causal selection)
            "driver": r.driver,
            "role": r.role,
            "position": r.strategic_position,
            "gap_s": r.strategic_gap_s,
            "ahead": r.strategic_rival_ahead,
            "relevance_score": r.relevance_score,
            "tick_level": (nl.strategic_rival.tick_level if nl is not None and nl.strategic_rival else False),
            "changes_this_lap": (nl.strategic_rival.changes_this_lap if nl is not None and nl.strategic_rival else 0),
            "tick_share": (dict(list((nl.strategic_rival.tick_share or {}).items())[:4])
                           if nl is not None and nl.strategic_rival else {}),
        },
        "rival_energy_inference": {   # inferred from observable performance — NOT measured
            "label": "RIVAL ENERGY INFERENCE (probabilistic, from observable performance)",
            "driver": r.driver,   # whose observables this estimate is built from
            "estimated_reserve_mj": r.mean_reserve_mj,
            "std_mj": r.reserve_std_mj,
            "bucket": r.bucket,
            "distribution": r.distribution,
            "confidence": r.confidence,
            "n_observations": r.n_observations,
            "p_defend": r.p_defend,
            # posterior health fields
            "posterior_mean_mj": r.mean_reserve_mj,
            "posterior_std_mj": r.reserve_std_mj,
            "effective_sample_size": r.effective_sample_size,
            "evidence_quality": r.evidence_quality,
            "posterior_health": r.posterior_health,
            "baseline_ready": r.baseline_ready,
            "energy_provenance": "INFERRED",
        },
        # rival's own ChronoPace-modeled SoC — not loaded in the ego replay path
        # ponytail: set None; populate if/when rival laps are co-loaded
        "rival_reference_soc_mj": None,
        "rival_reference_provenance": "CHRONOPACE_MODELED",
        "opportunity": {
            "recommended_strategy": o.recommended_strategy,
            "prefers_wait": o.prefers_wait,
            "opportunity_trend": o.opportunity_trend,
            "current_window_overtake_prob": o.current_window_overtake_prob,
            "foregone_strategy": o.foregone_strategy,
            "foregone_value_gap_s": o.foregone_value_gap_s,
        },
        "monte_carlo": {
            "n_iterations": mc.n_iterations,
            "seed": mc.seed,
            "recommended_mode": mc.recommended_mode,
            "ranked_modes": [
                {"mode": m.mode, "mean_laptime_delta_s": m.mean_laptime_delta_s,
                 "overtake_probability": m.overtake_probability, "sharpe_ratio": m.sharpe_ratio}
                for m in mc.ranked_modes
            ],
        },
        "compliance": {
            "legal": snap.compliance.legal,
            "legal_modes": snap.compliance.legal_modes,
        },
        "energy": {   # MODELED — F1 publishes no ERS SoC; this is ChronoPace's 2026 model
            "label": "MODELED CHRONOPACE 2026 ENERGY STATE (not measured — F1 publishes no ERS SoC)",
            "soc_mj": snap.energy.soc_mj,
            "soc_pct": snap.energy.soc_pct,
            "soc_capacity_mj": snap.energy.soc_capacity_mj,
            "lap_start_soc_mj": snap.energy.lap_start_soc_mj,
            "deployed_this_lap_mj": snap.energy.deployed_this_lap_mj,
            "recovered_this_lap_mj": snap.energy.recovered_this_lap_mj,
            "net_swing_mj": snap.energy.net_swing_mj,
            "deployment_headroom_mj": snap.energy.deployment_headroom_mj,
            "projected_reserve_mj": snap.energy.projected_reserve_mj,
            "modeled_mgu_k_peak_kw": snap.energy.modeled_mgu_k_peak_kw,
            "mgu_k_power_ceiling_kw": snap.energy.mgu_k_power_ceiling_kw,
            "can_afford_aggressive": snap.energy.can_afford_aggressive,
            "energy_is_modeled": snap.energy.energy_is_modeled,
            "accounting": "SoC_next = SoC + recovered - deployed (net of a nominal lap), clipped [floor, capacity]",
        },
        "telemetry_real": {   # REAL 2024 FastF1 observation for this lap
            "mean_throttle": (nl.our_mean_throttle if nl is not None else None),
            "mean_brake": (nl.our_mean_brake if nl is not None else None),
            "top_speed_kmh": (nl.our_speed_kmh if nl is not None else None),
        },
        "data_quality": {
            "status": snap.data_quality.status,
            "quality_score": snap.data_quality.quality_score,
        },
        "gap_to_rival_s": _relative_gap(nl),
        "rival_role": _rival_role(nl),
        "position": (nl.position if nl is not None else None),
        "overtake_mode_eligible": (nl.overtake_mode_eligible if nl is not None else None),
        "our_speed_kmh": (nl.our_speed_kmh if nl is not None else None),
        "lap_status": (nl.lap_status if nl is not None else None),
        "rival_lap_status": (nl.rival_lap_status if nl is not None else None),
        "provenance": _LAP_PROVENANCE,
        "trace": [{"stage": s.stage, "detail": s.detail} for s in snap.trace],
    }


_LAP_PROVENANCE = {
    "REAL (2024 FastF1 observation)": [
        "speed, throttle, brake, gear, RPM, DRS, distance",
        "lap & sector timing, running position, relative gap",
    ],
    "MODELED (ChronoPace 2026, MODEL_ASSUMPTION)": [
        "own SoC / deployment / recovery / net swing / MGU-K peak power",
        "opportunity probability, Monte Carlo outcomes, confidence, the deployment-mode decision",
    ],
    "INFERRED (probabilistic, from observable performance)": [
        "the strategic rival's hidden energy distribution (mean / std / bucket) — no car's real ERS SoC is public or used",
    ],
    "note": "REAL 2024 OBSERVATION + CHRONOPACE 2026 MODEL = REPLAYED 2026 DECISION CONTEXT. "
            "The 2024 cars did NOT run under 2026 energy regulations.",
}


def _relative_gap(nl: Optional[NormalizedLap]) -> Optional[float]:
    """The gap to the rival, whichever side they are on (None during a pit cycle)."""
    if nl is None:
        return None
    return nl.gap_to_car_ahead_s if nl.gap_to_car_ahead_s is not None else nl.gap_to_car_behind_s


def _rival_role(nl: Optional[NormalizedLap]) -> Optional[str]:
    """'attacking' when the rival is the car ahead, 'defending' when they are the
    car behind, 'clear' when there is no measurable relative gap (e.g. pit cycle)."""
    if nl is None:
        return None
    if nl.gap_to_car_ahead_s is not None:
        return "attacking"
    if nl.gap_to_car_behind_s is not None:
        return "defending"
    return "clear"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _censored_provider(full: List[NormalizedLap], upto_lap: int, race_name: str) -> ReplayProvider:
    """HINDSIGHT BARRIER: the provider handed to the engine physically contains
    only laps <= upto_lap."""
    censored = [nl for nl in full if nl.lap <= upto_lap]
    return ReplayProvider(censored, label=f"{race_name} (through L{upto_lap})")


def run_historical_lap(
    *,
    race_key: Optional[str] = None,
    driver: Optional[str] = None,
    rival: Optional[str] = None,
    lap: int,
    seed: int = 42,
    cache_dir: str = _DEFAULT_CACHE,
    full_snapshot: bool = False,
    dynamic_rival: bool = True,
    season: Optional[int] = None,
    event: Optional[str] = None,
    session: Optional[str] = None,
) -> dict:
    race = resolve_any_race(race_key, season=season, event=event, session=session, driver=driver, rival=rival)
    drv = (driver or race.default_driver).upper()
    riv = (rival or race.default_rival).upper()
    full = _load_race_laps(race, drv, riv, cache_dir, dynamic_rival)
    available = [nl.lap for nl in full]
    if lap not in available:
        raise ValueError(f"lap {lap} not in the replay ({min(available)}..{max(available)})")

    provider = _censored_provider(full, lap, race.name)
    snap = run_decision(provider, lap=lap, config=DecisionConfig(seed=seed))
    target_nl = next(nl for nl in full if nl.lap == lap)
    out = {
        "race": _race_meta(race, drv, riv, len(available)),
        "lap": lap,
        "censored_to_lap": lap,
        "summary": lap_summary(snap, target_nl),
    }
    if full_snapshot:
        out["snapshot"] = snap.model_dump()
    return out


def strategic_rival_timeline(
    *,
    race_key: str,
    lap: int,
    driver: Optional[str] = None,
    rival: Optional[str] = None,
    cache_dir: str = _DEFAULT_CACHE,
    max_ticks: int = 400,
    debug: bool = False,
    season: Optional[int] = None,
    event: Optional[str] = None,
    session: Optional[str] = None,
) -> dict:
    """The telemetry-tick strategic-rival stream for ONE lap — the detailed view
    behind the compact per-lap summary (spec §14: heavy detail only on request).

    Causal: each tick's pick depends only on data <= that tick, so slicing the
    race-wide selection to this lap is identical to having stopped at this lap.
    """
    race = resolve_any_race(race_key, season=season, event=event, session=session, driver=driver, rival=rival)
    drv = (driver or race.default_driver).upper()
    riv = (rival or race.default_rival).upper()

    # ensure the dynamic replay (and thus the FieldTimeline) is built + cached
    _load_race_laps(race, drv, riv, cache_dir, dynamic_rival=True)

    from app.data.fastf1_service import field_timeline_cache_key, get_cached_field_timeline
    ft = get_cached_field_timeline(field_timeline_cache_key(race.year, race.event, race.session, drv))
    if ft is None:
        raise FastF1Unavailable(
            "no telemetry-tick reconstruction available for this replay "
            "(the lap-level selector was used) — no tick timeline to show"
        )

    tl = ft.selection_timeline()
    summaries = ft.lap_summaries(timeline=tl)
    lap_ticks = [s for s in tl if s.lap == lap]
    if not lap_ticks:
        raise ValueError(f"lap {lap} has no telemetry ticks in this replay")

    step = max(1, len(lap_ticks) // max_ticks)
    sampled = lap_ticks[::step]
    s = summaries.get(lap)
    t0 = lap_ticks[0].t
    return {
        "race": _race_meta(race, drv, riv, len({x.lap for x in tl})),
        "lap": lap,
        "our_driver": drv,
        "focus_rival": riv,
        "tick_cadence_hz": round(len(lap_ticks) / max(lap_ticks[-1].t - t0, 1e-6), 1),
        "total_ticks_this_lap": len(lap_ticks),
        "returned_ticks": len(sampled),
        "downsample_step": step,
        "summary": None if s is None else {
            "dominant_driver": s.dominant_driver, "dominant_role": s.dominant_role,
            "dominant_ahead": s.dominant_ahead, "dominant_gap_s": s.dominant_gap_s,
            "first_driver": s.first_driver, "last_driver": s.last_driver,
            "n_changes": s.n_changes, "share_by_driver": s.share_by_driver,
        },
        "change_events": s.change_events if s is not None else [],
        "ticks": [
            {
                "t": round(x.t - t0, 3), "session_time_s": round(x.t, 3),
                "driver": x.driver, "role": x.role, "ahead": x.ahead,
                "gap_s": x.gap_s, "relevance": x.relevance_score,
                "switched": x.switched, "raw_leader": x.raw_leader,
            }
            for x in sampled
        ],
        "debug": (
            _timeline_debug_rows(ft, lap, t0, step)
            if debug else
            "pass ?debug=true for the per-tick, per-candidate score breakdown + switch reason"
        ),
        "provenance": {
            "REAL": "FastF1 per-driver car telemetry (SessionTime, speed) + official running position",
            "MODELED": "which opponent is 'strategically relevant' at each tick — deterministic score "
                       "over reconstructed track gap / adjacency / closing rate / pace (MODEL_ASSUMPTION)",
        },
    }


def _timeline_debug_rows(ft, lap: int, t0: float, step: int) -> list:
    """Per-tick candidate scores + switch reason, so a rival switch can be
    reconstructed: `t | PIA score | NOR score | selected | switch_reason`.
    Every number comes from the engine — nothing hardcoded."""
    rows = ft.debug_timeline(lap=lap)
    return [
        {
            "t": round(r["t"] - t0, 3),
            "session_time_s": round(r["t"], 3),
            "selected": r["selected"],
            "raw_leader": r["raw_leader"],
            "switched": r["switched"],
            "switch_reason": r["switch_reason"],
            "dwell_ticks": r["dwell_ticks"],
            "candidates": sorted(
                r["candidates"], key=lambda c: (c["relevance"] is None, -(c["relevance"] or 0.0))
            ),
        }
        for r in rows[::step]
    ]


def run_historical_replay(
    *,
    race_key: str,
    driver: Optional[str] = None,
    rival: Optional[str] = None,
    start_lap: int = 1,
    end_lap: Optional[int] = None,
    seed: int = 42,
    cache_dir: str = _DEFAULT_CACHE,
    dynamic_rival: bool = True,
    season: Optional[int] = None,
    event: Optional[str] = None,
    session: Optional[str] = None,
) -> dict:
    race = resolve_any_race(race_key, season=season, event=event, session=session, driver=driver, rival=rival)
    drv = (driver or race.default_driver).upper()
    riv = (rival or race.default_rival).upper()
    full = _load_race_laps(race, drv, riv, cache_dir, dynamic_rival)
    available = [nl.lap for nl in full]
    start = max(start_lap, min(available))
    end = min(end_lap or max(available), max(available))

    by_lap = {nl.lap: nl for nl in full}
    cfg = DecisionConfig(seed=seed)
    laps_out: List[dict] = []
    for n in range(start, end + 1):
        if n not in available:
            continue
        provider = _censored_provider(full, n, race.name)
        snap = run_decision(provider, lap=n, config=cfg)
        laps_out.append(lap_summary(snap, by_lap[n]))

    changes = _strategic_rival_changes(full, start, end)
    tick_level = any(nl.strategic_rival and nl.strategic_rival.tick_level for nl in full)
    intra = sum(nl.strategic_rival.changes_this_lap for nl in full
                if nl.strategic_rival and start <= nl.lap <= end)
    n_replayed = max(1, end - start + 1)

    return {
        "race": _race_meta(race, drv, riv, len(available)),
        "driver": drv,
        "rival": riv,
        "telemetry": _telemetry_meta(race, drv),
        "total_laps": race.scheduled_laps or max(available),
        "laps_replayed": [start, end],
        "source": "fastf1_historical_replay",
        "model": "chronopace_2026",
        "strategic_rival": {
            "dynamic": dynamic_rival,
            "tick_level": tick_level,
            "focus_rival": riv,
            "selector": "app.replay.field_state + app.replay.strategic_rival (deterministic, causal)",
            "note": (
                "The strategic rival is chosen at every FastF1 telemetry tick from the "
                "whole field, using only data <= that tick; the per-lap value is the "
                "dominant rival by time-share. 'focus_rival' is the fallback."
                if tick_level else
                "The rival is re-selected from the full field every lap using only data "
                "<= that lap. 'focus_rival' is the fallback when no opponent is relevant."
                if dynamic_rival else
                "Fixed two-car analysis: the rival is the configured driver every lap."
            ),
            "changes": changes,   # lap-boundary identity changes (kept for compatibility)
            "lap_boundary_changes": changes,
            "intra_lap_changes_total": intra,
            "avg_switches_per_lap": round((len(changes) + intra) / n_replayed, 2),
            "drivers_tracked": sorted({
                nl.strategic_rival.driver for nl in full
                if nl.strategic_rival is not None and nl.strategic_rival.driver
            }),
            "timeline_endpoint": f"GET /api/v1/replay/historical/{race.key}/{{lap}}/timeline",
        },
        "provenance": {
            "REAL": [
                "2024 lap & sector timing, speed, throttle, brake, gear, RPM, DRS, distance "
                "(FastF1 / official F1 timing)",
                "actual observable driver performance lap by lap, for the whole field",
                "running positions and relative gaps (cumulative lap-time difference through lap N)",
            ],
            "MODELED (MODEL_ASSUMPTION)": [
                "which opponent is 'strategically relevant' — a deterministic score over "
                "observable position / gap / trend / pace (app.replay.strategic_rival)",
                "rival hidden energy state — INFERRED probabilistically from the SELECTED "
                "rival's observable performance; no car's real ERS SoC is public or used",
                "our own SoC / energy budget — a 2026-model mean-reverting trajectory "
                "(energy_is_modeled=true on every lap)",
                "the ChronoPace 2026 decision model, Monte Carlo outcomes, opportunity "
                "probabilities, and regulatory-legality assumptions",
            ],
            "note": "2024 cars did NOT run under ChronoPace's 2026 energy rules. Real 2024 "
                    "telemetry flows through the 2026 decision model. "
                    "HISTORICAL TELEMETRY REPLAY · CHRONOPACE 2026 MODEL.",
        },
        "laps": laps_out,
    }


def _telemetry_meta(race: HistoricalRace, driver: str) -> dict:
    """Honest telemetry source + cadence metadata. FastF1 historical car telemetry
    is NOT a 128 Hz raw stream — the actual cadence is source-dependent (~4 Hz on
    2024 data). Measured from the reconstructed tick grid when available."""
    from app.data.fastf1_service import field_timeline_cache_key, get_cached_field_timeline
    cadence_hz = None
    ticks_total = None
    ft = get_cached_field_timeline(field_timeline_cache_key(race.year, race.event, race.session, driver))
    if ft is not None and getattr(ft, "ticks", None) is not None and len(ft.ticks) > 2:
        import numpy as np
        dt = float(np.median(np.diff(ft.ticks)))
        cadence_hz = round(1.0 / dt, 1) if dt > 0 else None
        ticks_total = int(len(ft.ticks))
    return {
        "source": "FastF1 historical (official F1 timing + car telemetry archive)",
        "kind": "HISTORICAL TELEMETRY REPLAY",
        "cadence_hz_measured": cadence_hz,
        "cadence_note": "SOURCE-DEPENDENT / ~4 Hz typical for 2024 car data — NOT a 128 Hz raw stream",
        "ticks_total": ticks_total,
        "is_live": False,
        "is_synthetic": False,
    }


def _strategic_rival_changes(full: List[NormalizedLap], start: int, end: int) -> List[dict]:
    """Laps within [start, end] where the selected strategic rival's identity changed."""
    out: List[dict] = []
    prev: Optional[str] = None
    for nl in sorted(full, key=lambda n: n.lap):
        if nl.strategic_rival is None:
            continue
        cur = nl.strategic_rival.driver
        if prev is not None and cur != prev and start <= nl.lap <= end:
            out.append({
                "lap": nl.lap,
                "from": prev,
                "to": cur,
                "role": nl.strategic_rival.role,
                "gap_s": nl.strategic_rival.gap_s,
                "relevance_score": nl.strategic_rival.relevance_score,
            })
        prev = cur
    return out


def _race_meta(race: HistoricalRace, drv: str, riv: str, n_available: int) -> dict:
    return {
        "key": race.key,
        "name": race.name,
        "circuit": race.circuit,
        "year": race.year,
        "session": race.session,
        "scheduled_laps": race.scheduled_laps,
        "laps_with_telemetry": n_available,
        "driver": drv,
        "rival": riv,
        "note": race.note,
    }
