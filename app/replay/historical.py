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
from app.replay.races import HistoricalRace, resolve_race

_DEFAULT_CACHE = ".fastf1_cache"

# In-process cache of the *parsed* session (FastF1 caches the raw download itself).
# Keyed so different driver pairings / races don't collide.
_SESSION_CACHE: Dict[Tuple[str, str, str], List[NormalizedLap]] = {}


def _load_race_laps(
    race: HistoricalRace, driver: str, rival: str, cache_dir: str
) -> List[NormalizedLap]:
    key = (race.key, driver.upper(), rival.upper())
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = load_replay(
            year=race.year,
            event=race.event,
            session=race.session,
            our_driver=driver.upper(),
            rival_driver=rival.upper(),
            cache_dir=cache_dir,
            label=race.name,
            scheduled_laps=race.scheduled_laps,
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
        "rival_energy_inference": {   # inferred from observable performance — NOT measured
            "label": "RIVAL ENERGY INFERENCE (probabilistic, from observable performance)",
            "estimated_reserve_mj": r.mean_reserve_mj,
            "std_mj": r.reserve_std_mj,
            "bucket": r.bucket,
            "distribution": r.distribution,
            "confidence": r.confidence,
            "n_observations": r.n_observations,
            "p_defend": r.p_defend,
        },
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
        "energy": {
            "soc_mj": snap.energy.soc_mj,
            "can_afford_aggressive": snap.energy.can_afford_aggressive,
            "energy_is_modeled": snap.energy.energy_is_modeled,
        },
        "data_quality": {
            "status": snap.data_quality.status,
            "quality_score": snap.data_quality.quality_score,
        },
        "gap_to_rival_s": (nl.gap_to_car_ahead_s if nl is not None else None),
        "position": (nl.position if nl is not None else None),
        "drs": (nl.drs_available if nl is not None else None),
        "our_speed_kmh": (nl.our_speed_kmh if nl is not None else None),
        "trace": [{"stage": s.stage, "detail": s.detail} for s in snap.trace],
    }


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
    race_key: str,
    driver: Optional[str] = None,
    rival: Optional[str] = None,
    lap: int,
    seed: int = 42,
    cache_dir: str = _DEFAULT_CACHE,
    full_snapshot: bool = False,
) -> dict:
    race = resolve_race(race_key)
    drv = (driver or race.default_driver).upper()
    riv = (rival or race.default_rival).upper()
    full = _load_race_laps(race, drv, riv, cache_dir)
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


def run_historical_replay(
    *,
    race_key: str,
    driver: Optional[str] = None,
    rival: Optional[str] = None,
    start_lap: int = 1,
    end_lap: Optional[int] = None,
    seed: int = 42,
    cache_dir: str = _DEFAULT_CACHE,
) -> dict:
    race = resolve_race(race_key)
    drv = (driver or race.default_driver).upper()
    riv = (rival or race.default_rival).upper()
    full = _load_race_laps(race, drv, riv, cache_dir)
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

    return {
        "race": _race_meta(race, drv, riv, len(available)),
        "driver": drv,
        "rival": riv,
        "total_laps": race.scheduled_laps or max(available),
        "laps_replayed": [start, end],
        "source": "fastf1_historical_replay",
        "model": "chronopace_2026",
        "provenance": {
            "REAL": [
                "2024 lap & sector timing, speed, throttle, brake, gear, RPM, DRS, distance "
                "(FastF1 / official F1 timing)",
                "actual observable driver performance lap by lap",
                "relative gap (cumulative lap-time difference through lap N)",
            ],
            "MODELED (MODEL_ASSUMPTION)": [
                "rival hidden energy state — INFERRED probabilistically from observable "
                "performance; the 2024 cars' real ERS SoC is not public and is never used",
                "our own SoC / energy budget — a 2026-model mean-reverting trajectory "
                "(energy_is_modeled=true on every lap)",
                "the ChronoPace 2026 decision model, Monte Carlo outcomes, opportunity "
                "probabilities, and regulatory-legality assumptions",
            ],
            "note": "2024 cars did NOT run under ChronoPace's 2026 energy rules. Real 2024 "
                    "telemetry flows through the 2026 decision model.",
        },
        "laps": laps_out,
    }


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
