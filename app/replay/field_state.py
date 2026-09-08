"""
field_state.py — causal, synchronised full-field state at telemetry-tick cadence.

The lap-level strategic-rival selector (``strategic_rival.py``) answers "who matters
this lap". A real fight changes between corners. This module reconstructs, from an
already-loaded FastF1 session, **every driver's race state at every telemetry
timestamp our own car produced** (FastF1's real cadence — roughly 4 Hz / ~330
samples per lap, NOT 128 Hz and NOT 1 ms), and turns each tick into the same
``RivalCandidate`` objects the existing scorer already consumes.

    session (all drivers' car_data, already parsed)
        -> FieldTimeline.build(session, our_driver)          [cached per session]
        -> (T, D) matrices: progress / speed / running-position / lap-status
        -> vectorised relevance score (locked to strategic_rival._score_candidate)
        -> tick-level pick + hysteresis   -> TickSelection[]
        -> per-lap dominant rival + change count   (the lap-level SUMMARY, spec §3)

CAUSALITY: every per-driver channel is sampled onto our tick grid with a strict
*previous-value hold* (`_causal_hold`) — it never interpolates toward a future
sample. Running position is the real per-lap timing position, held forward. So
corrupting any sample with SessionTime > t cannot change the state, the
candidates, or the selection at t (see tests).

DIRECTION NOTE: ahead/behind comes from the official per-lap running position
(held forward), not from raw GPS/integrated distance — public data does not
resolve a 0.2 s side-by-side reliably. An overtake therefore flips the role at
the lap the position table updates (≤ ~1 lap latency); a genuine sub-second
side-by-side reads as POSITION_BATTLE, which is what it is.

Nothing here estimates energy, runs Monte Carlo, or makes a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.replay.strategic_rival import (
    RivalCandidate,
    RivalSelectorConfig,
    score_matrix,
    select_strategic_rival,
)

_RACING = "racing"
_STATUS_CODE = {"racing": 0, "pit": 1, "out_lap": 2, "invalid_for_energy_inference": 3}
_CODE_STATUS = {v: k for k, v in _STATUS_CODE.items()}
_MIN_CLOSING_SPEED_MS = 20.0
_DEFAULT_REF_LAPTIME_S = 90.0
# Gap magnitude between nearby cars, reconstructed from speed-integrated distance,
# carries ~+/-0.3 s of noise on a ~0.4 s signal. Smooth it hard and take the trend
# as a robust slope over a multi-second window so noise can't manufacture a
# "closing fast" score spike. MODEL_ASSUMPTION.
_GAP_EMA_WINDOW_S = 3.5
_TREND_WINDOW_S = 6.0
# Hard cap on the tick-level closing-rate signal. Chosen so its worst-case score
# contribution (w_gap_trend * clamp) stays *below* `switch_margin` — a noisy
# closing rate can never on its own flip the tracked rival; only proximity +
# adjacency (both anchored to real timing) can. MODEL_ASSUMPTION.
_TREND_CLAMP_S_PER_LAP = 0.5
_EARLY_LAPS_FOR_CONSTANTS = 6    # track length / ref lap-time use only laps 2..6


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TickSelection:
    t: float
    lap: int
    driver: str
    role: str
    ahead: Optional[bool]
    gap_s: Optional[float]
    relevance_score: float
    position: Optional[int] = None
    raw_leader: str = ""
    switched: bool = False


@dataclass(frozen=True)
class LapRivalSummary:
    """Lap-level SUMMARY derived from the tick selections in that lap (spec §3)."""

    lap: int
    dominant_driver: str
    dominant_role: str
    dominant_ahead: Optional[bool]
    dominant_gap_s: Optional[float]
    dominant_position: Optional[int]
    dominant_relevance: float
    first_driver: str
    last_driver: str
    n_changes: int
    ticks: int
    share_by_driver: Dict[str, float]
    change_events: List[dict]


# ---------------------------------------------------------------------------
# causal sampling helpers
# ---------------------------------------------------------------------------
def _causal_hold(src_t: np.ndarray, src_y: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """Value of `src_y` at the most recent `src_t` <= each `grid_t` (previous-hold).
    Strictly causal. NaN before the first source sample."""
    if len(src_t) == 0:
        return np.full(len(grid_t), np.nan)
    idx = np.searchsorted(src_t, grid_t, side="right") - 1
    return np.where(idx >= 0, src_y[np.clip(idx, 0, len(src_y) - 1)], np.nan)


def _causal_ema(t: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    """Irregular-sample exponential moving average — causal (only past samples)."""
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    dt = np.diff(t, prepend=t[0])
    alpha = 1.0 - np.exp(-dt / max(window_s, 1e-6))
    acc = y[0] if np.isfinite(y[0]) else 0.0
    for i in range(len(y)):
        if np.isfinite(y[i]):
            acc = acc + alpha[i] * (y[i] - acc)
        out[i] = acc
    return out


def _hash_seed(driver: str) -> int:
    """Deterministic, process-independent per-driver seed offset."""
    import zlib
    return int(zlib.adler32(driver.encode("utf-8")) % 100_000)


# ---------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------
@dataclass
class FieldTimeline:
    our_driver: str
    drivers: List[str]                 # opponents present in the matrices, sorted
    ticks: np.ndarray                  # (T,) SessionTime seconds — our car's grid
    lap_of_tick: np.ndarray            # (T,) int — the lap OUR car is on
    our_lap_status: np.ndarray         # (T,) object
    track_length_m: float
    ref_laptime_s: float
    # (T, D) matrices, columns aligned to `drivers`
    _gap_s: np.ndarray
    _ahead: np.ndarray                 # float: 1.0 ahead / 0.0 behind / NaN unknown
    _trend: np.ndarray                 # gap trend, s per lap (neg = closing)
    _apart: np.ndarray                 # positions apart (int)
    _pace: np.ndarray                  # pace delta s (nan where unknown)
    _status: np.ndarray               # int codes
    _present: np.ndarray               # bool — opponent on track at this tick
    _pos_at_tick: np.ndarray           # (T, D) running position
    _our_pos_at_tick: np.ndarray       # (T,)
    _laptimes: Dict[str, Dict[int, float]]

    # -- construction ---------------------------------------------------
    @classmethod
    def build(cls, session, our_driver: str, *, scheduled_laps: Optional[int] = None) -> "FieldTimeline":
        from app.data.fastf1_service import (
            _field_drivers,
            _lap_status_by_lap,
            _laptimes_by_lap,
            _positions_by_lap,
        )

        our_driver = our_driver.upper()
        all_drivers = _field_drivers(session)
        if our_driver not in all_drivers:
            raise ValueError(f"{our_driver!r} not in this session's field")
        num_by_abbr = _driver_numbers(session)

        our_cd = _car_data_arrays(session, num_by_abbr.get(our_driver))
        our_ldf = _laps_frame(session, our_driver)
        if our_cd is None or our_ldf is None or len(our_ldf["start"]) == 0:
            raise ValueError(f"no telemetry / laps for {our_driver!r}")
        our_t, our_spd = our_cd

        start_t = float(our_ldf["start"][0])
        end_t = float(our_ldf["start"][-1] + (our_ldf["laptime"][-1] or _DEFAULT_REF_LAPTIME_S))
        ticks = our_t[(our_t >= start_t) & (our_t <= end_t)]
        if len(ticks) < 4:
            raise ValueError("degenerate tick grid")
        T = len(ticks)

        our_cumdist = np.cumsum(our_spd * np.diff(our_t, prepend=our_t[0]))

        our_status_by_lap = _lap_status_by_lap(session, our_driver)
        our_pos_by_lap = _positions_by_lap(session, our_driver)
        lap_of_our = np.searchsorted(our_ldf["start"], ticks, side="right")
        our_lap_status = np.array([our_status_by_lap.get(int(l), _RACING) for l in lap_of_our], dtype=object)
        our_pos_tick = np.array([our_pos_by_lap.get(int(l)) or 0 for l in lap_of_our], dtype=float)

        # Track length and a reference lap time are treated as circuit constants
        # (public before the race). Both are estimated from the FIRST few racing
        # laps only, so nothing after ~lap 6 — corrupt or not — can shift them,
        # keeping build() causal for every mid-race hindsight test.
        track_len = _median_lap_distance(our_t, our_cumdist, our_ldf["start"][:_EARLY_LAPS_FOR_CONSTANTS + 1])
        ref_lap = _early_ref_laptime(our_ldf, our_status_by_lap) or _DEFAULT_REF_LAPTIME_S

        our_prog = _progress_on_grid(our_t, our_cumdist, our_ldf["start"], ticks, track_len)
        our_spd_grid = _causal_hold(our_t, our_spd, ticks)

        cols: List[str] = []
        gap_l, ahead_l, apart_l, pace_l, status_l, present_l, pos_l = [], [], [], [], [], [], []
        laptimes: Dict[str, Dict[int, float]] = {}

        for d in sorted(x for x in all_drivers if x != our_driver):
            cd = _car_data_arrays(session, num_by_abbr.get(d))
            ldf = _laps_frame(session, d)
            if cd is None or ldf is None or len(ldf["start"]) == 0:
                continue
            dt_t, dt_spd = cd
            d_cumdist = np.cumsum(dt_spd * np.diff(dt_t, prepend=dt_t[0]))
            d_prog = _progress_on_grid(dt_t, d_cumdist, ldf["start"], ticks, track_len)
            d_spd = _causal_hold(dt_t, dt_spd, ticks)
            d_lap = np.searchsorted(ldf["start"], ticks, side="right")

            present = np.isfinite(d_prog) & (ticks <= (dt_t[-1] + 5.0)) & (d_lap >= 1)
            dprog = d_prog - our_prog
            trailing_spd = np.where(dprog > 0, our_spd_grid, d_spd)
            trailing_spd = np.where(np.isfinite(trailing_spd) & (trailing_spd > _MIN_CLOSING_SPEED_MS),
                                    trailing_spd, track_len / ref_lap)
            gap_raw = np.abs(dprog) * track_len / trailing_spd
            gap = _causal_ema(ticks, np.where(present, gap_raw, np.nan), _GAP_EMA_WINDOW_S)

            d_pos_by_lap = _positions_by_lap(session, d)
            d_status_by_lap = _lap_status_by_lap(session, d)
            d_pos_tick = np.array([d_pos_by_lap.get(int(l)) or 0 for l in d_lap], dtype=float)
            # direction from official running position; fall back to progress sign;
            # NaN only when we truly know neither. (float: 1.0 ahead / 0.0 behind)
            have_pos = (d_pos_tick > 0) & (our_pos_tick > 0)
            ahead = np.where(
                have_pos, (d_pos_tick < our_pos_tick).astype(float),
                np.where(np.isfinite(dprog), (dprog > 0).astype(float), np.nan),
            )
            apart = np.where(have_pos, np.abs(d_pos_tick - our_pos_tick), np.nan)
            d_status_codes = np.array(
                [_STATUS_CODE.get(d_status_by_lap.get(int(l), _RACING), 0) if l >= 1 else 0 for l in d_lap]
            )
            pace = _pace_delta_series(laptimes.setdefault(our_driver, _laptimes_by_lap(session, our_driver)),
                                      laptimes.setdefault(d, _laptimes_by_lap(session, d)),
                                      lap_of_our)

            cols.append(d)
            gap_l.append(gap); ahead_l.append(ahead); apart_l.append(apart)
            pace_l.append(pace); status_l.append(d_status_codes); present_l.append(present)
            pos_l.append(d_pos_tick)

        if not cols:
            raise ValueError("no opponents with telemetry in this session")

        stack = lambda ls: np.column_stack(ls)
        gap_m = stack(gap_l)
        # positions_apart from running-order rank where official positions missing
        apart_m = stack(apart_l)
        pos_m = stack(pos_l)
        _fill_apart_from_rank(apart_m, pos_m, our_pos_tick, stack(present_l))
        trend_m = _gap_trend_matrix(ticks, gap_m, ref_lap)

        return cls(
            our_driver=our_driver, drivers=cols, ticks=ticks, lap_of_tick=lap_of_our,
            our_lap_status=our_lap_status, track_length_m=track_len, ref_laptime_s=ref_lap,
            _gap_s=gap_m, _ahead=stack(ahead_l), _trend=trend_m, _apart=apart_m,
            _pace=stack(pace_l), _status=stack(status_l), _present=stack(present_l),
            _pos_at_tick=pos_m, _our_pos_at_tick=our_pos_tick, _laptimes=laptimes,
        )

    # -- per-tick candidate view (introspection / tests) --------------
    def candidates_at(self, i: int) -> List[RivalCandidate]:
        out: List[RivalCandidate] = []
        for j, d in enumerate(self.drivers):
            if not self._present[i, j]:
                continue
            apart = self._apart[i, j]
            av = self._ahead[i, j]
            out.append(RivalCandidate(
                driver=d,
                position=int(self._pos_at_tick[i, j]) or None,
                gap_s=round(float(self._gap_s[i, j]), 3) if np.isfinite(self._gap_s[i, j]) else None,
                ahead=(None if np.isnan(av) else bool(av > 0.5)),
                gap_trend_s_per_lap=round(float(self._trend[i, j]), 4) if np.isfinite(self._trend[i, j]) else None,
                pace_delta_s=round(float(self._pace[i, j]), 4) if np.isfinite(self._pace[i, j]) else None,
                positions_apart=int(apart) if np.isfinite(apart) else None,
                lap_status=_CODE_STATUS.get(int(self._status[i, j]), _RACING),
            ))
        return out

    # -- vectorised relevance over the whole race --------------------
    def relevance_matrix(self, cfg: Optional[RivalSelectorConfig] = None) -> np.ndarray:
        cfg = cfg or RivalSelectorConfig()
        rel = score_matrix(
            gap_s=self._gap_s, ahead=self._ahead, gap_trend=self._trend,
            pace_delta=self._pace, positions_apart=self._apart, status_code=self._status,
            cfg=cfg,
        )
        return np.where(self._present, rel, -1.0)

    # -- the tick-level selection stream ----------------------------
    def selection_timeline(self, config: Optional[RivalSelectorConfig] = None) -> List[TickSelection]:
        cfg = config or RivalSelectorConfig()
        rel = self.relevance_matrix(cfg)
        return _hysteresis_over_matrix(
            rel=rel, drivers=self.drivers, ticks=self.ticks, lap_of=self.lap_of_tick,
            gap_s=self._gap_s, ahead=self._ahead, apart=self._apart, status=self._status,
            pos=self._pos_at_tick,
            immediate_range_s=cfg.immediate_range_s, relevant_floor=cfg.relevant_floor,
            switch_margin=cfg.switch_margin, min_dwell_ticks=cfg.min_dwell_ticks,
        )

    # -- lap-level summary (spec §3) -------------------------------
    def lap_summaries(
        self, config: Optional[RivalSelectorConfig] = None,
        timeline: Optional[List[TickSelection]] = None,
    ) -> Dict[int, LapRivalSummary]:
        tl = timeline if timeline is not None else self.selection_timeline(config)
        by_lap: Dict[int, List[TickSelection]] = {}
        for s in tl:
            by_lap.setdefault(s.lap, []).append(s)
        out: Dict[int, LapRivalSummary] = {}
        for lap, sels in by_lap.items():
            n = len(sels)
            share: Dict[str, float] = {}
            for s in sels:
                share[s.driver] = share.get(s.driver, 0.0) + 1.0 / n
            dominant = max(share, key=share.get)
            rep = [s for s in sels if s.driver == dominant][-1]
            changes = [
                {"t": round(sels[i].t, 3),
                 "lap_fraction": round((sels[i].t - sels[0].t) / max(sels[-1].t - sels[0].t, 1e-6), 3),
                 "from": sels[i - 1].driver, "to": sels[i].driver, "role": sels[i].role}
                for i in range(1, n) if sels[i].driver != sels[i - 1].driver
            ]
            out[lap] = LapRivalSummary(
                lap=lap, dominant_driver=dominant, dominant_role=rep.role,
                dominant_ahead=rep.ahead, dominant_gap_s=rep.gap_s,
                dominant_position=rep.position,
                dominant_relevance=rep.relevance_score,
                first_driver=sels[0].driver, last_driver=sels[-1].driver,
                n_changes=len(changes), ticks=n,
                share_by_driver={k: round(v, 4) for k, v in sorted(share.items(), key=lambda kv: -kv[1])},
                change_events=changes,
            )
        return out

    def estimator_seed_offset(self, driver: str) -> int:
        return _hash_seed(driver)


# ---------------------------------------------------------------------------
# hysteresis over the precomputed relevance matrix (fast; deterministic; causal)
# ---------------------------------------------------------------------------
def _hysteresis_over_matrix(
    *, rel: np.ndarray, drivers: List[str], ticks: np.ndarray, lap_of: np.ndarray,
    gap_s: np.ndarray, ahead: np.ndarray, apart: np.ndarray, status: np.ndarray,
    pos: np.ndarray,
    immediate_range_s: float, relevant_floor: float,
    switch_margin: float, min_dwell_ticks: int,
) -> List[TickSelection]:
    T, D = rel.shape
    best_j = np.argmax(rel, axis=1)
    best_rel = rel[np.arange(T), best_j]
    out: List[TickSelection] = []
    inc = -1
    dwell = 0
    for i in range(T):
        bj = int(best_j[i])
        pick = bj
        if inc >= 0 and rel[i, inc] >= 0.0:
            inc_rel = rel[i, inc]
            if bj == inc:
                pick = inc
            elif dwell < min_dwell_ticks:
                pick = inc
            elif best_rel[i] - inc_rel >= switch_margin:
                pick = bj
            else:
                pick = inc
        switched = inc >= 0 and pick != inc
        if pick != inc:
            inc, dwell = pick, 0
        else:
            dwell += 1

        r = float(rel[i, pick]) if rel[i, pick] >= 0 else 0.0
        av = ahead[i, pick]
        is_ahead = None if np.isnan(av) else bool(av > 0.5)
        if r < relevant_floor:
            role = "NONE"
        else:
            in_range = (np.isfinite(gap_s[i, pick]) and gap_s[i, pick] <= immediate_range_s
                        and int(status[i, pick]) == 0)
            if in_range and is_ahead is True:
                role = "ATTACK_TARGET"
            elif in_range and is_ahead is False:
                role = "DEFENDING_THREAT"
            elif np.isfinite(apart[i, pick]) and apart[i, pick] <= 1:
                role = "POSITION_BATTLE"
            else:
                role = "STRATEGICALLY_RELEVANT"
        pp = int(pos[i, pick]) if np.isfinite(pos[i, pick]) and pos[i, pick] > 0 else None
        out.append(TickSelection(
            t=float(ticks[i]), lap=int(lap_of[i]), driver=drivers[pick], role=role,
            ahead=is_ahead, gap_s=round(float(gap_s[i, pick]), 3) if np.isfinite(gap_s[i, pick]) else None,
            relevance_score=round(r, 4), position=pp, raw_leader=drivers[bj], switched=switched,
        ))
    return out


# ---------------------------------------------------------------------------
# pure tick-level selection over synthetic candidate lists (tests use this)
# ---------------------------------------------------------------------------
def select_over_ticks(
    per_tick_candidates: Sequence[Sequence[RivalCandidate]],
    tick_meta: Sequence[Tuple[float, int]],
    config: Optional[RivalSelectorConfig] = None,
) -> List[TickSelection]:
    """Run the UNCHANGED scorer at every tick, then the switching policy. Causal by
    construction — tick i only reads its own candidates plus the running state."""
    from app.replay.strategic_rival import _role_for, _score_candidate
    cfg = config or RivalSelectorConfig()
    out: List[TickSelection] = []
    incumbent: Optional[str] = None
    dwell = 0
    for i, cands in enumerate(per_tick_candidates):
        t, lap = tick_meta[i]
        by_driver = {c.driver: c for c in cands}
        raw = select_strategic_rival(cands, config=cfg)
        pick = raw.driver
        if incumbent is not None and incumbent in by_driver:
            inc_rel = _score_candidate(by_driver[incumbent], cfg)["relevance"]
            if raw.driver == incumbent:
                pick = incumbent
            elif dwell < cfg.min_dwell_ticks:
                pick = incumbent
            elif raw.relevance_score - inc_rel >= cfg.switch_margin:
                pick = raw.driver
            else:
                pick = incumbent
        switched = incumbent is not None and pick != incumbent
        if pick != incumbent:
            incumbent, dwell = pick, 0
        else:
            dwell += 1
        c = by_driver.get(pick)
        if c is not None:
            rel = _score_candidate(c, cfg)["relevance"]
            out.append(TickSelection(
                t=t, lap=lap, driver=pick, role=_role_for(c, rel, cfg),
                ahead=c.ahead, gap_s=c.gap_s, relevance_score=round(rel, 4),
                position=c.position, raw_leader=raw.driver, switched=switched,
            ))
        else:
            out.append(TickSelection(t=t, lap=lap, driver=pick or "", role="NONE",
                                     ahead=None, gap_s=None, relevance_score=0.0,
                                     raw_leader=raw.driver, switched=switched))
    return out


# ---------------------------------------------------------------------------
# FastF1 extraction helpers (session already loaded — cheap array access)
# ---------------------------------------------------------------------------
def _driver_numbers(session) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for num in getattr(session, "drivers", []) or []:
        try:
            out[str(session.get_driver(num)["Abbreviation"])] = str(num)
        except Exception:
            continue
    return out


def _car_data_arrays(session, num: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if num is None:
        return None
    try:
        cd = session.car_data[num]
        t = cd["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
        spd = cd["Speed"].to_numpy(dtype=float) / 3.6
        keep = np.isfinite(t) & np.isfinite(spd)
        t, spd = t[keep], spd[keep]
        order = np.argsort(t, kind="stable")
        return t[order], spd[order]
    except Exception:
        return None


def _laps_frame(session, driver: str) -> Optional[dict]:
    try:
        d = session.laps.pick_drivers(driver) if hasattr(session.laps, "pick_drivers") else session.laps.pick_driver(driver)
        starts, nums, times = [], [], []
        for _, lap in d.iterlaps():
            st = lap["LapStartTime"]
            if st is None or st != st:
                continue
            starts.append(float(st.total_seconds()))
            nums.append(int(lap["LapNumber"]))
            lt = lap["LapTime"]
            times.append(float(lt.total_seconds()) if (lt is not None and lt == lt) else None)
        if not starts:
            return None
        order = np.argsort(starts, kind="stable")
        return {"start": np.array(starts)[order], "num": np.array(nums)[order],
                "laptime": [times[o] for o in order]}
    except Exception:
        return None


def _progress_on_grid(src_t, cumdist, lap_starts, grid_t, track_len) -> np.ndarray:
    d_lap = np.searchsorted(lap_starts, grid_t, side="right")
    lap_start_dist = _causal_hold(src_t, cumdist, lap_starts)
    idx = np.clip(d_lap - 1, 0, len(lap_start_dist) - 1)
    cur_start = np.where((d_lap - 1 >= 0) & (d_lap - 1 < len(lap_start_dist)), lap_start_dist[idx], np.nan)
    dist_now = _causal_hold(src_t, cumdist, grid_t)
    frac = np.clip((dist_now - cur_start) / max(track_len, 1.0), 0.0, 1.05)
    prog = (d_lap - 1) + frac
    return np.where(d_lap >= 1, prog, np.nan)


def _pace_delta_series(our_lt: Dict[int, float], their_lt: Dict[int, float],
                       our_lap_of_tick: np.ndarray, k: int = 5) -> np.ndarray:
    per_lap: Dict[int, float] = {}
    for lap in range(1, int(our_lap_of_tick.max()) + 2):
        o = [our_lt[l] for l in range(max(1, lap - k), lap) if our_lt.get(l)]
        t = [their_lt[l] for l in range(max(1, lap - k), lap) if their_lt.get(l)]
        per_lap[lap] = (np.median(t) - np.median(o)) if (o and t) else np.nan
    return np.array([per_lap.get(int(l), np.nan) for l in our_lap_of_tick])


def _gap_trend_matrix(ticks: np.ndarray, gap_m: np.ndarray, ref_lap: float) -> np.ndarray:
    """Causal robust slope of each gap over the last `_TREND_WINDOW_S`, expressed
    in the scorer's s-per-lap convention (negative = gap closing)."""
    T = len(ticks)
    dt = float(np.median(np.diff(ticks))) if T > 1 else 0.25
    k = max(2, int(round(_TREND_WINDOW_S / max(dt, 1e-3))))
    out = np.zeros_like(gap_m, dtype=float)
    x = np.arange(k, dtype=float) * dt
    x = x - x.mean()
    denom = float(np.sum(x * x)) or 1.0
    for i in range(T):
        a = max(0, i - k + 1)
        w = gap_m[a:i + 1]
        if len(w) < 2:
            continue
        xx = x[-len(w):]
        xx = xx - xx.mean()
        d = float(np.sum(xx * xx)) or 1.0
        slope = (xx[:, None] * (w - np.nanmean(w, axis=0))).sum(axis=0) / d   # s per s, per column
        out[i] = slope * ref_lap
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(out, -_TREND_CLAMP_S_PER_LAP, _TREND_CLAMP_S_PER_LAP)


def _fill_apart_from_rank(apart_m, pos_m, our_pos_tick, present_m) -> None:
    T, D = apart_m.shape
    for i in range(T):
        if np.all(np.isfinite(apart_m[i])):
            continue
        vals = [(pos_m[i, j], j) for j in range(D) if present_m[i, j]]
        # rank present cars by their (possibly zero) position proxy — stable
        vals.sort()
        rank = {j: r for r, (_, j) in enumerate(vals)}
        our_rank = len([v for v, _ in vals if v and our_pos_tick[i] and v < our_pos_tick[i]])
        for j in range(D):
            if not np.isfinite(apart_m[i, j]) and j in rank:
                apart_m[i, j] = abs(rank[j] - our_rank)


def _median_lap_distance(t, cumdist, lap_starts) -> float:
    dists = []
    for a, b in zip(lap_starts[:-1], lap_starts[1:]):
        ia = int(np.searchsorted(t, a, side="right") - 1)
        ib = int(np.searchsorted(t, b, side="right") - 1)
        if 0 <= ia < ib < len(cumdist):
            dists.append(cumdist[ib] - cumdist[ia])
    dists = [x for x in dists if 2000.0 < x < 12000.0]
    return float(np.median(dists)) if dists else 5000.0


def _early_ref_laptime(ldf: dict, status_by_lap: Dict[int, str]) -> Optional[float]:
    """Median racing lap time over laps 2.._EARLY_LAPS_FOR_CONSTANTS only — a
    circuit-scale constant that no later (possibly corrupted) lap can move."""
    good = [lt for n, lt in zip(ldf["num"], ldf["laptime"])
            if lt is not None and 2 <= int(n) <= _EARLY_LAPS_FOR_CONSTANTS
            and status_by_lap.get(int(n), _RACING) == _RACING]
    return float(np.median(good)) if good else None
