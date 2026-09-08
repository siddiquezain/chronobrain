"""
strategic_rival.py — dynamic, causal selection of the opponent that matters.

A real race is not a permanent two-car duel. The car that matters to the current
energy decision changes as positions swap, gaps open and close, and cars pit.
This module answers ONE question, deterministically, from data available at or
before the current lap:

    "Given our driver's race state and the currently observable field, which
     single opponent is most strategically relevant right now — and in what role?"

It does NOT decide attack/conserve, does NOT run Monte Carlo, does NOT estimate
our energy, and does NOT replace the Rival Energy Estimator. Its output selects
*whose* observable telemetry the estimator then infers a hidden energy state from.

CAUSAL: `build_strategic_rivals` only ever reads field entries with `lap <= N`
when selecting the rival for lap N. Corrupting a later lap cannot change an
earlier selection (see tests).

All weights and thresholds live in `RivalSelectorConfig` (MODEL_ASSUMPTION — a
deterministic scoring model, not measured and not learned).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

from rule_gate import GateConfig

Role = Literal[
    "ATTACK_TARGET",
    "DEFENDING_THREAT",
    "POSITION_BATTLE",
    "STRATEGICALLY_RELEVANT",
    "NONE",
]

_RACING = "racing"


# ---------------------------------------------------------------------------
# inputs / outputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LapFieldEntry:
    """One car's observable state on one lap — everything here is known at that
    lap's end (position, completed lap time, cumulative race time, pit status)."""

    driver: str
    lap: int
    position: Optional[int] = None
    cum_time_s: Optional[float] = None
    lap_time_s: Optional[float] = None
    lap_status: str = _RACING


@dataclass(frozen=True)
class RivalCandidate:
    """A candidate opponent, reduced to the signals the scorer needs (all causal)."""

    driver: str
    position: Optional[int]
    gap_s: Optional[float]                    # unsigned magnitude
    ahead: Optional[bool]                     # True = ahead of us, False = behind, None = unknown
    gap_trend_s_per_lap: Optional[float]      # negative = the gap is closing
    pace_delta_s: Optional[float]             # candidate rolling lap time - ours; negative = candidate faster
    positions_apart: Optional[int]
    lap_status: str = _RACING


@dataclass(frozen=True)
class StrategicRival:
    driver: str
    role: Role
    position: Optional[int]
    gap_s: Optional[float]
    ahead: Optional[bool]
    relevance_score: float
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def is_none(self) -> bool:
        return self.role == "NONE" or not self.driver


@dataclass(frozen=True)
class RivalSelectorConfig:
    """Deterministic scoring model. MODEL_ASSUMPTION — explicit weights, no magic
    numbers buried in the code. Weights sum to 1.0 for interpretability."""

    w_proximity: float = 0.40         # how close, by time gap
    w_adjacency: float = 0.25         # adjacent in running order
    w_gap_trend: float = 0.15         # closing vs falling away
    w_pace: float = 0.10             # on comparable pace (a real battle, not lapped traffic)
    w_directional: float = 0.10       # we actually know ahead/behind

    # gap (s) beyond which an opponent is not strategically relevant to THIS lap's
    # energy decision. Anchored above the overtake-proximity rule, not equal to it.
    strategic_gap_ceiling_s: float = 10.0
    # reference scales for normalising the trend / pace components
    gap_trend_ref_s_per_lap: float = 1.0
    pace_ref_s: float = 1.5
    # a non-racing (pit / out / invalid) candidate is damped, not deleted — it may
    # still matter strategically, but it is never an "immediate rival".
    non_racing_damping: float = 0.15
    # relevance below this -> role NONE (no opponent worth spending energy against)
    relevant_floor: float = 0.12
    # gap (s) at or under which the *directional* role (attack / defend) applies.
    # Defaults to the 2026 overtake-proximity rule so the two never drift apart.
    immediate_range_s: float = field(default_factory=lambda: GateConfig().overtake_detection_gap_threshold_s)

    # --- tick-level switching stability (hysteresis) -----------------------
    # A challenger must beat the *current* incumbent's live score by this margin
    # (in relevance units, 0..1) before the tracked rival switches. Prevents
    # PIA->NOR->PIA chatter from sub-noise score differences. MODEL_ASSUMPTION.
    switch_margin: float = 0.10
    # ...and the incumbent must have been held at least this many ticks before any
    # switch is allowed (~3 s at FastF1's ~4 Hz — a strategic rival should not flip
    # faster than a driver could react to it, and two cars at a near-identical gap
    # must not trade the title on reconstruction noise). MODEL_ASSUMPTION.
    min_dwell_ticks: int = 12

    def __post_init__(self) -> None:
        total = self.w_proximity + self.w_adjacency + self.w_gap_trend + self.w_pace + self.w_directional
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RivalSelectorConfig weights must sum to 1.0 (got {total})")
        if self.switch_margin < 0 or self.min_dwell_ticks < 0:
            raise ValueError("switch_margin and min_dwell_ticks must be >= 0")


_DEFAULT_CONFIG = RivalSelectorConfig()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _score_candidate(c: RivalCandidate, cfg: RivalSelectorConfig) -> Dict[str, float]:
    # proximity — closer = more relevant; nothing past the ceiling
    if c.gap_s is None:
        proximity = 0.0
    else:
        proximity = _clamp01(1.0 - c.gap_s / cfg.strategic_gap_ceiling_s)

    # adjacency in running order
    if c.positions_apart is None:
        adjacency = 0.0
    elif c.positions_apart <= 1:
        adjacency = 1.0
    elif c.positions_apart == 2:
        adjacency = 0.5
    elif c.positions_apart == 3:
        adjacency = 0.2
    else:
        adjacency = 0.0

    # gap trend — a closing car (negative trend) climbs; a car falling away drops
    if c.gap_trend_s_per_lap is None:
        gap_trend = 0.0
    else:
        gap_trend = _clamp01(-c.gap_trend_s_per_lap / cfg.gap_trend_ref_s_per_lap)

    # pace — a genuine battle (similar pace) matters more than lapped/backmarker traffic
    if c.pace_delta_s is None:
        pace = 0.5
    else:
        pace = _clamp01(1.0 - abs(c.pace_delta_s) / cfg.pace_ref_s)

    directional = 1.0 if c.ahead is not None else 0.3

    raw = (
        cfg.w_proximity * proximity
        + cfg.w_adjacency * adjacency
        + cfg.w_gap_trend * gap_trend
        + cfg.w_pace * pace
        + cfg.w_directional * directional
    )

    # multiplicative damping — a pitting car or a car past the strategic ceiling is
    # not an immediate rival regardless of the other components.
    damping = 1.0
    if c.lap_status != _RACING:
        damping *= cfg.non_racing_damping
    if c.gap_s is not None and c.gap_s > cfg.strategic_gap_ceiling_s:
        damping *= 0.1

    return {
        "proximity": round(proximity, 4),
        "adjacency": round(adjacency, 4),
        "gap_trend": round(gap_trend, 4),
        "pace": round(pace, 4),
        "directional": round(directional, 4),
        "damping": round(damping, 4),
        "relevance": round(_clamp01(raw) * damping, 4),
    }


def score_matrix(
    *,
    gap_s,
    ahead,
    gap_trend,
    pace_delta,
    positions_apart,
    status_code,
    cfg: Optional[RivalSelectorConfig] = None,
):
    """Vectorised twin of `_score_candidate`'s `relevance` output, over (T, D)
    arrays. Element-for-element identical to calling `_score_candidate` per cell
    (locked by test_strategic_rival.test_score_matrix_matches_scalar). `status_code`
    is 0 for 'racing', non-zero otherwise. NaN inputs mean "unknown" exactly as in
    the scalar path.

    This is NOT a re-derivation of the scoring model — it is the same formula,
    kept next to the scalar version so the tick-level path stays fast without a
    second source of truth.
    """
    import numpy as np

    cfg = cfg or _DEFAULT_CONFIG
    gap_s = np.asarray(gap_s, dtype=float)
    ahead = np.asarray(ahead, dtype=float)          # 1.0 ahead, 0.0 behind, NaN unknown
    gap_trend = np.asarray(gap_trend, dtype=float)
    pace_delta = np.asarray(pace_delta, dtype=float)
    apart = np.asarray(positions_apart, dtype=float)
    status_code = np.asarray(status_code)

    proximity = np.where(np.isnan(gap_s), 0.0, np.clip(1.0 - gap_s / cfg.strategic_gap_ceiling_s, 0.0, 1.0))

    adjacency = np.select(
        [np.isnan(apart), apart <= 1, apart == 2, apart == 3],
        [0.0, 1.0, 0.5, 0.2],
        default=0.0,
    )

    gap_tr = np.where(np.isnan(gap_trend), 0.0,
                      np.clip(-gap_trend / cfg.gap_trend_ref_s_per_lap, 0.0, 1.0))

    pace = np.where(np.isnan(pace_delta), 0.5,
                    np.clip(1.0 - np.abs(pace_delta) / cfg.pace_ref_s, 0.0, 1.0))

    directional = np.where(np.isnan(ahead), 0.3, 1.0)

    raw = (cfg.w_proximity * proximity + cfg.w_adjacency * adjacency
           + cfg.w_gap_trend * gap_tr + cfg.w_pace * pace + cfg.w_directional * directional)

    damping = np.ones_like(raw, dtype=float)
    damping = np.where(status_code != 0, damping * cfg.non_racing_damping, damping)
    damping = np.where(~np.isnan(gap_s) & (gap_s > cfg.strategic_gap_ceiling_s), damping * 0.1, damping)

    return np.round(np.clip(raw, 0.0, 1.0) * damping, 4)


def _role_for(c: RivalCandidate, relevance: float, cfg: RivalSelectorConfig) -> Role:
    if relevance < cfg.relevant_floor:
        return "NONE"
    in_range = c.gap_s is not None and c.gap_s <= cfg.immediate_range_s and c.lap_status == _RACING
    if in_range and c.ahead is True:
        return "ATTACK_TARGET"
    if in_range and c.ahead is False:
        return "DEFENDING_THREAT"
    if c.positions_apart is not None and c.positions_apart <= 1:
        return "POSITION_BATTLE"
    return "STRATEGICALLY_RELEVANT"


def select_strategic_rival(
    candidates: Sequence[RivalCandidate],
    *,
    config: Optional[RivalSelectorConfig] = None,
) -> StrategicRival:
    """Pick the single most strategically relevant opponent from `candidates`.

    Deterministic: ties break on (driver ahead first, then smaller gap, then
    lexicographic driver code) so the same field always yields the same choice.
    Returns a `StrategicRival` with role NONE when nothing clears the floor.
    """
    cfg = config or _DEFAULT_CONFIG
    if not candidates:
        return StrategicRival("", "NONE", None, None, None, 0.0, {})

    scored: List[tuple] = []
    for c in candidates:
        comp = _score_candidate(c, cfg)
        scored.append((comp["relevance"], c, comp))

    def _key(item):
        relevance, c, _ = item
        ahead_rank = 0 if c.ahead else 1
        gap = c.gap_s if c.gap_s is not None else 1e9
        return (-relevance, ahead_rank, gap, c.driver)

    scored.sort(key=_key)
    relevance, c, comp = scored[0]
    role = _role_for(c, relevance, cfg)
    return StrategicRival(
        driver=c.driver,
        role=role,
        position=c.position,
        gap_s=c.gap_s,
        ahead=c.ahead,
        relevance_score=relevance,
        components=comp,
    )


# ---------------------------------------------------------------------------
# causal orchestration over a whole race
# ---------------------------------------------------------------------------
def _rolling_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def build_candidate(
    field: Dict[str, Dict[int, LapFieldEntry]],
    our_driver: str,
    other: str,
    lap: int,
    *,
    trend_laps: int = 3,
    pace_laps: int = 5,
) -> Optional[RivalCandidate]:
    """Reduce one opponent's history *through lap N* to a `RivalCandidate`. Returns
    None when the opponent has no entry on lap N."""
    ours = field.get(our_driver, {})
    theirs = field.get(other, {})
    us_n, them_n = ours.get(lap), theirs.get(lap)
    if them_n is None or us_n is None:
        return None

    # signed gap magnitude from cumulative race time; direction from position when
    # known (more reliable across lapped traffic), else from the cumulative sign.
    gap_s: Optional[float] = None
    ahead: Optional[bool] = None
    if us_n.cum_time_s is not None and them_n.cum_time_s is not None:
        signed = them_n.cum_time_s - us_n.cum_time_s   # >0 => they took longer => they are behind
        gap_s = round(abs(signed), 3)
        ahead = signed < 0
    if us_n.position is not None and them_n.position is not None:
        ahead = them_n.position < us_n.position

    positions_apart = (
        abs(them_n.position - us_n.position)
        if (us_n.position is not None and them_n.position is not None)
        else None
    )

    # gap trend over the last `trend_laps` racing laps <= N
    trend: Optional[float] = None
    hist_laps = [l for l in range(lap - trend_laps, lap + 1) if l >= 1]
    abs_gaps: List[tuple] = []
    for l in hist_laps:
        u, t = ours.get(l), theirs.get(l)
        if u is None or t is None or u.cum_time_s is None or t.cum_time_s is None:
            continue
        abs_gaps.append((l, abs(t.cum_time_s - u.cum_time_s)))
    if len(abs_gaps) >= 2:
        (l0, g0), (l1, g1) = abs_gaps[0], abs_gaps[-1]
        if l1 != l0:
            trend = round((g1 - g0) / (l1 - l0), 4)

    # pace delta over the last `pace_laps` racing laps <= N
    our_lts = [
        ours[l].lap_time_s for l in range(max(1, lap - pace_laps + 1), lap + 1)
        if l in ours and ours[l].lap_status == _RACING and ours[l].lap_time_s is not None
    ]
    their_lts = [
        theirs[l].lap_time_s for l in range(max(1, lap - pace_laps + 1), lap + 1)
        if l in theirs and theirs[l].lap_status == _RACING and theirs[l].lap_time_s is not None
    ]
    our_pace, their_pace = _rolling_mean(our_lts), _rolling_mean(their_lts)
    pace_delta = (
        round(their_pace - our_pace, 4)
        if (our_pace is not None and their_pace is not None)
        else None
    )

    return RivalCandidate(
        driver=other,
        position=them_n.position,
        gap_s=gap_s,
        ahead=ahead,
        gap_trend_s_per_lap=trend,
        pace_delta_s=pace_delta,
        positions_apart=positions_apart,
        lap_status=them_n.lap_status,
    )


def build_strategic_rivals(
    field: Dict[str, Dict[int, LapFieldEntry]],
    our_driver: str,
    lap_numbers: Sequence[int],
    *,
    config: Optional[RivalSelectorConfig] = None,
    fallback_driver: Optional[str] = None,
) -> Dict[int, StrategicRival]:
    """
    For each lap in `lap_numbers`, select the strategic rival using ONLY field
    entries with `lap <= N`. Returns `{lap: StrategicRival}`.

    When no opponent clears the relevance floor the result still names the best
    candidate (or `fallback_driver` if given and present) with role NONE — so the
    Rival Energy Estimator always has an observation slot, and a race can still
    produce a valid decision.
    """
    cfg = config or _DEFAULT_CONFIG
    others = sorted(d for d in field if d != our_driver)
    out: Dict[int, StrategicRival] = {}

    for n in lap_numbers:
        candidates: List[RivalCandidate] = []
        for other in others:
            c = build_candidate(field, our_driver, other, n)
            if c is not None:
                candidates.append(c)
        sr = select_strategic_rival(candidates, config=cfg)

        if sr.is_none:
            seed = _fallback_candidate(candidates, fallback_driver)
            if seed is not None:
                comp = _score_candidate(seed, cfg)
                sr = StrategicRival(
                    driver=seed.driver, role="NONE", position=seed.position,
                    gap_s=seed.gap_s, ahead=seed.ahead,
                    relevance_score=comp["relevance"], components=comp,
                )
        out[n] = sr
    return out


def _fallback_candidate(
    candidates: Sequence[RivalCandidate], fallback_driver: Optional[str]
) -> Optional[RivalCandidate]:
    if fallback_driver:
        for c in candidates:
            if c.driver == fallback_driver:
                return c
    racing = [c for c in candidates if c.lap_status == _RACING and c.gap_s is not None]
    if racing:
        return min(racing, key=lambda c: c.gap_s)
    return candidates[0] if candidates else None
