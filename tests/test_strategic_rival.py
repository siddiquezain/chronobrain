"""
Strategic-rival selector — the dynamic, causal "who matters right now" layer.

These tests drive the pure selector with hand-built field data (a UNIT-TEST
fixture, not passed off as real telemetry). Real-FastF1 behaviour is covered in
test_historical_replay.py / test_replay_fixes.py.
"""

from __future__ import annotations

import pytest

import numpy as np

from app.replay.strategic_rival import (
    LapFieldEntry,
    RivalCandidate,
    RivalSelectorConfig,
    _score_candidate,
    build_candidate,
    build_strategic_rivals,
    score_matrix,
    select_strategic_rival,
)


# ---------------------------------------------------------------------------
# field-table helpers
# ---------------------------------------------------------------------------
def _entry(driver, lap, pos, cum, lt=83.0, status="racing"):
    return LapFieldEntry(driver=driver, lap=lap, position=pos, cum_time_s=cum,
                         lap_time_s=lt, lap_status=status)


def _field(spec: dict) -> dict:
    """spec = {driver: [(lap, pos, cum_time, [lap_time], [status]), ...]}"""
    out: dict = {}
    for drv, rows in spec.items():
        out[drv] = {}
        for row in rows:
            lap, pos, cum = row[0], row[1], row[2]
            lt = row[3] if len(row) > 3 else 83.0
            st = row[4] if len(row) > 4 else "racing"
            out[drv][lap] = _entry(drv, lap, pos, cum, lt, st)
    return out


# ---------------------------------------------------------------------------
# TEST 1 — basic selection: closest strategically relevant car wins
# ---------------------------------------------------------------------------
def test_closest_relevant_car_is_selected():
    cands = [
        RivalCandidate("NOR", position=4, gap_s=0.7, ahead=True,
                       gap_trend_s_per_lap=-0.1, pace_delta_s=0.05, positions_apart=1),
        RivalCandidate("SAI", position=5, gap_s=1.4, ahead=True,
                       gap_trend_s_per_lap=0.0, pace_delta_s=0.1, positions_apart=2),
        RivalCandidate("PIA", position=6, gap_s=8.5, ahead=True,
                       gap_trend_s_per_lap=0.2, pace_delta_s=0.3, positions_apart=3),
    ]
    sr = select_strategic_rival(cands)
    assert sr.driver == "NOR"
    assert sr.relevance_score > 0.5


# ---------------------------------------------------------------------------
# TEST 2 — ahead vs behind -> attack target vs defending threat
# ---------------------------------------------------------------------------
def test_directional_roles():
    attack = select_strategic_rival([
        RivalCandidate("PIA", position=1, gap_s=0.8, ahead=True,
                       gap_trend_s_per_lap=-0.2, pace_delta_s=0.0, positions_apart=1),
    ])
    assert attack.role == "ATTACK_TARGET"

    defend = select_strategic_rival([
        RivalCandidate("PIA", position=2, gap_s=0.6, ahead=False,
                       gap_trend_s_per_lap=-0.3, pace_delta_s=0.0, positions_apart=1),
    ])
    assert defend.role == "DEFENDING_THREAT"


def test_position_battle_only_when_adjacent_AND_close():
    # adjacent in the order + a genuine fighting gap (< position_battle_max_gap_s) -> POSITION_BATTLE
    close = select_strategic_rival([
        RivalCandidate("RUS", position=3, gap_s=1.6, ahead=True,
                       gap_trend_s_per_lap=-0.05, pace_delta_s=0.1, positions_apart=1),
    ])
    assert close.role == "POSITION_BATTLE"

    # adjacent in the order BUT well clear (2.5 s) -> not a "battle", just relevant
    far = select_strategic_rival([
        RivalCandidate("RUS", position=3, gap_s=2.5, ahead=True,
                       gap_trend_s_per_lap=-0.05, pace_delta_s=0.1, positions_apart=1),
    ])
    assert far.role == "STRATEGICALLY_RELEVANT"


def test_role_classification_matrix():
    """Issue 10 — POSITION_BATTLE must mean an actual fight for position, not just
    'adjacent in the running order regardless of gap'."""
    from app.replay.strategic_rival import classify_role, RivalSelectorConfig
    cfg = RivalSelectorConfig()

    def role(gap, ahead, apart, rel=0.8, status="racing"):
        return classify_role(gap_s=gap, ahead=ahead, positions_apart=apart,
                             lap_status=status, relevance=rel, cfg=cfg)

    # very close, ahead / behind -> directional
    assert role(0.3, True, 1) == "ATTACK_TARGET"
    assert role(0.3, False, 1) == "DEFENDING_THREAT"
    assert role(1.0, True, 1) == "ATTACK_TARGET"        # exactly at the proximity rule
    # moderate gap, adjacent -> POSITION_BATTLE (still a fight)
    assert role(1.5, True, 1) == "POSITION_BATTLE"
    assert role(2.0, False, 1) == "POSITION_BATTLE"     # exactly at the battle cutoff
    # large gap, adjacent -> NOT a battle
    assert role(2.1, True, 1) == "STRATEGICALLY_RELEVANT"
    assert role(6.0, False, 1) == "STRATEGICALLY_RELEVANT"
    # two places apart -> never a battle even if close-ish
    assert role(1.2, True, 2) == "STRATEGICALLY_RELEVANT"
    # closing rapidly but far -> still not a battle (gap is what defines a battle)
    assert role(4.0, True, 1) == "STRATEGICALLY_RELEVANT"
    # below the floor -> NONE
    assert role(0.5, True, 1, rel=0.05) == "NONE"
    # in-range but pitting -> not a directional threat
    assert role(0.4, True, 1, status="pit") in ("POSITION_BATTLE", "STRATEGICALLY_RELEVANT", "NONE")


# ---------------------------------------------------------------------------
# TEST 3 — the strategic rival changes across a race
# ---------------------------------------------------------------------------
def test_strategic_rival_changes_across_the_race():
    field = _field({
        "LEC": [(1, 3, 100.0), (2, 3, 183.0), (3, 3, 266.0), (4, 3, 349.0)],
        # NOR: right with LEC early, then drops away
        "NOR": [(1, 4, 100.6), (2, 4, 183.5), (3, 4, 270.0), (4, 4, 361.0)],
        # SAI: comes from behind and closes onto LEC by lap 3-4
        "SAI": [(1, 5, 104.0), (2, 5, 186.0), (3, 5, 266.6), (4, 5, 349.4)],
        # PIA: miles ahead, never relevant
        "PIA": [(1, 1, 80.0), (2, 1, 163.0), (3, 1, 246.0), (4, 1, 329.0)],
    })
    rivals = build_strategic_rivals(field, "LEC", [1, 2, 3, 4])
    picks = [rivals[n].driver for n in (1, 2, 3, 4)]
    assert picks[0] == "NOR"                     # early: NOR alongside
    assert "SAI" in picks[2:]                    # later: SAI has closed in
    assert len(set(picks)) >= 2                  # the rival actually changed


# ---------------------------------------------------------------------------
# TEST 4 — overtake flips attack target into defending threat
# ---------------------------------------------------------------------------
def test_overtake_flips_role_from_attack_to_defend():
    field = _field({
        "LEC": [(1, 3, 100.0), (2, 3, 182.0), (3, 3, 264.0)],
        # PIA ahead on lap 1-2, LEC passes -> PIA behind on lap 3
        "PIA": [(1, 2, 99.6), (2, 2, 181.6), (3, 4, 264.5)],
    })
    r = build_strategic_rivals(field, "LEC", [1, 2, 3])
    assert r[1].driver == "PIA" and r[1].role == "ATTACK_TARGET"
    assert r[3].driver == "PIA" and r[3].role == "DEFENDING_THREAT"


# ---------------------------------------------------------------------------
# TEST 5 — a 15 s car does not beat a 0.7 s car
# ---------------------------------------------------------------------------
def test_large_gap_never_outranks_a_close_car():
    cands = [
        RivalCandidate("NOR", position=4, gap_s=0.7, ahead=True,
                       gap_trend_s_per_lap=0.0, pace_delta_s=0.0, positions_apart=1),
        RivalCandidate("PIA", position=6, gap_s=15.0, ahead=False,
                       gap_trend_s_per_lap=-3.0, pace_delta_s=-1.2, positions_apart=3),
    ]
    sr = select_strategic_rival(cands)
    assert sr.driver == "NOR"


# ---------------------------------------------------------------------------
# TEST 6 — a pitting car is not a fake immediate rival
# ---------------------------------------------------------------------------
def test_pitting_car_is_damped():
    cands = [
        RivalCandidate("PIA", position=2, gap_s=0.3, ahead=True,
                       gap_trend_s_per_lap=-5.0, pace_delta_s=8.0, positions_apart=1,
                       lap_status="pit"),
        RivalCandidate("SAI", position=4, gap_s=3.0, ahead=True,
                       gap_trend_s_per_lap=-0.1, pace_delta_s=0.1, positions_apart=2,
                       lap_status="racing"),
    ]
    sr = select_strategic_rival(cands)
    assert sr.driver == "SAI"
    # even if a pitting car is the only candidate, it is never an ATTACK/DEFEND role
    only_pit = select_strategic_rival([cands[0]])
    assert only_pit.role in ("STRATEGICALLY_RELEVANT", "POSITION_BATTLE", "NONE")


# ---------------------------------------------------------------------------
# TEST 7 — no hindsight: corrupting laps > N cannot change the pick at lap N
# ---------------------------------------------------------------------------
def test_selection_is_causal_future_laps_cannot_change_lap_n():
    base = _field({
        "LEC": [(l, 3, 100.0 + 83.0 * (l - 1)) for l in range(1, 11)],
        "NOR": [(l, 4, 100.6 + 83.1 * (l - 1)) for l in range(1, 11)],
        "SAI": [(l, 5, 104.0 + 83.4 * (l - 1)) for l in range(1, 11)],
    })
    pick_5_before = build_strategic_rivals(base, "LEC", [5])[5]

    corrupted = {
        drv: {
            l: (e if l <= 5 else LapFieldEntry(drv, l, position=1, cum_time_s=0.0,
                                               lap_time_s=1.0, lap_status="pit"))
            for l, e in laps.items()
        }
        for drv, laps in base.items()
    }
    pick_5_after = build_strategic_rivals(corrupted, "LEC", [5])[5]
    assert pick_5_before == pick_5_after


# ---------------------------------------------------------------------------
# TEST 8 — determinism
# ---------------------------------------------------------------------------
def test_deterministic():
    field = _field({
        "LEC": [(l, 2, 90.0 + 83.0 * (l - 1)) for l in range(1, 21)],
        "PIA": [(l, 1, 89.0 + 83.0 * (l - 1)) for l in range(1, 21)],
        "NOR": [(l, 3, 91.0 + 83.2 * (l - 1)) for l in range(1, 21)],
    })
    a = build_strategic_rivals(field, "LEC", list(range(1, 21)))
    b = build_strategic_rivals(field, "LEC", list(range(1, 21)))
    assert a == b


# ---------------------------------------------------------------------------
# edge cases (spec §15)
# ---------------------------------------------------------------------------
def test_no_candidates_returns_none_role():
    sr = select_strategic_rival([])
    assert sr.is_none and sr.driver == ""


def test_missing_position_and_gap_do_not_crash():
    sr = select_strategic_rival([
        RivalCandidate("NOR", position=None, gap_s=None, ahead=None,
                       gap_trend_s_per_lap=None, pace_delta_s=None, positions_apart=None),
    ])
    assert sr.driver == "NOR"
    assert sr.role in ("STRATEGICALLY_RELEVANT", "NONE")


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RivalSelectorConfig(w_proximity=0.9)


def test_build_candidate_returns_none_when_opponent_absent_on_lap():
    field = _field({"LEC": [(1, 2, 90.0)], "NOR": [(2, 3, 175.0)]})
    assert build_candidate(field, "LEC", "NOR", 1) is None


# ---------------------------------------------------------------------------
# score_matrix is a LOCKED vectorised twin of _score_candidate (same formula)
# ---------------------------------------------------------------------------
def test_score_matrix_matches_scalar():
    rng = np.random.default_rng(7)
    cfg = RivalSelectorConfig()
    N = 4000
    gap = np.where(rng.random(N) < 0.1, np.nan, rng.uniform(0, 15, N))
    ahead = np.where(rng.random(N) < 0.15, np.nan, (rng.random(N) > 0.5).astype(float))
    trend = np.where(rng.random(N) < 0.2, np.nan, rng.uniform(-3, 3, N))
    pace = np.where(rng.random(N) < 0.2, np.nan, rng.uniform(-4, 4, N))
    apart = np.where(rng.random(N) < 0.1, np.nan, rng.integers(1, 9, N).astype(float))
    status = rng.integers(0, 4, N)
    mat = score_matrix(gap_s=gap, ahead=ahead, gap_trend=trend, pace_delta=pace,
                       positions_apart=apart, status_code=status, cfg=cfg)
    worst = 0.0
    for i in range(N):
        c = RivalCandidate(
            driver="X", position=1,
            gap_s=None if np.isnan(gap[i]) else float(gap[i]),
            ahead=None if np.isnan(ahead[i]) else bool(ahead[i] > 0.5),
            gap_trend_s_per_lap=None if np.isnan(trend[i]) else float(trend[i]),
            pace_delta_s=None if np.isnan(pace[i]) else float(pace[i]),
            positions_apart=None if np.isnan(apart[i]) else int(apart[i]),
            lap_status="racing" if status[i] == 0 else "pit",
        )
        worst = max(worst, abs(_score_candidate(c, cfg)["relevance"] - mat[i]))
    # both sides round the relevance to 4 dp; a disagreement of one unit in the
    # last place is float rounding at an exact .5 boundary, not a formula drift.
    assert worst <= 2e-4, f"vectorised score drifts from scalar by {worst}"
