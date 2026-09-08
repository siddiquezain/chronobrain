"""
field_state.py — telemetry-tick full-field reconstruction + tick-level selection.

Pure tests drive `select_over_ticks` / `_causal_hold` with synthetic candidates.
Real `FieldTimeline.build` is exercised cache-gated in test_tick_rival_replay.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.replay.field_state import _causal_hold, select_over_ticks
from app.replay.strategic_rival import RivalCandidate, RivalSelectorConfig


# ---------------------------------------------------------------------------
# _causal_hold — strict previous-value, never peeks forward
# ---------------------------------------------------------------------------
def test_causal_hold_is_previous_value_only():
    src_t = np.array([0.0, 1.0, 2.0, 3.0])
    src_y = np.array([10.0, 20.0, 30.0, 40.0])
    grid = np.array([-0.5, 0.0, 0.9, 1.0, 1.5, 3.0, 5.0])
    got = _causal_hold(src_t, src_y, grid)
    assert np.isnan(got[0])                     # before first sample
    assert list(got[1:]) == [10.0, 10.0, 20.0, 20.0, 40.0, 40.0]


def test_causal_hold_ignores_future_samples():
    src_t = np.array([0.0, 1.0, 2.0, 3.0])
    src_y = np.array([1.0, 2.0, 3.0, 4.0])
    base = _causal_hold(src_t, src_y, np.array([1.5]))
    src_y2 = src_y.copy(); src_y2[2:] = 999.0   # corrupt everything after t=1
    after = _causal_hold(src_t, src_y2, np.array([1.5]))
    assert base[0] == after[0] == 2.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _c(driver, gap, ahead=True, trend=0.0, pace=0.0, apart=1, status="racing"):
    return RivalCandidate(driver=driver, position=apart + 1, gap_s=gap, ahead=ahead,
                          gap_trend_s_per_lap=trend, pace_delta_s=pace,
                          positions_apart=apart, lap_status=status)


def _meta(n):
    return [(i * 0.25, 1 + i // 400) for i in range(n)]


# ---------------------------------------------------------------------------
# tiny score fluctuations must NOT cause oscillation (spec test 8)
# ---------------------------------------------------------------------------
def test_hysteresis_suppresses_chatter():
    cfg = RivalSelectorConfig()
    per_tick = []
    for i in range(200):
        # PIA and NOR essentially tied, wobbling by <0.02 each tick
        wob = 0.015 * np.sin(i)
        per_tick.append([
            _c("PIA", gap=0.50 - wob), _c("NOR", gap=0.50 + wob),
        ])
    tl = select_over_ticks(per_tick, _meta(200), cfg)
    switches = sum(1 for s in tl if s.switched)
    assert switches <= 1, f"chatter: {switches} switches on sub-margin noise"
    assert len({s.driver for s in tl}) == 1


# ---------------------------------------------------------------------------
# a genuine, sustained score gap DOES switch (spec test 9)
# ---------------------------------------------------------------------------
def test_hysteresis_allows_real_switch():
    cfg = RivalSelectorConfig()
    per_tick = []
    for i in range(120):
        if i < 40:
            per_tick.append([_c("PIA", gap=0.4), _c("NOR", gap=3.5)])
        else:
            per_tick.append([_c("PIA", gap=3.5), _c("NOR", gap=0.3)])  # NOR clearly closer now
    tl = select_over_ticks(per_tick, _meta(120), cfg)
    assert tl[0].driver == "PIA"
    assert tl[-1].driver == "NOR"
    assert sum(1 for s in tl if s.switched) == 1


def test_min_dwell_blocks_immediate_reswitch():
    cfg = RivalSelectorConfig(min_dwell_ticks=10)
    per_tick = [[_c("PIA", 0.3), _c("NOR", 3.0)]] * 5
    per_tick += [[_c("PIA", 3.0), _c("NOR", 0.3)]] * 5   # NOR better, but dwell not met
    per_tick += [[_c("PIA", 3.0), _c("NOR", 0.3)]] * 20  # now allowed
    tl = select_over_ticks(per_tick, _meta(30), cfg)
    assert tl[5].driver == "PIA" and tl[9].driver == "PIA"   # held through the dwell window
    assert tl[-1].driver == "NOR"


# ---------------------------------------------------------------------------
# AHEAD -> BEHIND within one tick stream (spec tests 4, 5, 6)
# ---------------------------------------------------------------------------
def test_role_flips_when_rival_goes_ahead_to_behind():
    cfg = RivalSelectorConfig()
    per_tick = [[_c("PIA", 0.3, ahead=True)]] * 20 + [[_c("PIA", 0.2, ahead=False)]] * 20
    tl = select_over_ticks(per_tick, _meta(40), cfg)
    assert tl[0].driver == "PIA" and tl[0].role == "ATTACK_TARGET"
    assert tl[-1].driver == "PIA" and tl[-1].role == "DEFENDING_THREAT"


def test_role_flips_when_rival_goes_behind_to_ahead():
    cfg = RivalSelectorConfig()
    per_tick = [[_c("PIA", 0.25, ahead=False)]] * 20 + [[_c("PIA", 0.35, ahead=True)]] * 20
    tl = select_over_ticks(per_tick, _meta(40), cfg)
    assert tl[0].role == "DEFENDING_THREAT"
    assert tl[-1].role == "ATTACK_TARGET"


# ---------------------------------------------------------------------------
# determinism + causality of the tick selection (spec tests 10, 11, 12)
# ---------------------------------------------------------------------------
def test_tick_selection_is_deterministic():
    cfg = RivalSelectorConfig()
    pt = [[_c("PIA", 0.4 + 0.1 * np.sin(i)), _c("NOR", 1.0)] for i in range(300)]
    a = select_over_ticks(pt, _meta(300), cfg)
    b = select_over_ticks(pt, _meta(300), cfg)
    assert a == b


def test_future_ticks_cannot_change_an_earlier_selection():
    cfg = RivalSelectorConfig()
    pt = [[_c("PIA", 0.4), _c("NOR", 1.2)] for _ in range(60)]
    before = select_over_ticks(pt[:31], _meta(31), cfg)[30]
    pt_corrupt = pt[:31] + [[_c("NOR", 0.01, ahead=False, apart=1)] for _ in range(29)]
    after = select_over_ticks(pt_corrupt, _meta(60), cfg)[30]
    assert before == after


def test_pitting_candidate_never_wins_at_tick_level():
    cfg = RivalSelectorConfig()
    pt = [[_c("PIA", 0.1, status="pit"), _c("NOR", 2.5, status="racing")]] * 50
    tl = select_over_ticks(pt, _meta(50), cfg)
    assert all(s.driver == "NOR" for s in tl)
