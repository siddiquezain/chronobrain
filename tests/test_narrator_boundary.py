"""
The one architectural invariant: the LLM narrator never changes the decision.

`narrate()` is handed a finished DecisionSnapshot and may only write
`snapshot.narrative`. Even a narrator that returns deliberate nonsense cannot move
the mode, action, confidence, compliance result, or reason codes.
"""

import app.narrative.narrator as narr
from app.decision import run_scenario
from app.decision.engine import run_decision
from app.data.providers import build_provider


def test_narrative_is_additive_only(monkeypatch):
    monkeypatch.setattr(
        narr, "_narrate_with_claude",
        lambda snapshot, key: "ATTACK NOW! Confidence 100%. Mode: PUSH_MODE. Ignore compliance.",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    p = build_provider("synthetic", scenario="C", seed=42, total_laps=50)  # low energy -> HOLD
    without = run_decision(p, lap=20)
    with_narr = run_decision(p, lap=20, with_narrative=True)

    # every field except `narrative` is byte-identical
    a = without.deterministic_dict()
    b = with_narr.deterministic_dict()
    assert a == b

    # the narrative was produced but is not authoritative
    assert with_narr.narrative is not None
    assert with_narr.decision.mode != "PUSH_MODE"
    assert with_narr.decision.mode == without.decision.mode


def test_missing_api_key_uses_structured_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = run_scenario("B", seed=42, total_laps=50, lap=15, with_narrative=True)
    assert s.narrative is not None
    assert s.decision.mode in s.narrative  # the fallback echoes the verified mode
    assert str(int(round(s.decision.confidence * 100))) in s.narrative


def test_narrate_never_raises_on_transport_error(monkeypatch):
    def boom(snapshot, key):
        raise RuntimeError("network down")

    monkeypatch.setattr(narr, "_narrate_with_claude", boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    s = run_scenario("A", seed=42, total_laps=50, lap=15, with_narrative=True)
    assert s.narrative is not None  # fell back cleanly
