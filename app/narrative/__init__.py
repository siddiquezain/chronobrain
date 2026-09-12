"""
app.narrative — Stage 4. The ONLY place a language model is allowed to run.

`narrate(snapshot)` takes a finished `DecisionSnapshot` and returns a plain-language
string. That string is assigned to `snapshot.narrative` and nowhere else. The LLM
cannot change the mode, the action, any probability, any energy figure, the
compliance result, or a reason code — it is handed the already-verified numbers
and asked only to phrase them. If `ANTHROPIC_API_KEY` is unset or the call fails,
a deterministic structured fallback is used and the snapshot is unaffected.
"""

from app.narrative.narrator import narrate

__all__ = ["narrate"]
