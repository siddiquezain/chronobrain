"""
outcome_log.py — append-only prediction/decision/outcome log (P1, opt-in).

    prediction  ->  decision  ->  actual action  ->  actual outcome

Off the hot path: `run_decision()` does NOT call this. A caller (a replay harness,
a demo script, or the API layer) records a snapshot and, later, what actually
happened. The file is JSONL so it can be replayed for calibration without a DB.

This is a hook, not a learning system — nothing here feeds back into the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from app.decision.snapshot import DecisionSnapshot


@dataclass
class OutcomeRecord:
    logged_at: str
    seed: int
    lap: int
    config_fingerprint: str
    predicted_mode: str
    predicted_action: str
    confidence: float
    predicted_overtake_prob: float
    rival_mean_reserve_mj: float
    rival_reserve_std_mj: float
    actual_action: Optional[str] = None
    actual_outcome: Optional[str] = None      # e.g. "OVERTAKE_COMPLETED" | "HELD" | "LOST_POSITION"
    actual_laptime_delta_s: Optional[float] = None


class OutcomeLog:
    def __init__(self, path: str | Path = "data/outcome_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        snapshot: "DecisionSnapshot",
        *,
        actual_action: Optional[str] = None,
        actual_outcome: Optional[str] = None,
        actual_laptime_delta_s: Optional[float] = None,
    ) -> OutcomeRecord:
        rec = OutcomeRecord(
            logged_at=datetime.now(timezone.utc).isoformat(),
            seed=snapshot.meta.seed,
            lap=snapshot.meta.lap,
            config_fingerprint=snapshot.meta.config_fingerprint,
            predicted_mode=snapshot.decision.mode,
            predicted_action=snapshot.decision.action,
            confidence=snapshot.decision.confidence,
            predicted_overtake_prob=snapshot.opportunity.current_window_overtake_prob,
            rival_mean_reserve_mj=snapshot.rival.mean_reserve_mj,
            rival_reserve_std_mj=snapshot.rival.reserve_std_mj,
            actual_action=actual_action,
            actual_outcome=actual_outcome,
            actual_laptime_delta_s=actual_laptime_delta_s,
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.__dict__) + "\n")
        return rec

    def read(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter(())
        return (json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line)
