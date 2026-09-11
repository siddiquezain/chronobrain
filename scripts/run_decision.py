#!/usr/bin/env python3
"""
Run one ChronoPace decision and print the snapshot + trace.

    python scripts/run_decision.py [scenario] [--seed N] [--lap N] [--total-laps N] [--narrative] [--json]

Defaults: scenario=B, seed=42, lap=last, total_laps=50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.decision import run_scenario  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lap", type=int, default=None)
    ap.add_argument("--total-laps", type=int, default=50)
    ap.add_argument("--narrative", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    snap = run_scenario(
        args.scenario, seed=args.seed, lap=args.lap,
        total_laps=args.total_laps, with_narrative=args.narrative,
    )

    if args.json:
        print(json.dumps(snap.model_dump(), indent=2, default=str))
        return

    d = snap.decision
    print(f"\nChronoPace — scenario {args.scenario}, seed {args.seed}, "
          f"lap {snap.meta.lap}/{snap.meta.total_laps} ({snap.meta.data_mode})")
    print("=" * 66)
    print(f"  DECISION : {d.mode}  /  {d.action}")
    print(f"  confidence {d.confidence:.0%}"
          + (f"   [OVERRIDE: {d.override_reason}]" if d.confidence_overridden else ""))
    print(f"  compliance.legal = {snap.compliance.legal}   legal modes: {snap.compliance.legal_modes}")
    print(f"  rival: {snap.rival.bucket}  mean {snap.rival.mean_reserve_mj:.2f} "
          f"+/- {snap.rival.reserve_std_mj:.2f} MJ  ({snap.rival.distribution})")
    print("\n  reasons:")
    for r in snap.reasons:
        print(f"    - {r}")
    print("\n  DECISION TRACE")
    for step in snap.trace:
        print(f"    {step.stage:20s} {step.detail}")
    if snap.narrative:
        print("\n  NARRATIVE\n" + "\n".join("    " + ln for ln in snap.narrative.splitlines()))
    print()


if __name__ == "__main__":
    main()
