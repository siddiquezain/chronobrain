#!/usr/bin/env python3
"""
Run a scenario simulation end-to-end and print each tick payload.

Usage:
    python scripts/run_simulation.py [scenario] [seed]

Defaults: scenario=B, seed=42
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from app.simulation.scenarios import get_scenario_config, SCENARIO_PRESETS
from app.simulation.simulator import RaceSimulator


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "B"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    if scenario not in SCENARIO_PRESETS:
        print(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIO_PRESETS)}")
        sys.exit(1)

    config = get_scenario_config(scenario, seed=seed)
    sim = RaceSimulator(config=config, seed=seed)

    print(f"Running scenario {scenario} (seed={seed})")
    print("-" * 60)

    tick = 0
    while True:
        try:
            payload = sim.tick()
            tick += 1
            rs = payload["race_state"]
            st = payload["strategy"]
            print(
                f"Lap {rs['lap']:3d}/{rs['total_laps']} | "
                f"SoC {rs['soc_mj']:.2f} MJ | "
                f"Gap {rs['gap_to_car_ahead_s']:.3f}s | "
                f"Mode: {st['recommended_mode']:<25} | "
                f"Conf: {st['confidence']:.3f}"
            )
        except StopIteration:
            print(f"\nSimulation complete after {tick} ticks.")
            break


if __name__ == "__main__":
    main()
