"""
In-memory race state manager — single source of truth during a session.
No database needed for hackathon demo.
ponytail: global lock; per-account locks if multi-user throughput matters.
"""

import threading
from datetime import datetime
from typing import Optional


class RaceStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._state: Optional[dict] = None
        self._energy: Optional[dict] = None
        self._overtake: Optional[dict] = None
        self._strategy: Optional[dict] = None
        self._simulation_running: bool = False
        self._simulation_scenario: str = "B"
        self._simulation_tick: int = 0
        self._simulation_start_time: Optional[datetime] = None

    def update_race_state(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def update_energy(self, energy: dict) -> None:
        with self._lock:
            self._energy = energy

    def update_overtake(self, overtake: dict) -> None:
        with self._lock:
            self._overtake = overtake

    def update_strategy(self, strategy: dict) -> None:
        with self._lock:
            self._strategy = strategy

    def get_race_state(self) -> Optional[dict]:
        with self._lock:
            return self._state

    def get_energy(self) -> Optional[dict]:
        with self._lock:
            return self._energy

    def get_overtake(self) -> Optional[dict]:
        with self._lock:
            return self._overtake

    def get_strategy(self) -> Optional[dict]:
        with self._lock:
            return self._strategy

    def set_simulation_running(self, running: bool, scenario: str = "B") -> None:
        with self._lock:
            self._simulation_running = running
            if running:
                self._simulation_scenario = scenario
                self._simulation_tick = 0
                self._simulation_start_time = datetime.utcnow()
            else:
                self._simulation_start_time = None

    def increment_tick(self) -> int:
        with self._lock:
            self._simulation_tick += 1
            return self._simulation_tick

    def get_simulation_status(self) -> dict:
        with self._lock:
            elapsed = 0.0
            if self._simulation_start_time and self._simulation_running:
                elapsed = (datetime.utcnow() - self._simulation_start_time).total_seconds()
            return {
                "running": self._simulation_running,
                "scenario": self._simulation_scenario,
                "tick": self._simulation_tick,
                "elapsed_s": round(elapsed, 2),
            }


_manager: Optional[RaceStateManager] = None


def get_race_state_manager() -> RaceStateManager:
    global _manager
    if _manager is None:
        _manager = RaceStateManager()
    return _manager
