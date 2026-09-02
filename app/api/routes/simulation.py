"""Simulation control endpoints — start/stop/status for the demo race loop."""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.api.websocket import broadcast
from app.core.state import get_race_state_manager
from app.simulation.scenarios import SCENARIO_PRESETS, get_scenario_config
from app.simulation.simulator import RaceSimulator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/simulation", tags=["simulation"])

_TICK_INTERVAL_S = 2.0  # seconds between ticks
_active_task: asyncio.Task | None = None


class SimulationStartRequest(BaseModel):
    scenario: str = "B"
    tick_interval_s: float = _TICK_INTERVAL_S
    seed: int = 42


async def _run_simulation(scenario: str, tick_interval_s: float, seed: int) -> None:
    mgr = get_race_state_manager()
    mgr.set_simulation_running(True, scenario)

    config = get_scenario_config(scenario, seed=seed)
    sim = RaceSimulator(config=config, seed=seed)

    try:
        while True:
            try:
                payload = sim.tick()
            except StopIteration:
                logger.info("Simulation complete — all laps done")
                break

            mgr.update_race_state(payload["race_state"])
            mgr.update_energy(payload["energy"])
            mgr.update_overtake(payload["overtake"])
            mgr.update_strategy(payload["strategy"])
            mgr.increment_tick()

            await broadcast(payload)
            await asyncio.sleep(tick_interval_s)
    except asyncio.CancelledError:
        logger.info("Simulation loop cancelled")
    finally:
        mgr.set_simulation_running(False)


@router.post("/start")
async def start_simulation(
    body: SimulationStartRequest, background_tasks: BackgroundTasks
) -> dict:
    global _active_task

    if body.scenario not in SCENARIO_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{body.scenario}'. Choose from {list(SCENARIO_PRESETS)}",
        )

    if _active_task and not _active_task.done():
        raise HTTPException(status_code=409, detail="Simulation already running")

    loop = asyncio.get_event_loop()
    _active_task = loop.create_task(
        _run_simulation(body.scenario, body.tick_interval_s, body.seed)
    )
    return {"status": "started", "scenario": body.scenario, "seed": body.seed}


@router.post("/stop")
async def stop_simulation() -> dict:
    global _active_task
    if _active_task and not _active_task.done():
        _active_task.cancel()
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/status")
async def simulation_status() -> dict:
    mgr = get_race_state_manager()
    return mgr.get_simulation_status()
