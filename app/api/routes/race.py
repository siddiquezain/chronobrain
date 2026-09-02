"""Race state endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.core.state import get_race_state_manager
from app.models.race import RaceState, RaceStateUpdate

router = APIRouter(prefix="/api/race", tags=["race"])


@router.get("/state", response_model=RaceState)
async def get_race_state() -> RaceState:
    mgr = get_race_state_manager()
    state = mgr.get_race_state()
    if state is None:
        raise HTTPException(status_code=404, detail="No active race state")
    return RaceState(**state)


@router.post("/update", response_model=RaceState)
async def update_race_state(update: RaceStateUpdate) -> RaceState:
    mgr = get_race_state_manager()
    current = mgr.get_race_state() or {}
    patch = {k: v for k, v in update.model_dump().items() if v is not None}
    current.update(patch)
    current.setdefault("timestamp", datetime.utcnow().isoformat())
    mgr.update_race_state(current)
    return RaceState(**current)
