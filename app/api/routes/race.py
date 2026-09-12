"""
DEPRECATED — legacy race-state endpoints.

Backed by the in-memory Stack B `RaceStateManager`, populated only while a legacy
simulation loop is running. Not authoritative and not required by the frontend:
`POST /api/v1/decision` carries lap / total_laps / data_mode in `snapshot.meta`.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deprecation import mark_deprecated
from app.core.state import get_race_state_manager
from app.models.race import RaceState, RaceStateUpdate

router = APIRouter(prefix="/api/race", tags=["race (deprecated)"], deprecated=True)


@router.get("/state", response_model=RaceState, dependencies=[Depends(mark_deprecated)])
async def get_race_state() -> RaceState:
    """DEPRECATED. Use `POST /api/v1/decision` -> `meta`."""
    mgr = get_race_state_manager()
    state = mgr.get_race_state()
    if state is None:
        raise HTTPException(status_code=404, detail="No active race state")
    return RaceState(**state)


@router.post("/update", response_model=RaceState, dependencies=[Depends(mark_deprecated)])
async def update_race_state(update: RaceStateUpdate) -> RaceState:
    """DEPRECATED. The v1 pipeline reads telemetry from a provider, not this store."""
    mgr = get_race_state_manager()
    current = mgr.get_race_state() or {}
    patch = {k: v for k, v in update.model_dump().items() if v is not None}
    current.update(patch)
    current.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    mgr.update_race_state(current)
    return RaceState(**current)
