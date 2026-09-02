"""Energy state endpoint."""

from fastapi import APIRouter, HTTPException

from app.core.state import get_race_state_manager
from app.models.energy import EnergyState

router = APIRouter(prefix="/api/energy", tags=["energy"])


@router.get("/state", response_model=EnergyState)
async def get_energy_state() -> EnergyState:
    mgr = get_race_state_manager()
    state = mgr.get_energy()
    if state is None:
        raise HTTPException(status_code=404, detail="No energy state available")
    return EnergyState(**state)
