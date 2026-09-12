"""
DEPRECATED — legacy energy state endpoint.

Served by the legacy Stack B `EnergyEngine`. Not authoritative. The energy figures
in `POST /api/v1/decision` (`snapshot.energy`) are the source of truth.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deprecation import mark_deprecated
from app.core.state import get_race_state_manager
from app.models.energy import EnergyState

router = APIRouter(prefix="/api/energy", tags=["energy (deprecated)"], deprecated=True)


@router.get("/state", response_model=EnergyState, dependencies=[Depends(mark_deprecated)])
async def get_energy_state() -> EnergyState:
    """DEPRECATED. Use `POST /api/v1/decision` -> `energy`."""
    mgr = get_race_state_manager()
    state = mgr.get_energy()
    if state is None:
        raise HTTPException(status_code=404, detail="No energy state available")
    return EnergyState(**state)
