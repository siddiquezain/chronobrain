"""
DEPRECATED — legacy overtake analysis endpoint.

Served by the legacy Stack B `OvertakeEngine`. Not authoritative. The v1 pipeline
exposes opportunity data in `POST /api/v1/decision` (`snapshot.opportunity`).
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deprecation import mark_deprecated
from app.core.state import get_race_state_manager
from app.models.overtake import OvertakeAnalysis

router = APIRouter(prefix="/api/overtake", tags=["overtake (deprecated)"], deprecated=True)


@router.get("/current", response_model=OvertakeAnalysis, dependencies=[Depends(mark_deprecated)])
async def get_overtake_analysis() -> OvertakeAnalysis:
    """DEPRECATED. Use `POST /api/v1/decision` -> `opportunity`."""
    mgr = get_race_state_manager()
    analysis = mgr.get_overtake()
    if analysis is None:
        raise HTTPException(status_code=404, detail="No overtake analysis available")
    return OvertakeAnalysis(**analysis)
