"""Overtake analysis endpoint."""

from fastapi import APIRouter, HTTPException

from app.core.state import get_race_state_manager
from app.models.overtake import OvertakeAnalysis

router = APIRouter(prefix="/api/overtake", tags=["overtake"])


@router.get("/current", response_model=OvertakeAnalysis)
async def get_overtake_analysis() -> OvertakeAnalysis:
    mgr = get_race_state_manager()
    analysis = mgr.get_overtake()
    if analysis is None:
        raise HTTPException(status_code=404, detail="No overtake analysis available")
    return OvertakeAnalysis(**analysis)
