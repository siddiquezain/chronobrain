"""
DEPRECATED — legacy strategy recommendation endpoint.

Served by the legacy Stack B `StrategyEngine`, which is a DIFFERENT engine from the
canonical v1 pipeline. Its numbers do not match `POST /api/v1/decision`. Kept only
for backward compatibility; the frontend must use `/api/v1/decision`.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deprecation import mark_deprecated
from app.core.state import get_race_state_manager
from app.models.strategy import StrategyRecommendation

router = APIRouter(prefix="/api/strategy", tags=["strategy (deprecated)"], deprecated=True)


@router.get("/recommendation", response_model=StrategyRecommendation, dependencies=[Depends(mark_deprecated)])
async def get_strategy_recommendation() -> StrategyRecommendation:
    """DEPRECATED. Use `POST /api/v1/decision`."""
    mgr = get_race_state_manager()
    rec = mgr.get_strategy()
    if rec is None:
        raise HTTPException(status_code=404, detail="No strategy recommendation available")
    return StrategyRecommendation(**rec)
