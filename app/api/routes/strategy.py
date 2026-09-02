"""Strategy recommendation endpoint."""

from fastapi import APIRouter, HTTPException

from app.core.state import get_race_state_manager
from app.models.strategy import StrategyRecommendation

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


@router.get("/recommendation", response_model=StrategyRecommendation)
async def get_strategy_recommendation() -> StrategyRecommendation:
    mgr = get_race_state_manager()
    rec = mgr.get_strategy()
    if rec is None:
        raise HTTPException(status_code=404, detail="No strategy recommendation available")
    return StrategyRecommendation(**rec)
