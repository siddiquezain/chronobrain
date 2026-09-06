"""
/api/v1/replay/* — historical F1 race replay through the existing ChronoPace
pipeline. Real FastF1 telemetry, censored lap by lap (no hindsight), same
`run_decision` as the synthetic and demo paths.

No ground truth is ever returned (F1 publishes no rival ERS SoC; ChronoPace
infers it). If `fastf1` is missing or the session can't load, returns 503 with a
clear message — it never fabricates data.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.data.fastf1_service import FastF1Unavailable, fastf1_available
from app.replay import HISTORICAL_RACES, run_historical_lap, run_historical_replay

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/replay", tags=["replay"])


class HistoricalReplayRequest(BaseModel):
    race: str = Field("2024_italian_gp", description="race key (see GET /api/v1/replay/races)")
    driver: Optional[str] = Field(None, description="3-letter code; defaults from the registry (LEC)")
    rival: Optional[str] = Field(None, description="3-letter code; defaults from the registry (PIA)")
    start_lap: int = 1
    end_lap: Optional[int] = None
    seed: int = 42


@router.get("/races")
def list_races() -> dict:
    return {
        "fastf1_available": fastf1_available(),
        "races": [
            {
                "key": r.key, "name": r.name, "circuit": r.circuit, "year": r.year,
                "session": r.session, "scheduled_laps": r.scheduled_laps,
                "default_driver": r.default_driver, "default_rival": r.default_rival,
                "note": r.note,
            }
            for r in HISTORICAL_RACES.values()
        ],
    }


@router.post("/historical")
def historical_replay(req: HistoricalReplayRequest) -> dict:
    try:
        return run_historical_replay(
            race_key=req.race, driver=req.driver, rival=req.rival,
            start_lap=req.start_lap, end_lap=req.end_lap, seed=req.seed,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FastF1Unavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("historical replay failed")
        raise HTTPException(500, f"replay error: {exc}") from exc


@router.get("/historical/{race}/{lap}")
def historical_lap(
    race: str, lap: int, driver: Optional[str] = None, rival: Optional[str] = None,
    seed: int = 42, full_snapshot: bool = False,
) -> dict:
    try:
        return run_historical_lap(
            race_key=race, driver=driver, rival=rival, lap=lap, seed=seed,
            full_snapshot=full_snapshot,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FastF1Unavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("historical lap replay failed")
        raise HTTPException(500, f"replay error: {exc}") from exc
