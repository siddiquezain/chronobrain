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
from app.replay.historical import strategic_rival_timeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/replay", tags=["replay"])


class HistoricalReplayRequest(BaseModel):
    race: str = Field("2024_italian_gp", description="race key (see GET /api/v1/replay/races)")
    driver: Optional[str] = Field(None, description="3-letter code; defaults from the registry (LEC)")
    rival: Optional[str] = Field(
        None,
        description="3-letter code; the FOCUS / fallback rival (default PIA). With "
        "dynamic_rival the strategic rival is re-selected from the field each lap.",
    )
    start_lap: int = 1
    end_lap: Optional[int] = None
    seed: int = 42
    dynamic_rival: bool = Field(
        True,
        description="Re-select the strategically relevant opponent from the full field "
        "every lap (causal). False = fixed two-car analysis against `rival`.",
    )


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
            dynamic_rival=req.dynamic_rival,
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
    seed: int = 42, full_snapshot: bool = False, dynamic_rival: bool = True,
) -> dict:
    try:
        return run_historical_lap(
            race_key=race, driver=driver, rival=rival, lap=lap, seed=seed,
            full_snapshot=full_snapshot, dynamic_rival=dynamic_rival,
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


@router.get("/historical/{race}/{lap}/timeline")
def historical_lap_timeline(
    race: str, lap: int, driver: Optional[str] = None, rival: Optional[str] = None,
    max_ticks: int = 400,
) -> dict:
    """Telemetry-tick strategic-rival stream for one lap (the detail behind the
    compact per-lap summary). Sub-lap rival changes are real selector output, not
    a cosmetic timeline."""
    try:
        return strategic_rival_timeline(
            race_key=race, lap=lap, driver=driver, rival=rival, max_ticks=max_ticks,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FastF1Unavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("historical timeline failed")
        raise HTTPException(500, f"timeline error: {exc}") from exc
