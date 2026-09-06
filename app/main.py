"""ChronoPace FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, race, energy, overtake, strategy, simulation
from app.api.routes.v1 import decision as v1_decision
from app.api.websocket import websocket_endpoint
from app.core.logging import setup_logging

# Legacy Stack B route prefixes. `POST /api/v1/decision` is the single authoritative
# ChronoPace decision source; anything under these prefixes is non-authoritative.
_LEGACY_PREFIXES = ("/api/race", "/api/energy", "/api/overtake", "/api/strategy")

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("ChronoPace backend starting up")
    yield
    logger.info("ChronoPace backend shutting down")


app = FastAPI(
    title="ChronoPace",
    description="2026 Formula 1 energy-deployment decision engine (TrackShift 2026)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(race.router)
app.include_router(energy.router)
app.include_router(overtake.router)
app.include_router(strategy.router)
app.include_router(simulation.router)
app.include_router(v1_decision.router)


@app.middleware("http")
async def _stamp_legacy_deprecation(request: Request, call_next):
    """Deprecation headers land on legacy responses even on 404 / error paths."""
    response = await call_next(request)
    if request.url.path.startswith(_LEGACY_PREFIXES):
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = '</api/v1/decision>; rel="successor-version"'
        response.headers["Warning"] = (
            '299 - "Legacy Stack B endpoint - NOT authoritative. '
            'Use POST /api/v1/decision."'
        )
    return response


# WebSocket
app.add_api_websocket_route("/ws", websocket_endpoint)
