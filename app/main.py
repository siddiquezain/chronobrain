"""ChronoPace FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, race, energy, overtake, strategy, simulation
from app.api.routes.v1 import decision as v1_decision
from app.api.websocket import websocket_endpoint
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChronoPace",
    description="Motorsport intelligence backend — energy deployment strategy engine",
    version="1.0.0",
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

# WebSocket
app.add_api_websocket_route("/ws", websocket_endpoint)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("ChronoPace backend starting up")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("ChronoPace backend shutting down")
