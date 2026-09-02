"""WebSocket manager for real-time race state broadcast."""

import json
import logging
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

_connections: Set[WebSocket] = set()


async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_connections)}")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        _connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(_connections)}")


async def broadcast(data: dict) -> None:
    """Broadcast race state update to all connected WebSocket clients."""
    if not _connections:
        return
    message = json.dumps(data, default=str)
    disconnected: Set[WebSocket] = set()
    for ws in list(_connections):
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    _connections -= disconnected
