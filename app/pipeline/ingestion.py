"""Telemetry ingestion — validates raw input before it enters the pipeline."""

import logging
from typing import Any

from app.models.telemetry import TelemetryState

logger = logging.getLogger(__name__)


def ingest(raw: dict[str, Any]) -> TelemetryState:
    """Parse and validate a raw telemetry dict into a TelemetryState."""
    return TelemetryState(**raw)
