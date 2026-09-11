"""
app.data — telemetry provider abstraction.

    TELEMETRY SOURCES
          |
   +------+---------------------+
   |                            |
 FastF1 historical replay   Synthetic simulator
   (fastf1_service.py)      (telemetry_simulator.py, root)
   |                            |
   +------+---------------------+
          |
   TELEMETRY NORMALIZER  (normalizer.py -> NormalizedLap)
          |
   CHRONOPACE DECISION ENGINE  (app.decision.engine)

The ChronoPace decision engine imports from this package only. It never imports
`fastf1` and never touches a pandas DataFrame. Providers yield `NormalizedLap`
objects (normalizer.py); every field the engine needs is on that model, with
source-specific detail (session loading, DataFrame handling, caching, driver and
session selection) hidden inside `fastf1_service.py`.
"""

from app.data.samples import NormalizedLap, TelemetrySample
from app.data.providers import (
    TelemetryProvider,
    SyntheticProvider,
    ReplayProvider,
    FASTF1_AVAILABLE,
    build_provider,
)

__all__ = [
    "NormalizedLap",
    "TelemetrySample",
    "TelemetryProvider",
    "SyntheticProvider",
    "ReplayProvider",
    "FASTF1_AVAILABLE",
    "build_provider",
]
