"""Light preprocessing applied to TelemetryState before engines run."""

from app.models.telemetry import TelemetryState


def preprocess(telemetry: TelemetryState) -> TelemetryState:
    """
    Clamp / normalise fields that engines expect within tight bounds.
    Returns a new TelemetryState (model_copy keeps Pydantic immutability intact).
    """
    updates: dict = {}

    if telemetry.soc_mj is not None:
        updates["soc_mj"] = max(0.0, min(9.0, telemetry.soc_mj))
    if telemetry.soc_pct is not None:
        updates["soc_pct"] = max(0.0, min(100.0, telemetry.soc_pct))
    if telemetry.energy_deployment_mj is not None:
        updates["energy_deployment_mj"] = max(0.0, telemetry.energy_deployment_mj)
    if telemetry.energy_harvest_mj is not None:
        updates["energy_harvest_mj"] = max(0.0, telemetry.energy_harvest_mj)

    if not updates:
        return telemetry
    return telemetry.model_copy(update=updates)
