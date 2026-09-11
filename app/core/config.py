"""Centralized configuration for ChronoPace. Never scatter magic numbers in code."""

from __future__ import annotations

import os
from typing import Optional

from rule_gate import GateConfig

# Single source of truth for the regulatory constants. The v1 decision pipeline
# reads these straight from GateConfig; the (deprecated) Stack B engines read them
# from Settings, which now DEFAULTS from GateConfig rather than re-declaring the
# numbers. Provenance is catalogued in app/regulation/constants.py.
_GATE = GateConfig()


class Settings:
    """
    Application configuration. All values come from environment variables
    or fall back to sensible defaults. Use get_settings() to access the singleton.
    """

    def __init__(self):
        self.app_env: str = os.getenv("APP_ENV", "development")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # API
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        _cors = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
        self.cors_origins: list[str] = [x.strip() for x in _cors.split(",")]

        # Simulation
        self.simulation_interval_s: float = float(os.getenv("SIMULATION_INTERVAL_S", "1.0"))
        self.simulation_seed: int = int(os.getenv("SIMULATION_SEED", "42"))

        # Regulatory constants — DEFAULT from rule_gate.GateConfig (the single source
        # of truth). Env vars still override for local experiments. See
        # docs/regulation.md for the VERIFIED_FIA vs MODEL_ASSUMPTION split.
        self.energy_capacity_mj: float = float(
            os.getenv("ENERGY_CAPACITY_MJ", str(_GATE.max_deployment_per_lap_mj))
        )
        self.min_energy_reserve_mj: float = float(os.getenv("MIN_ENERGY_RESERVE_MJ", "1.0"))
        self.max_deployment_per_lap_mj: float = float(
            os.getenv("MAX_DEPLOYMENT_PER_LAP_MJ", str(_GATE.max_deployment_per_lap_mj))
        )
        self.max_delta_soc_mj: float = float(
            os.getenv("MAX_DELTA_SOC_MJ", str(_GATE.max_delta_soc_mj))
        )
        self.overtake_bonus_mj: float = float(
            os.getenv("OVERTAKE_BONUS_MJ", str(_GATE.overtake_bonus_mj))
        )
        self.overtake_detection_gap_threshold_s: float = float(
            os.getenv(
                "OVERTAKE_DETECTION_GAP_THRESHOLD_S",
                str(_GATE.overtake_detection_gap_threshold_s),
            )
        )

        # Overtake weights (configurable, not hardcoded — these are not FIA values)
        self.overtake_weight_closing_speed: float = float(
            os.getenv("OVERTAKE_WEIGHT_CLOSING_SPEED", "0.25")
        )
        self.overtake_weight_gap: float = float(os.getenv("OVERTAKE_WEIGHT_GAP", "0.20"))
        self.overtake_weight_braking_zone: float = float(
            os.getenv("OVERTAKE_WEIGHT_BRAKING_ZONE", "0.20")
        )
        self.overtake_weight_slipstream: float = float(
            os.getenv("OVERTAKE_WEIGHT_SLIPSTREAM", "0.15")
        )
        self.overtake_weight_corner_exit: float = float(
            os.getenv("OVERTAKE_WEIGHT_CORNER_EXIT", "0.10")
        )
        self.overtake_weight_energy_advantage: float = float(
            os.getenv("OVERTAKE_WEIGHT_ENERGY_ADVANTAGE", "0.10")
        )

        # Risk weights
        self.risk_weight_energy: float = float(os.getenv("RISK_WEIGHT_ENERGY", "0.35"))
        self.risk_weight_overtake: float = float(os.getenv("RISK_WEIGHT_OVERTAKE", "0.40"))
        self.risk_weight_strategic: float = float(os.getenv("RISK_WEIGHT_STRATEGIC", "0.25"))

        # ML model
        self.ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/overtake_model.joblib")

        # LLM narrator (optional — Stage 4 only)
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
