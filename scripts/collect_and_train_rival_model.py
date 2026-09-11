#!/usr/bin/env python
"""
Offline training script for the Rival Energy Observation Model.

Loads real FastF1 sessions, accumulates rival observable features per lap using
RivalObservationBaseline (causal, per-driver), generates KMeans weak labels,
trains a calibrated Random Forest, saves artifact to app/ml/models/.

Usage:
    cd /path/to/ChronoPace-Backend
    python scripts/collect_and_train_rival_model.py

Requires FastF1 network access. Run offline; artifact stored in app/ml/models/.
Does NOT use our_soc_mj or any internal energy state — only rival observables.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sessions to collect from. Edit as needed.
# ponytail: hardcoded list; add argparse if dataset grows beyond ~10 sessions
SESSIONS = [
    {"year": 2024, "event": "Monza",       "session": "R", "our_driver": "VER", "rival_driver": "LEC"},
    {"year": 2024, "event": "Silverstone", "session": "R", "our_driver": "NOR", "rival_driver": "VER"},
    {"year": 2024, "event": "Spa",         "session": "R", "our_driver": "HAM", "rival_driver": "LEC"},
]


def _make_obs_adapter(nl):
    """Adapt NormalizedLap rival fields to the interface RivalObservationBaseline expects."""
    class _Obs:
        terminal_speed_kmh      = nl.rival_terminal_speed_kmh
        clipping_point_fraction = nl.rival_clipping_point_fraction
        corner_exit_accel_g     = nl.rival_corner_exit_accel_g
        sector_delta_s          = nl.rival_sector_delta_s
    return _Obs()


def collect_features(sessions: list[dict]) -> np.ndarray:
    """
    Load FastF1 sessions. For each valid rival lap, extract the 7-element feature vector:
    [z_speed, z_clip, z_accel, z_sector, speed_slope, speed_persistence, sector_persistence].

    Returns float array (n_rows, 7). No SoC or internal energy columns.

    load_replay() signature (keyword-only):
        year: int, event, session: str, our_driver: str, rival_driver: str,
        laps=None, cache_dir=..., energy_model=None, label=None,
        scheduled_laps=None, dynamic_rival=False, selector_config=None
    """
    from rival_estimator import RivalObservationBaseline, RivalTemporalTracker
    from app.data.fastf1_service import load_replay

    rows: list[list[float]] = []

    for s in sessions:
        logger.info("Loading %s %s %s ...", s["year"], s["event"], s["session"])
        try:
            laps = load_replay(
                year=s["year"],
                event=s["event"],
                session=s["session"],
                our_driver=s["our_driver"],
                rival_driver=s["rival_driver"],
            )
        except Exception as exc:
            logger.warning("Skipping %s %s: %s", s["event"], s["year"], exc)
            continue

        baseline = RivalObservationBaseline()
        temporal = RivalTemporalTracker()

        for nl in laps:
            if not nl.has_rival_observation or nl.rival_lap_status != "racing":
                continue

            obs = _make_obs_adapter(nl)
            cpd = nl.rival_compound or "UNKNOWN"
            baseline.update(obs, compound=cpd)

            if not baseline.is_ready:
                continue  # warm-up; Z-scores unreliable before min_obs

            z_sp, z_cl, z_ac, z_se = baseline.z_score(obs, compound=cpd)
            temporal.update(z_sp, z_se)

            rows.append([
                z_sp,
                z_cl,
                z_ac,
                z_se,
                temporal.speed_slope,
                float(temporal.speed_persistence),
                float(temporal.sector_persistence),
            ])

    if not rows:
        raise RuntimeError(
            "No valid feature rows collected. "
            "Check FastF1 availability and SESSIONS list."
        )

    X = np.array(rows, dtype=float)
    logger.info("Collected %d feature rows from %d sessions", len(X), len(sessions))
    return X


if __name__ == "__main__":
    from app.ml.rival_observation_model import RivalObservationModel

    X = collect_features(SESSIONS)
    model = RivalObservationModel.train(X, save=True)
    logger.info("Training complete. Artifact at app/ml/models/rival_obs_model.joblib")

    # Quick sanity check
    ev = model.predict_evidence(np.zeros((1, 7)))
    logger.info("Evidence for all-zero feature vector: %s", ev)
