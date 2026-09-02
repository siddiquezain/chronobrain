"""Load the trained model and expose a predict function for the overtake engine."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "models" / "overtake_rf.joblib"
_clf = None


def _load() -> Optional[object]:
    global _clf
    if _clf is not None:
        return _clf
    if not _MODEL_PATH.exists():
        logger.warning("Overtake RF model not found — using fallback heuristic")
        return None
    import joblib
    _clf = joblib.load(_MODEL_PATH)
    logger.info("Overtake RF model loaded")
    return _clf


def predict_probability(features: np.ndarray) -> float:
    """
    Return overtake success probability [0,1].
    Falls back to the feature-weighted heuristic if the model file is missing.
    """
    clf = _load()
    if clf is not None:
        return float(clf.predict_proba(features)[0, 1])

    # Heuristic fallback — matches dataset label formula
    gap, closing, slipstream, soc, _, _, _, drs = features[0]
    score = (
        max(0.0, 1.0 - gap / 3.0) * 0.3
        + max(0.0, min(1.0, closing / 15.0)) * 0.25
        + slipstream * 0.15
        + (soc / 9.0) * 0.15
        + drs * 0.15
    )
    return float(np.clip(score, 0.0, 1.0))
