"""
Load the trained overtake-success model and expose a deterministic predict.

Reproducibility contract:
- feature ORDER is fixed by `app.ml.features.FEATURE_NAMES` (8 features)
- the model is trained with a fixed `random_state` (see train.py) and RandomForest
  inference is deterministic
- a sidecar `overtake_rf.meta.json` records feature names + sklearn version + a
  training-data hash; `model_info()` surfaces it, and a feature-count mismatch
  raises rather than silently mis-predicting
- if the model file is absent, a documented heuristic fallback is used (also
  deterministic) so the pipeline still runs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "models" / "overtake_rf.joblib"
_META_PATH = Path(__file__).parent / "models" / "overtake_rf.meta.json"
_clf = None
_meta: Optional[dict] = None


def _load():
    global _clf, _meta
    if _clf is not None:
        return _clf
    if not _MODEL_PATH.exists():
        logger.warning("Overtake RF model not found - using fallback heuristic")
        return None
    import joblib

    _clf = joblib.load(_MODEL_PATH)
    if _META_PATH.exists():
        _meta = json.loads(_META_PATH.read_text())
    logger.info("Overtake RF model loaded (%s)", model_info().get("trained_at", "no meta"))
    return _clf


def model_info() -> dict:
    if _META_PATH.exists():
        return json.loads(_META_PATH.read_text())
    return {"status": "no-metadata", "feature_names": FEATURE_NAMES, "source": "heuristic-fallback"}


def predict_probability(features: np.ndarray) -> float:
    """Overtake success probability [0, 1]. Deterministic on both code paths."""
    if features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES)} features {FEATURE_NAMES}, got shape {features.shape}"
        )

    clf = _load()
    if clf is not None:
        return float(clf.predict_proba(features)[0, 1])

    # Heuristic fallback — matches the dataset label formula in dataset.py
    gap, closing, slipstream, soc, _, _, _, drs = features[0]
    score = (
        max(0.0, 1.0 - gap / 3.0) * 0.3
        + max(0.0, min(1.0, closing / 15.0)) * 0.25
        + slipstream * 0.15
        + (soc / 9.0) * 0.15
        + drs * 0.15
    )
    return float(np.clip(score, 0.0, 1.0))
