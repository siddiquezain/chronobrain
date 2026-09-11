"""Train and persist the overtake-success RandomForestClassifier + reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from app.ml.dataset import N_SAMPLES, generate
from app.ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "overtake_rf.joblib"
META_PATH = Path(__file__).parent / "models" / "overtake_rf.meta.json"

_RANDOM_STATE = 42
_N_ESTIMATORS = 100


def train(save: bool = True) -> RandomForestClassifier:
    X, y = generate()
    clf = RandomForestClassifier(
        n_estimators=_N_ESTIMATORS, random_state=_RANDOM_STATE, n_jobs=1
    )
    clf.fit(X, y)

    cv = cross_val_score(clf, X, y, cv=5)
    logger.info("RF CV accuracy: %.3f +/- %.3f", cv.mean(), cv.std())

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "n_estimators": _N_ESTIMATORS,
            "random_state": _RANDOM_STATE,
            "n_samples": N_SAMPLES,
            "dataset_hash": hashlib.sha256(
                np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()
            ).hexdigest()[:16],
            "cv_accuracy_mean": round(float(cv.mean()), 4),
            "cv_accuracy_std": round(float(cv.std()), 4),
            "note": "synthetic training data — structural domain knowledge, not measured outcomes",
        }
        META_PATH.write_text(json.dumps(meta, indent=2))
        logger.info("Model + metadata saved to %s", MODEL_PATH.parent)

    return clf
