"""Train and persist the overtake success RandomForestClassifier."""

import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from app.ml.dataset import generate

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "overtake_rf.joblib"


def train(save: bool = True) -> RandomForestClassifier:
    X, y = generate()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    cv_scores = cross_val_score(clf, X, y, cv=5)
    logger.info(f"RF CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

    return clf
