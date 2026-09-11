"""
Rival Energy Observation Model — ML-based observation likelihood for the particle filter.

PROXY LABEL NOTICE: This model is trained on KMeans-clustered behavioral patterns
from observable telemetry. LOW/MEDIUM/HIGH labels are proxies for energy-management
behavior, NOT true rival SoC measurements. FastF1 does not publish rival battery SoC.

GROUND-TRUTH LEAKAGE PROHIBITION: our_soc_mj and any derivative MUST NEVER appear
in training features or runtime inference. Enforced structurally by RIVAL_OBS_FEATURE_NAMES
and verified by tests/test_rival_no_leakage.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature contract — immutable, must never include our_soc_mj or internal energy
RIVAL_OBS_FEATURE_NAMES = [
    "z_terminal_speed",     # Z-score vs rival's own running baseline
    "z_clipping_fraction",  # Z-score vs rival's own running baseline
    "z_corner_exit_accel",  # Z-score vs rival's own running baseline
    "z_sector_delta",       # Z-score vs rival's own running baseline
    "speed_slope",          # Lap-over-lap trend of speed Z (last 5 laps)
    "speed_persistence",    # Consecutive laps with negative speed Z
    "sector_persistence",   # Consecutive laps with positive sector Z (slower)
]

ENERGY_LABELS = ["LOW", "MEDIUM", "HIGH"]

_MODEL_PATH = Path(__file__).parent / "models" / "rival_obs_model.joblib"
_META_PATH = Path(__file__).parent / "models" / "rival_obs_model.meta.json"


class RivalObservationModel:
    """
    Calibrated ML observation model for rival energy-state inference.

    predict_evidence() returns P(LOW/MEDIUM/HIGH | observable features).
    These probabilities are used as particle weights in RivalStateEstimator.update().
    When absent, the estimator falls back to hand-coded Gaussian likelihoods.
    """

    _UNIFORM = {label: 1.0 / 3.0 for label in ENERGY_LABELS}

    def __init__(self, clf, meta: dict) -> None:
        self._clf = clf
        self._meta = meta
        # Map class label string → column index in predict_proba output
        self._label_idx: dict[str, int] = {
            str(c): i for i, c in enumerate(clf.classes_)
        }

    def predict_evidence(self, feature_vec: np.ndarray) -> dict[str, float]:
        """
        Compute energy-state probabilities from a (1, 7) feature vector.

        Args:
            feature_vec: shape (1, 7) — values from RIVAL_OBS_FEATURE_NAMES order.
                         Must contain only observable rival features; our_soc_mj excluded.

        Returns:
            {'LOW': p, 'MEDIUM': p, 'HIGH': p} summing to 1.0.
        """
        if feature_vec.shape[-1] != len(RIVAL_OBS_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(RIVAL_OBS_FEATURE_NAMES)} features "
                f"({RIVAL_OBS_FEATURE_NAMES}), got {feature_vec.shape[-1]}"
            )
        proba = self._clf.predict_proba(feature_vec)[0]
        return {
            label: float(proba[self._label_idx[label]])
            for label in ENERGY_LABELS
            if label in self._label_idx
        }

    def model_info(self) -> dict:
        return dict(self._meta)

    @classmethod
    def uniform_evidence(cls) -> dict[str, float]:
        """Uniform distribution — maximum uncertainty. Used as a fallback."""
        return dict(cls._UNIFORM)

    @classmethod
    def load_or_none(cls) -> Optional["RivalObservationModel"]:
        """
        Load model artifact. Returns None if absent or feature-count mismatch —
        the estimator will use the Gaussian fallback in that case.
        """
        if not _MODEL_PATH.exists():
            logger.info(
                "Rival observation model not found at %s — Gaussian fallback active", _MODEL_PATH
            )
            return None
        import joblib  # lazy: only imported when actually loading

        clf = joblib.load(_MODEL_PATH)
        meta = json.loads(_META_PATH.read_text()) if _META_PATH.exists() else {}

        expected = len(RIVAL_OBS_FEATURE_NAMES)
        if hasattr(clf, "n_features_in_") and clf.n_features_in_ != expected:
            logger.error(
                "Rival obs model feature count mismatch (expected %d, got %d) — fallback",
                expected,
                clf.n_features_in_,
            )
            return None

        logger.info(
            "Rival observation model loaded (trained_at=%s, labels=%s)",
            meta.get("trained_at", "unknown"),
            meta.get("label_scheme", "unknown"),
        )
        return cls(clf, meta)

    @classmethod
    def train(cls, X: np.ndarray, save: bool = True) -> "RivalObservationModel":
        """
        Train on observable feature matrix X (n_samples, 7).

        Labels are generated via KMeans(3) weak supervision — behavioral clusters
        of observable telemetry patterns, NOT true rival SoC. This is a proxy-label
        approach. See module docstring.

        X must contain only features from RIVAL_OBS_FEATURE_NAMES.
        our_soc_mj and any internal energy field must never appear in X.

        Args:
            X: Feature matrix shape (n_samples, 7). All columns from RIVAL_OBS_FEATURE_NAMES.
            save: If True, persist model + metadata to app/ml/models/.

        Returns:
            Fitted RivalObservationModel instance.
        """
        import hashlib
        from datetime import datetime, timezone

        import joblib
        import sklearn
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        if X.shape[1] != len(RIVAL_OBS_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(RIVAL_OBS_FEATURE_NAMES)} features, got {X.shape[1]}. "
                f"Feature names: {RIVAL_OBS_FEATURE_NAMES}"
            )

        # --- Weak label generation ---
        # KMeans(3) clusters observable behavioral patterns into energy-management regimes.
        # Cluster with the lowest mean speed Z → LOW energy behavior (slow terminal speed).
        # ponytail: KMeans labels are proxies — no true SoC ground truth exists in FastF1.
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        raw_labels = km.fit_predict(X)
        speed_z_col = RIVAL_OBS_FEATURE_NAMES.index("z_terminal_speed")
        centers = km.cluster_centers_
        order = np.argsort(centers[:, speed_z_col])  # ascending: LOW → MID → HIGH
        label_map = {int(order[i]): label for i, label in enumerate(ENERGY_LABELS)}
        y = np.array([label_map[int(c)] for c in raw_labels])

        # --- Train calibrated classifier ---
        n_cv = max(2, min(5, len(np.unique(y))))
        base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        clf = CalibratedClassifierCV(base, cv=n_cv, method="isotonic")
        clf.fit(X, y)

        # Cross-val on base for diagnostic logging only
        cv_acc = cross_val_score(base, X, y, cv=n_cv, scoring="accuracy")
        logger.info(
            "Rival obs RF CV accuracy: %.3f ± %.3f (n=%d)", cv_acc.mean(), cv_acc.std(), len(X)
        )

        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "feature_names": RIVAL_OBS_FEATURE_NAMES,
            "n_features": len(RIVAL_OBS_FEATURE_NAMES),
            "label_scheme": "KMeans(3) weak supervision — LOW/MEDIUM/HIGH behavioral clusters",
            "ground_truth": "NONE — proxy labels from observable telemetry clustering only",
            "leakage_check": "our_soc_mj excluded by design; enforced by RIVAL_OBS_FEATURE_NAMES",
            "n_samples": int(X.shape[0]),
            "dataset_hash": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16],
            "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
            "cv_accuracy_std": round(float(cv_acc.std()), 4),
        }

        if save:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, _MODEL_PATH)
            _META_PATH.write_text(json.dumps(meta, indent=2))
            logger.info("Rival obs model saved to %s", _MODEL_PATH)

        return cls(clf, meta)
