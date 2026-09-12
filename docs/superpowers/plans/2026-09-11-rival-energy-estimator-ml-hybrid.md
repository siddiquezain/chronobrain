# Rival Energy Estimator — ML Hybrid Observation Model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace synthetic-tuned Gaussian observation likelihoods in the particle filter with a calibrated ML observation model trained on real FastF1 telemetry behavioral patterns, while preserving all architectural guarantees (determinism, no-leakage, Bayesian state tracking, API compatibility).

**Architecture:** A `RivalObservationModel` (calibrated Random Forest trained on KMeans weak-supervision labels from real FastF1 observable features) plugs into the existing `RivalStateEstimator.update()` as an optional obs_model parameter. When present it replaces the four hand-coded Gaussian likelihoods with ML-derived energy-state probabilities mapped onto particle SoC values. The existing Gaussian path becomes the fallback when the artifact is absent, so the pipeline degrades gracefully. `RivalTemporalTracker` extends the baseline with rolling slope and persistence — temporal evidence the particle filter currently lacks. `our_soc_mj` and all internal energy values are structurally excluded from every feature vector and training label.

**Tech Stack:** Python, numpy, scikit-learn (`RandomForestClassifier`, `CalibratedClassifierCV`, `KMeans`), joblib, existing FastF1 service (`app/data/fastf1_service.py`), existing particle filter (`rival_estimator.py`).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rival_estimator.py` | **Modify** | Add `RivalTemporalTracker`; extend `RivalEstimatorConfig` with bucket thresholds; extend `RivalSocEstimate` with `ml_model_active` + `state_probs`; wire `obs_model` into `RivalStateEstimator` |
| `app/ml/rival_observation_model.py` | **Create** | `RivalObservationModel` class: `predict_evidence()`, `load_or_none()`, `train()`, `RIVAL_OBS_FEATURE_NAMES` contract, fallback `uniform_evidence()` |
| `scripts/collect_and_train_rival_model.py` | **Create** | Offline script: loads FastF1 sessions, extracts rival observable features, generates KMeans weak labels, trains and saves model artifact |
| `app/decision/engine.py` | **Modify** | Lazy-load `RivalObservationModel` once; inject into all `RivalStateEstimator` constructions in `_thread_state()` |
| `tests/test_rival_no_leakage.py` | **Create** | Static: `RIVAL_OBS_FEATURE_NAMES` contains no soc/energy terms; dynamic: injecting a CaptureModel verifies the 7-feature vector contains only Z-score-range values |
| `tests/test_rival_observation_model.py` | **Create** | Unit: fallback, load_or_none, feature-count guard, predict_evidence sums to 1 |
| `tests/test_rival_ml_integration.py` | **Create** | Integration: ML stub drives posterior toward LOW; single outlier doesn't collapse; ml_model_active flag; determinism with seed |

---

## Task 1 — `RivalTemporalTracker` + bucket thresholds in `RivalEstimatorConfig`

**Files:**
- Modify: `rival_estimator.py` (after line 119 — end of `RivalObservationBaseline`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_rival_temporal_tracker.py`:

```python
"""Tests for RivalTemporalTracker."""
import numpy as np
import pytest
from rival_estimator import RivalTemporalTracker


def test_speed_persistence_increments_on_negative_z():
    t = RivalTemporalTracker()
    t.update(-1.0, 0.0)
    t.update(-0.5, 0.0)
    assert t.speed_persistence == 2


def test_speed_persistence_resets_on_positive_z():
    t = RivalTemporalTracker()
    t.update(-1.0, 0.0)
    t.update(-1.0, 0.0)
    t.update(0.5, 0.0)   # positive → reset
    assert t.speed_persistence == 0


def test_sector_persistence_increments_on_positive_z():
    """Positive sector Z = slower laps = energy-consistent drain signal."""
    t = RivalTemporalTracker()
    t.update(0.0, 1.0)
    t.update(0.0, 0.8)
    assert t.sector_persistence == 2


def test_speed_slope_negative_when_declining():
    t = RivalTemporalTracker()
    for z in [1.0, 0.5, 0.0, -0.5, -1.0]:
        t.update(z, 0.0)
    assert t.speed_slope < 0


def test_speed_slope_zero_before_two_obs():
    t = RivalTemporalTracker()
    t.update(-1.0, 0.0)
    assert t.speed_slope == 0.0


def test_bucket_low_mj_default():
    from rival_estimator import RivalEstimatorConfig
    cfg = RivalEstimatorConfig()
    assert cfg.bucket_low_mj == 2.5
    assert cfg.bucket_high_mj == 5.5
```

- [ ] **Step 2: Run test to confirm failure**

```
pytest tests/test_rival_temporal_tracker.py -v
```

Expected: ImportError or AttributeError (`RivalTemporalTracker` not found).

- [ ] **Step 3: Add `RivalTemporalTracker` to `rival_estimator.py`**

After line 1 (`from __future__ import annotations`), add `from collections import deque` to the imports block.

After the closing of `RivalObservationBaseline` (after line 119), insert:

```python
@dataclass
class RivalTemporalTracker:
    """
    Rolling Z-score history per rival for temporal feature extraction.
    Call update(z_speed, z_sector) once per lap, after baseline.z_score().
    """
    window_size: int = 5
    _speed_z_history: "deque[float]" = field(init=False, repr=False)
    _speed_persist: int = field(default=0, init=False, repr=False)
    _sector_persist: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._speed_z_history = deque(maxlen=self.window_size)

    def update(self, z_speed: float, z_sector: float) -> None:
        """Update after each z_score() call. Causal — same call-order rule as baseline."""
        self._speed_z_history.append(z_speed)
        self._speed_persist = self._speed_persist + 1 if z_speed < 0.0 else 0
        self._sector_persist = self._sector_persist + 1 if z_sector > 0.0 else 0

    @property
    def speed_slope(self) -> float:
        """Linear trend of speed Z over the last window_size laps. Negative = declining."""
        h = list(self._speed_z_history)
        if len(h) < 2:
            return 0.0
        x = np.arange(len(h), dtype=float)
        return float(np.polyfit(x, h, 1)[0])

    @property
    def speed_persistence(self) -> int:
        return self._speed_persist

    @property
    def sector_persistence(self) -> int:
        return self._sector_persist
```

- [ ] **Step 4: Add `bucket_low_mj` and `bucket_high_mj` to `RivalEstimatorConfig`**

Inside `RivalEstimatorConfig` (after `default_seed: int = 42`), add:

```python
    # Energy-state bucket boundaries for ML likelihood mapping.
    # ponytail: these are model assumptions with no ground-truth calibration.
    #   Adjust if posterior diagnostics show systematic bias on real sessions.
    bucket_low_mj: float = 2.5   # SoC below this → LOW energy bucket
    bucket_high_mj: float = 5.5  # SoC above this → HIGH energy bucket
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/test_rival_temporal_tracker.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rival_estimator.py tests/test_rival_temporal_tracker.py
git commit -m "feat: add RivalTemporalTracker and bucket thresholds to RivalEstimatorConfig"
```

---

## Task 2 — Extend `RivalSocEstimate` + scaffold `RivalStateEstimator` for ML

**Files:**
- Modify: `rival_estimator.py` — `RivalSocEstimate`, `RivalStateEstimator.__init__()`, `estimate()`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rival_temporal_tracker.py` (or a new file `tests/test_rival_estimate_fields.py`):

```python
"""Tests for new RivalSocEstimate fields."""
from rival_estimator import RivalStateEstimator, RivalEstimatorConfig


def _obs():
    class O:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5
    return O()


def test_estimate_has_ml_model_active_field():
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42)
    for _ in range(10):
        est.predict()
        est.update(_obs())
    result = est.estimate()
    assert hasattr(result, "ml_model_active")
    assert result.ml_model_active is False  # no model injected


def test_estimate_has_state_probs_field():
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42)
    for _ in range(10):
        est.predict()
        est.update(_obs())
    result = est.estimate()
    assert hasattr(result, "state_probs")
    assert isinstance(result.state_probs, dict)
    if result.state_probs:  # populated after baseline ready
        assert abs(sum(result.state_probs.values()) - 1.0) < 1e-4


def test_estimator_accepts_obs_model_none():
    """obs_model=None should be accepted without error."""
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42, obs_model=None)
    assert est is not None
```

- [ ] **Step 2: Run test to confirm failure**

```
pytest tests/test_rival_estimate_fields.py -v
```

Expected: AttributeError (`ml_model_active` not on `RivalSocEstimate`).

- [ ] **Step 3: Add fields to `RivalSocEstimate`**

In `RivalSocEstimate` (after `baseline_ready: bool = ...`), add:

```python
    ml_model_active: bool = Field(
        False,
        description="True when ML observation model contributed to the last update cycle."
    )
    state_probs: dict = Field(
        default_factory=dict,
        description="{'low': p, 'medium': p, 'high': p} bucket distribution from posterior mean ± std."
    )
```

- [ ] **Step 4: Scaffold `RivalStateEstimator.__init__()` to accept `obs_model`**

In `RivalStateEstimator.__init__()` (currently at line ~253), add `obs_model=None` parameter and initialise instance vars:

```python
    def __init__(
        self,
        config: Optional[RivalEstimatorConfig] = None,
        seed: Optional[int] = None,
        obs_model=None,  # Optional[RivalObservationModel] — avoid circular import
    ):
        self.config = config or RivalEstimatorConfig()
        _seed = seed if seed is not None else self.config.default_seed
        self._rng = np.random.default_rng(_seed)

        cfg = self.config
        self._particles = self._rng.uniform(cfg.soc_min_mj, cfg.soc_max_mj, cfg.n_particles)
        self._weights = np.ones(cfg.n_particles) / cfg.n_particles
        self._n_observations = 0
        self._last_observation: Optional[dict] = None
        self._baseline = RivalObservationBaseline(min_obs=cfg.min_obs_for_baseline)
        self._temporal = RivalTemporalTracker()   # NEW
        self._obs_model = obs_model               # NEW
        self._ml_active = False                   # NEW — tracks last update cycle
```

- [ ] **Step 5: Update `estimate()` to populate new fields**

In `estimate()`, replace the `return RivalSocEstimate(...)` block with:

```python
        cfg = self.config
        # Compute state_probs from Gaussian summary of the posterior
        state_probs: dict = {}
        if self._n_observations > 0:
            _std = max(reported_std, 1e-6)
            _p_low = _norm_cdf((cfg.bucket_low_mj - mean) / _std)
            _p_high = 1.0 - _norm_cdf((cfg.bucket_high_mj - mean) / _std)
            _p_med = max(0.0, 1.0 - _p_low - _p_high)
            _tot = _p_low + _p_med + _p_high
            state_probs = {
                "low": round(_p_low / _tot, 4),
                "medium": round(_p_med / _tot, 4),
                "high": round(_p_high / _tot, 4),
            }

        return RivalSocEstimate(
            mean_soc_mj=round(mean, 4),
            std_soc_mj=round(reported_std, 4),
            n_observations=n,
            effective_sample_size=round(ess, 1),
            evidence_quality=evidence_quality,
            posterior_health=posterior_health,
            baseline_ready=self._baseline.is_ready,
            ml_model_active=self._ml_active,   # NEW
            state_probs=state_probs,            # NEW
        )
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_rival_estimate_fields.py tests/test_rival_temporal_tracker.py -v
```

Expected: all PASS.

- [ ] **Step 7: Confirm existing tests still pass**

```
pytest test_rival_estimator.py -v
```

Expected: all PASS (no regressions from new fields).

- [ ] **Step 8: Commit**

```bash
git add rival_estimator.py tests/test_rival_estimate_fields.py
git commit -m "feat: add ml_model_active + state_probs to RivalSocEstimate, scaffold obs_model in estimator"
```

---

## Task 3 — `RivalObservationModel`: inference class + feature contract

**Files:**
- Create: `app/ml/rival_observation_model.py`
- Create: `tests/test_rival_observation_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rival_observation_model.py`:

```python
"""Unit tests for RivalObservationModel (inference path only — no FastF1 needed)."""
import numpy as np
import pytest
from unittest.mock import MagicMock


def _fake_clf(proba_row):
    """Return a sklearn-compatible stub with given predict_proba output."""
    clf = MagicMock()
    clf.classes_ = np.array(["HIGH", "LOW", "MEDIUM"])
    clf.n_features_in_ = 7
    clf.predict_proba.return_value = np.array([proba_row])
    return clf


def test_feature_names_length():
    from app.ml.rival_observation_model import RIVAL_OBS_FEATURE_NAMES
    assert len(RIVAL_OBS_FEATURE_NAMES) == 7


def test_feature_names_exclude_soc_terms():
    from app.ml.rival_observation_model import RIVAL_OBS_FEATURE_NAMES
    forbidden = {"soc", "energy_mj", "our_", "internal", "replay"}
    for name in RIVAL_OBS_FEATURE_NAMES:
        for term in forbidden:
            assert term not in name.lower(), f"Forbidden term '{term}' in feature '{name}'"


def test_uniform_evidence_sums_to_one():
    from app.ml.rival_observation_model import RivalObservationModel
    ev = RivalObservationModel.uniform_evidence()
    assert set(ev.keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert abs(sum(ev.values()) - 1.0) < 1e-6
    assert all(abs(v - 1/3) < 1e-6 for v in ev.values())


def test_load_or_none_returns_none_when_absent(tmp_path, monkeypatch):
    import app.ml.rival_observation_model as mod
    monkeypatch.setattr(mod, "_MODEL_PATH", tmp_path / "nonexistent.joblib")
    assert mod.RivalObservationModel.load_or_none() is None


def test_predict_evidence_sums_to_one():
    from app.ml.rival_observation_model import RivalObservationModel, ENERGY_LABELS
    # HIGH=0.1, LOW=0.6, MEDIUM=0.3 (classes_ sorted alphabetically by sklearn)
    model = RivalObservationModel(_fake_clf([0.1, 0.6, 0.3]), {})
    ev = model.predict_evidence(np.zeros((1, 7)))
    assert set(ev.keys()) == set(ENERGY_LABELS)
    assert abs(sum(ev.values()) - 1.0) < 1e-6


def test_predict_evidence_maps_classes_correctly():
    from app.ml.rival_observation_model import RivalObservationModel
    # classes_ order: HIGH=col0, LOW=col1, MEDIUM=col2
    # proba: HIGH=0.1, LOW=0.7, MEDIUM=0.2
    model = RivalObservationModel(_fake_clf([0.1, 0.7, 0.2]), {})
    ev = model.predict_evidence(np.zeros((1, 7)))
    assert ev["LOW"] == pytest.approx(0.7)
    assert ev["HIGH"] == pytest.approx(0.1)
    assert ev["MEDIUM"] == pytest.approx(0.2)


def test_predict_evidence_wrong_feature_count_raises():
    from app.ml.rival_observation_model import RivalObservationModel
    model = RivalObservationModel(_fake_clf([0.33, 0.33, 0.34]), {})
    with pytest.raises(ValueError, match="Expected 7 features"):
        model.predict_evidence(np.zeros((1, 5)))


def test_load_or_none_rejects_feature_count_mismatch(tmp_path, monkeypatch):
    import app.ml.rival_observation_model as mod
    import joblib
    bad_clf = MagicMock()
    bad_clf.n_features_in_ = 99  # wrong count
    bad_clf.classes_ = np.array(["HIGH", "LOW", "MEDIUM"])
    model_path = tmp_path / "bad.joblib"
    joblib.dump(bad_clf, model_path)
    monkeypatch.setattr(mod, "_MODEL_PATH", model_path)
    monkeypatch.setattr(mod, "_META_PATH", tmp_path / "bad.meta.json")
    result = mod.RivalObservationModel.load_or_none()
    assert result is None
```

- [ ] **Step 2: Run tests to confirm failure**

```
pytest tests/test_rival_observation_model.py -v
```

Expected: `ModuleNotFoundError` for `app.ml.rival_observation_model`.

- [ ] **Step 3: Create `app/ml/rival_observation_model.py`**

```python
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

    _UNIFORM = {label: round(1.0 / 3.0, 6) for label in ENERGY_LABELS}

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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_rival_observation_model.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ml/rival_observation_model.py tests/test_rival_observation_model.py
git commit -m "feat: add RivalObservationModel inference class with fallback and feature contract"
```

---

## Task 4 — Add `train()` method to `RivalObservationModel`

**Files:**
- Modify: `app/ml/rival_observation_model.py` (add `train()` classmethod)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rival_observation_model.py`:

```python
def test_train_produces_model_with_correct_feature_count():
    """train() on synthetic feature matrix → model accepts 7-feature vectors."""
    import numpy as np
    from app.ml.rival_observation_model import RivalObservationModel, RIVAL_OBS_FEATURE_NAMES

    rng = np.random.default_rng(0)
    # 150 samples: 3 clear clusters (low, neutral, high speed Z)
    low   = rng.normal([-1.5, 0.2, -0.5, 1.2, -0.3, 3, 2], 0.3, (50, 7))
    mid   = rng.normal([ 0.0, 0.0,  0.0, 0.0,  0.0, 0, 0], 0.3, (50, 7))
    high  = rng.normal([ 1.5,-0.2,  0.5,-1.2,  0.3, 0, 0], 0.3, (50, 7))
    X = np.vstack([low, mid, high])

    model = RivalObservationModel.train(X, save=False)
    assert model is not None

    ev = model.predict_evidence(np.zeros((1, 7)))
    assert set(ev.keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert abs(sum(ev.values()) - 1.0) < 1e-5


def test_train_raises_on_wrong_feature_count():
    import numpy as np
    from app.ml.rival_observation_model import RivalObservationModel
    import pytest
    X = np.zeros((50, 5))  # wrong: 5 features
    with pytest.raises(ValueError, match="Expected 7"):
        RivalObservationModel.train(X, save=False)


def test_train_labels_exclude_soc(monkeypatch):
    """Confirm train() metadata records the no-ground-truth disclosure."""
    import numpy as np
    from app.ml.rival_observation_model import RivalObservationModel

    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal([-1.5, 0.2, -0.5, 1.2, -0.3, 3, 2], 0.3, (50, 7)),
        rng.normal([0.0]*7, 0.3, (50, 7)),
        rng.normal([1.5,-0.2, 0.5,-1.2, 0.3, 0, 0], 0.3, (50, 7)),
    ])
    model = RivalObservationModel.train(X, save=False)
    info = model.model_info()
    assert "NONE" in info.get("ground_truth", "")
    assert "our_soc_mj" in info.get("leakage_check", "").lower() or \
           "excluded" in info.get("leakage_check", "").lower()
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_rival_observation_model.py::test_train_produces_model_with_correct_feature_count -v
```

Expected: AttributeError (`RivalObservationModel` has no `train`).

- [ ] **Step 3: Add `train()` to `app/ml/rival_observation_model.py`**

Add after the `load_or_none()` classmethod:

```python
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
            X: Feature matrix (n_samples, 7). All columns from RIVAL_OBS_FEATURE_NAMES.
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
        base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        clf = CalibratedClassifierCV(base, cv=min(5, len(np.unique(y))), method="isotonic")
        clf.fit(X, y)

        # Cross-val on base (uncalibrated) for diagnostic logging only
        cv_acc = cross_val_score(base, X, y, cv=min(5, len(np.unique(y))), scoring="accuracy")
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
```

- [ ] **Step 4: Run all observation model tests**

```
pytest tests/test_rival_observation_model.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ml/rival_observation_model.py tests/test_rival_observation_model.py
git commit -m "feat: add RivalObservationModel.train() with KMeans weak-label supervision"
```

---

## Task 5 — Offline training script

**Files:**
- Create: `scripts/collect_and_train_rival_model.py`

This script requires FastF1 network access. It is not run during CI. Run it once to produce `app/ml/models/rival_obs_model.joblib`.

- [ ] **Step 1: Check `fastf1_service.load_replay()` signature**

Read `app/data/fastf1_service.py` lines 137–170. Note the exact parameter names and types for `load_replay()`. Update the `SESSIONS` list in the script to match.

- [ ] **Step 2: Create `scripts/collect_and_train_rival_model.py`**

```python
#!/usr/bin/env python
"""
Offline training script for the Rival Energy Observation Model.

Loads real FastF1 sessions, accumulates rival observable features per lap using
RivalObservationBaseline (causal, per-driver), generates KMeans weak labels,
trains a calibrated Random Forest, saves artifact to app/ml/models/.

Usage:
    python scripts/collect_and_train_rival_model.py

Requires FastF1 network access. Run offline; artifact committed separately.
Does NOT use our_soc_mj or any internal energy state.
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

# --- Edit sessions to match available FastF1 data ---
# ponytail: hardcoded list; add argparse if dataset grows beyond ~10 sessions
SESSIONS = [
    {"year": 2024, "event": "Monza",      "session": "R", "our_driver": "VER", "rival_driver": "LEC"},
    {"year": 2024, "event": "Silverstone","session": "R", "our_driver": "NOR", "rival_driver": "VER"},
    {"year": 2024, "event": "Spa",        "session": "R", "our_driver": "HAM", "rival_driver": "LEC"},
]


def _make_obs_adapter(nl):
    """Lightweight adapter: gives RivalObservationBaseline the interface it expects."""
    class _Obs:
        terminal_speed_kmh = nl.rival_terminal_speed_kmh
        clipping_point_fraction = nl.rival_clipping_point_fraction
        corner_exit_accel_g = nl.rival_corner_exit_accel_g
        sector_delta_s = nl.rival_sector_delta_s
    return _Obs()


def collect_features(sessions: list[dict]) -> np.ndarray:
    """
    Load FastF1 sessions. Extract rival Z-score + temporal features per valid lap.
    Returns feature matrix shape (n_rows, 7). No SoC columns anywhere.
    """
    from app.data.fastf1_service import load_replay
    from rival_estimator import RivalObservationBaseline, RivalTemporalTracker

    rows: list[list[float]] = []

    for s in sessions:
        logger.info("Loading %s %s %s ...", s["year"], s["event"], s["session"])
        try:
            # NOTE: load_replay signature — check fastf1_service.py for exact params.
            # Typical: load_replay(year, event, session, our_driver, rival_driver, ...)
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
            baseline.update(obs)

            if not baseline.is_ready:
                continue  # warm-up; Z-scores unreliable before min_obs

            z_sp, z_cl, z_ac, z_se = baseline.z_score(obs)
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
    import numpy as np
    ev = model.predict_evidence(np.zeros((1, 7)))
    logger.info("Evidence for all-zero feature vector: %s", ev)
```

- [ ] **Step 3: Verify script imports cleanly (no network call)**

```
python -c "import scripts.collect_and_train_rival_model"
```

If you get an ImportError about a missing package (e.g., fastf1), that's expected — the import is lazy inside `collect_features()`. If you get a Python syntax error, fix it.

- [ ] **Step 4: Commit**

```bash
git add scripts/collect_and_train_rival_model.py
git commit -m "feat: add offline training script for rival observation model"
```

---

## Task 6 — Wire ML likelihoods into `RivalStateEstimator.update()`

**Files:**
- Modify: `rival_estimator.py` — `update()` method

This task integrates `RivalTemporalTracker` (Task 1) and `obs_model` (Task 2) into the active update path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rival_estimate_fields.py`:

```python
def test_ml_model_called_when_injected():
    """When obs_model is injected, update() must call predict_evidence()."""
    from rival_estimator import RivalStateEstimator, RivalEstimatorConfig

    calls = []

    class TrackingModel:
        def predict_evidence(self, fv):
            calls.append(fv.shape)
            return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}

    est = RivalStateEstimator(
        config=RivalEstimatorConfig(), seed=42, obs_model=TrackingModel()
    )

    class Obs:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5

    # Run past baseline warm-up (min_obs_for_baseline = 5)
    for _ in range(10):
        est.predict()
        est.update(Obs())

    assert len(calls) > 0, "predict_evidence() was never called"
    assert calls[-1] == (1, 7), f"Expected feature shape (1, 7), got {calls[-1]}"
    assert est.estimate().ml_model_active is True


def test_ml_inactive_before_baseline_ready():
    """ML model must not be called during the warm-up period (< min_obs_for_baseline)."""
    from rival_estimator import RivalStateEstimator, RivalEstimatorConfig

    calls = []

    class TrackingModel:
        def predict_evidence(self, fv):
            calls.append(True)
            return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}

    est = RivalStateEstimator(
        config=RivalEstimatorConfig(), seed=42, obs_model=TrackingModel()
    )

    class Obs:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5

    # Only 3 observations — below min_obs_for_baseline=5
    for _ in range(3):
        est.predict()
        est.update(Obs())

    assert len(calls) == 0, "ML model called before baseline warm-up complete"
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_rival_estimate_fields.py::test_ml_model_called_when_injected -v
```

Expected: FAIL (`assert len(calls) > 0` fails — `predict_evidence` not yet wired).

- [ ] **Step 3: Modify `update()` in `rival_estimator.py`**

Replace the `if self._baseline.is_ready:` block (currently lines ~309–318) with:

```python
        if self._baseline.is_ready:
            # Z-score model: normalize against rival's own running distribution
            z_sp, z_cl, z_ac, z_se = self._baseline.z_score(observation)
            # Update temporal tracker after z_scores are available (causal)
            self._temporal.update(z_sp, z_se)

            if self._obs_model is not None:
                # ML path: energy-state evidence → piecewise bucket particle weights
                features = np.array([[
                    z_sp, z_cl, z_ac, z_se,
                    self._temporal.speed_slope,
                    float(self._temporal.speed_persistence),
                    float(self._temporal.sector_persistence),
                ]])
                evidence = self._obs_model.predict_evidence(features)
                # Map each particle's SoC → bucket probability
                # ponytail: piecewise constant likelihood; bucket-boundary discontinuities
                #   are absorbed by roughening and n_particles=1000. Upgrade to
                #   soft-bucket blending if posterior shows systematic boundary artifacts.
                log_w = np.log(np.clip(
                    np.where(
                        soc < cfg.bucket_low_mj,
                        evidence.get("LOW", 1/3),
                        np.where(
                            soc < cfg.bucket_high_mj,
                            evidence.get("MEDIUM", 1/3),
                            evidence.get("HIGH", 1/3),
                        ),
                    ),
                    1e-10,
                    None,
                ))
                self._ml_active = True
            else:
                # Gaussian fallback: existing hand-coded Z-score observation model
                noise = cfg.z_observation_noise
                log_w = (
                    log_gaussian(z_sp, cfg.expected_speed_z(soc_fraction), noise)
                    + log_gaussian(z_cl, cfg.expected_clip_z(soc_fraction), noise)
                    + log_gaussian(z_ac, cfg.expected_accel_z(soc_fraction), noise)
                    + log_gaussian(z_se, cfg.expected_sector_z(soc_fraction), noise)
                )
                self._ml_active = False
```

The `else:` (pre-baseline fallback block) and everything from `log_w -= log_w.max()` onward remains unchanged.

- [ ] **Step 4: Run all estimator tests**

```
pytest tests/test_rival_estimate_fields.py test_rival_estimator.py -v
```

Expected: all PASS (new tests pass, no regressions).

- [ ] **Step 5: Commit**

```bash
git add rival_estimator.py tests/test_rival_estimate_fields.py
git commit -m "feat: wire ML observation likelihoods into RivalStateEstimator.update()"
```

---

## Task 7 — Engine: lazy-load model, inject into estimators

**Files:**
- Modify: `app/decision/engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rival_estimate_fields.py`:

```python
def test_engine_thread_state_injects_obs_model(monkeypatch):
    """
    _thread_state should pass the cached obs_model to each RivalStateEstimator.
    Verify by monkeypatching _get_rival_obs_model to return a sentinel.
    """
    import app.decision.engine as eng

    sentinel = object()
    created_with = []

    original_init = eng.RivalStateEstimator.__init__

    def tracking_init(self, config=None, seed=None, obs_model=None):
        created_with.append(obs_model)
        original_init(self, config=config, seed=seed, obs_model=obs_model)

    monkeypatch.setattr(eng, "_get_rival_obs_model", lambda: sentinel)
    monkeypatch.setattr(eng.RivalStateEstimator, "__init__", tracking_init)

    # Minimal lap list to trigger _thread_state
    from app.data.samples import NormalizedLap
    from app.decision.config import DecisionConfig
    from app.decision.context import DecisionContext

    nl = NormalizedLap(
        lap=1, total_laps=10, data_mode="SYNTHETIC",
        our_speed_kmh=280.0,
    )
    ctx = DecisionContext(config=DecisionConfig(), lap_history=[nl])
    eng._thread_state(ctx)

    assert any(m is sentinel for m in created_with), (
        "No RivalStateEstimator was created with the sentinel obs_model"
    )
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_rival_estimate_fields.py::test_engine_thread_state_injects_obs_model -v
```

Expected: AttributeError (no `_get_rival_obs_model` in engine module yet).

- [ ] **Step 3: Add lazy loader to `engine.py`**

After the existing imports in `engine.py` (after line ~50), add:

```python
from app.ml.rival_observation_model import RivalObservationModel as _RivalObsModel

_rival_obs_model: Optional["_RivalObsModel"] = None
_rival_obs_model_loaded: bool = False


def _get_rival_obs_model() -> Optional["_RivalObsModel"]:
    """Load rival observation model once per process; None = Gaussian fallback."""
    global _rival_obs_model, _rival_obs_model_loaded
    if not _rival_obs_model_loaded:
        _rival_obs_model = _RivalObsModel.load_or_none()
        _rival_obs_model_loaded = True
    return _rival_obs_model
```

- [ ] **Step 4: Inject into `_thread_state()`**

In `_thread_state()`, replace both `RivalStateEstimator(...)` constructor calls:

```python
    obs_model = _get_rival_obs_model()   # NEW — loaded once, shared across all estimators

    default_est = RivalStateEstimator(
        config=cfg.rival_config(), seed=cfg.seed, obs_model=obs_model   # obs_model added
    )
    estimators: dict[str, RivalStateEstimator] = {}

    def _est_for(driver: Optional[str]) -> RivalStateEstimator:
        if driver is None:
            return default_est
        if driver not in estimators:
            estimators[driver] = RivalStateEstimator(
                config=cfg.rival_config(),
                seed=cfg.seed + _driver_seed_offset(driver),
                obs_model=obs_model,   # obs_model added
            )
        return estimators[driver]
```

- [ ] **Step 5: Run the new test + engine integration tests**

```
pytest tests/test_rival_estimate_fields.py::test_engine_thread_state_injects_obs_model \
       tests/test_v1_decision_api.py tests/test_decision_golden.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/decision/engine.py tests/test_rival_estimate_fields.py
git commit -m "feat: lazy-load RivalObservationModel in engine and inject into all rival estimators"
```

---

## Task 8 — No-leakage regression tests

**Files:**
- Create: `tests/test_rival_no_leakage.py`

- [ ] **Step 1: Create the test file**

```python
"""
Regression suite: our_soc_mj must never enter the rival inference path.

Two levels of verification:
  1. Static — RIVAL_OBS_FEATURE_NAMES contains no forbidden energy/soc terms.
  2. Dynamic — Injecting a CaptureModel and verifying the feature vector
     (a) has the correct length and (b) contains only Z-score-range values,
     not raw SoC values (which are 0–9 MJ and would produce values >> 5 in
     the feature vector).
"""
import numpy as np
import pytest
from rival_estimator import RivalStateEstimator, RivalEstimatorConfig


# --- Static leakage check ---

def test_feature_names_exclude_soc_and_energy_terms():
    from app.ml.rival_observation_model import RIVAL_OBS_FEATURE_NAMES
    forbidden_terms = {"soc", "energy_mj", "our_", "internal", "replay", "modeled"}
    for name in RIVAL_OBS_FEATURE_NAMES:
        for term in forbidden_terms:
            assert term not in name.lower(), (
                f"Forbidden term '{term}' found in rival obs feature '{name}'"
            )


# --- Dynamic leakage check ---

class CaptureModel:
    """Stub obs_model that records every feature vector it receives."""
    def __init__(self):
        self.captured = []

    def predict_evidence(self, fv: np.ndarray) -> dict:
        self.captured.append(fv.copy())
        return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}


def _warmup_obs():
    class O:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5
    return O()


def test_ml_feature_vector_length_is_seven():
    cap = CaptureModel()
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42, obs_model=cap)
    for _ in range(10):
        est.predict()
        est.update(_warmup_obs())
    assert len(cap.captured) > 0, "CaptureModel was never called"
    assert cap.captured[-1].shape == (1, 7), (
        f"Feature vector shape wrong: {cap.captured[-1].shape}"
    )


def test_ml_feature_values_are_z_score_range():
    """Z-score features should be bounded (|z| < 20 under normal racing conditions).
    Raw SoC values (0–9 MJ) would not satisfy this for the first 4 features if leaked."""
    cap = CaptureModel()
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42, obs_model=cap)
    for _ in range(15):
        est.predict()
        est.update(_warmup_obs())
    assert len(cap.captured) > 0
    fv = cap.captured[-1][0]  # (7,) array
    z_features = fv[:4]  # first 4 are Z-scores
    assert all(abs(v) < 20.0 for v in z_features), (
        f"Z-score features out of expected range (possible SoC leakage): {z_features}"
    )


def test_update_signature_has_no_soc_parameter():
    """RivalStateEstimator.update() must not accept our_soc_mj."""
    import inspect
    sig = inspect.signature(RivalStateEstimator.update)
    params = list(sig.parameters.keys())
    assert "our_soc_mj" not in params
    assert "soc_mj" not in params


def test_posterior_invariant_to_our_soc_mj_change():
    """
    Running two identical sequences of rival observations but with different
    our_soc_mj values on the NormalizedLap must produce identical posteriors.
    The estimator never receives our_soc_mj, so this should always hold.
    """
    from app.data.samples import NormalizedLap
    from app.data.normalizer import to_rival_observation

    def make_lap(lap: int, our_soc: float) -> NormalizedLap:
        return NormalizedLap(
            lap=lap,
            total_laps=50,
            data_mode="REPLAY",
            our_speed_kmh=280.0,
            our_soc_mj=our_soc,  # varies between the two runs
            rival_terminal_speed_kmh=315.0,
            rival_clipping_point_fraction=0.72,
            rival_corner_exit_accel_g=0.95,
            rival_sector_delta_s=-0.3,
        )

    def run(our_soc: float) -> float:
        est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42)
        for i in range(15):
            obs = to_rival_observation(make_lap(i + 1, our_soc))
            if obs is not None:
                est.predict()
                est.update(obs)
        return est.estimate().mean_soc_mj

    mean_low_soc = run(0.5)
    mean_high_soc = run(8.5)
    assert abs(mean_low_soc - mean_high_soc) < 1e-6, (
        f"Posterior differs with our_soc_mj: {mean_low_soc:.4f} vs {mean_high_soc:.4f} "
        "— possible leakage via to_rival_observation()"
    )
```

- [ ] **Step 2: Run the no-leakage suite**

```
pytest tests/test_rival_no_leakage.py -v
```

Expected: all PASS. If `test_posterior_invariant_to_our_soc_mj_change` fails, inspect `to_rival_observation()` in `app/data/normalizer.py` — it must not include `our_soc_mj` in the returned `RivalObservation`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rival_no_leakage.py
git commit -m "test: add no-leakage regression suite for rival estimator"
```

---

## Task 9 — Integration tests: temporal consistency and ML path

**Files:**
- Create: `tests/test_rival_ml_integration.py`

- [ ] **Step 1: Create the test file**

```python
"""
Integration tests for ML observation model → particle filter.
All tests use stub ML models — no FastF1 download, no model artifact required.
"""
import numpy as np
import pytest
from rival_estimator import RivalStateEstimator, RivalEstimatorConfig


def _obs(speed=300.0, clip=0.75, accel=0.90, sector=0.5):
    class O:
        terminal_speed_kmh = speed
        clipping_point_fraction = clip
        corner_exit_accel_g = accel
        sector_delta_s = sector
    return O()


class LowEnergyModel:
    """Always signals low energy — drives particles toward LOW bucket."""
    def predict_evidence(self, fv):
        return {"LOW": 0.80, "MEDIUM": 0.15, "HIGH": 0.05}


class HighEnergyModel:
    """Always signals high energy — drives particles toward HIGH bucket."""
    def predict_evidence(self, fv):
        return {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.80}


class NeutralModel:
    """Uniform — no evidence, posterior stays near prior."""
    def predict_evidence(self, fv):
        return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}


def _run(model, n_laps=20, obs_fn=None, seed=42):
    """Warm up 6 laps (uninformative fallback), then run n_laps with the model."""
    cfg = RivalEstimatorConfig()
    est = RivalStateEstimator(config=cfg, seed=seed, obs_model=model)
    warmup = _obs()  # baseline warm-up
    for _ in range(6):
        est.predict()
        est.update(warmup)
    main_obs = obs_fn() if obs_fn else _obs()
    for _ in range(n_laps):
        est.predict()
        est.update(main_obs)
    return est.estimate(), cfg


def test_low_evidence_shifts_posterior_below_neutral():
    low_est, cfg = _run(LowEnergyModel())
    neu_est, _   = _run(NeutralModel())
    assert low_est.mean_soc_mj < neu_est.mean_soc_mj, (
        f"Low-energy evidence didn't shift mean below neutral: "
        f"{low_est.mean_soc_mj:.2f} vs {neu_est.mean_soc_mj:.2f}"
    )


def test_high_evidence_shifts_posterior_above_neutral():
    hi_est,  cfg = _run(HighEnergyModel())
    neu_est, _   = _run(NeutralModel())
    assert hi_est.mean_soc_mj > neu_est.mean_soc_mj, (
        f"High-energy evidence didn't shift mean above neutral: "
        f"{hi_est.mean_soc_mj:.2f} vs {neu_est.mean_soc_mj:.2f}"
    )


def test_low_evidence_bucket_dist_skews_low():
    est, cfg = _run(LowEnergyModel())
    dist = est.bucket_distribution(cfg.bucket_low_mj, cfg.bucket_high_mj)
    assert dist["low"] > dist["high"], (
        f"Expected P(LOW)>P(HIGH) after low-energy evidence: {dist}"
    )


def test_high_evidence_bucket_dist_skews_high():
    est, cfg = _run(HighEnergyModel())
    dist = est.bucket_distribution(cfg.bucket_low_mj, cfg.bucket_high_mj)
    assert dist["high"] > dist["low"], (
        f"Expected P(HIGH)>P(LOW) after high-energy evidence: {dist}"
    )


def test_ml_model_active_flag_set():
    est, _ = _run(LowEnergyModel())
    assert est.ml_model_active is True


def test_ml_inactive_without_model():
    est, _ = _run(None)  # no obs_model
    assert est.ml_model_active is False


def test_single_outlier_does_not_collapse_posterior():
    """One extreme observation must not cause a > 3 MJ mean shift (roughening guard)."""
    cfg = RivalEstimatorConfig()
    est = RivalStateEstimator(config=cfg, seed=42, obs_model=NeutralModel())
    neutral = _obs()
    for _ in range(20):
        est.predict()
        est.update(neutral)
    baseline_mean = est.estimate().mean_soc_mj

    class ExtremeModel:
        def predict_evidence(self, fv):
            return {"LOW": 0.99, "MEDIUM": 0.005, "HIGH": 0.005}

    est._obs_model = ExtremeModel()
    est.predict()
    est.update(_obs(speed=220.0, sector=5.0))  # extreme observation
    post_mean = est.estimate().mean_soc_mj
    assert abs(post_mean - baseline_mean) < 3.0, (
        f"Single outlier caused {abs(post_mean - baseline_mean):.2f} MJ shift — "
        "check roughening_std_mj and min_reported_std_mj config"
    )


def test_posterior_deterministic_given_seed():
    """Same seed + same model + same observations → identical posterior."""
    def run_once():
        est, _ = _run(LowEnergyModel(), seed=99)
        return est.mean_soc_mj

    assert run_once() == run_once(), "Non-deterministic posterior detected"


def test_state_probs_populated_after_warmup():
    est, _ = _run(LowEnergyModel())
    assert est.state_probs, "state_probs dict is empty after warmup"
    assert abs(sum(est.state_probs.values()) - 1.0) < 1e-4
```

- [ ] **Step 2: Run integration tests**

```
pytest tests/test_rival_ml_integration.py -v
```

Expected: all PASS. If `test_low_evidence_shifts_posterior_below_neutral` fails, check that `bucket_low_mj` (2.5 MJ) and `bucket_high_mj` (5.5 MJ) are reasonable relative to `soc_max_mj` (9.0 MJ). The prior is uniform over [0, 9], so the prior mean ≈ 4.5 MJ — low evidence should pull below this.

- [ ] **Step 3: Run the full rival estimator test suite to check for regressions**

```
pytest test_rival_estimator.py tests/test_rival_calibration.py tests/test_rival_validation.py \
       tests/test_rival_causality.py -v
```

Expected: all PASS. The ML path is only active when `obs_model` is injected — existing tests that don't inject it use the Gaussian fallback unchanged.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rival_ml_integration.py
git commit -m "test: add ML integration tests for temporal consistency and posterior shifting"
```

---

## Task 10 — Full suite verification + no-artifact CI safety

- [ ] **Step 1: Confirm model artifact absent in repo doesn't break CI**

```bash
ls app/ml/models/rival_obs_model.joblib 2>/dev/null && echo "EXISTS" || echo "ABSENT"
```

If ABSENT: the Gaussian fallback path is active everywhere. Run the full test suite:

```
pytest -x -q
```

Expected: all tests pass. No test should require `rival_obs_model.joblib` to exist because `load_or_none()` returns None gracefully.

- [ ] **Step 2: Verify decision golden tests pass**

```
pytest tests/test_decision_golden.py tests/test_decision_determinism.py -v
```

Expected: all PASS. Same seed → same snapshot (determinism preserved).

- [ ] **Step 3: Verify API contract unchanged**

```
pytest tests/test_v1_decision_api.py -v
```

Expected: all PASS. If any test checks the response schema and fails on `ml_model_active` / `state_probs`, update the golden snapshot — these are additive fields, not breaking changes.

- [ ] **Step 4: Run the no-leakage suite one final time**

```
pytest tests/test_rival_no_leakage.py -v
```

Expected: all PASS.

- [ ] **Step 5: Final commit**

```bash
git add -p  # review and stage any remaining unstaged changes
git commit -m "test: full suite verification — ML hybrid rival estimator complete"
```

---

## Self-Review: Spec Coverage Check

| Spec Section | Covered? | Where |
|---|---|---|
| §2 Hybrid architecture | ✅ | Tasks 3, 4, 6 |
| §5 No leakage of `our_soc_mj` | ✅ | Task 8 (static + dynamic + functional) |
| §6 Obs model trained on synthetic → fails on real | ✅ | Task 5 (real FastF1 training) |
| §7 Physics as guardrail | ✅ | Existing process-noise clip + bucket bounds; `roughening` prevents collapse |
| §8 ML generates likelihood, not decision | ✅ | ML → particle weights only; decision engine unchanged |
| §9 Real FastF1 as primary training source | ✅ | Task 5 |
| §10 No fabricated "true rival SoC" labels | ✅ | KMeans weak labels; metadata documents "NONE" ground truth |
| §12 Temporal evidence | ✅ | Task 1 (`RivalTemporalTracker`), Task 6 (temporal features in update()) |
| §13 Bayesian tracker retained | ✅ | Particle filter unchanged; ML plugs in as likelihood |
| §16 Uncertainty first-class | ✅ | `state_probs`, `ml_model_active`, existing `evidence_quality` / `posterior_health` |
| §17 UI representation | ✅ | `state_probs` + `posterior_health` exposed in `RivalSocEstimate` |
| §20 Evaluation diagnostics | ✅ | ESS, observation quality, state probs all in `estimate()` output |
| §22 Determinism | ✅ | Task 10 determinism test; seed controls both PF and RF |
| §23 Feature contract + no silent retrain | ✅ | `RIVAL_OBS_FEATURE_NAMES` frozen; training is offline script only |
| §24 Separate from overtake ML | ✅ | New file `rival_observation_model.py`; overtake `predict.py` untouched |
| §30 API unchanged | ✅ | No route changes; new fields additive in `RivalSocEstimate` |
| §32 Inspect before changing | ✅ | Full read of `rival_estimator.py`, `engine.py`, `samples.py` before plan |

**Gaps / known limitations (acceptable per spec §20, §21):**
- Bucket boundary thresholds (2.5/5.5 MJ) are model assumptions; no ground-truth calibration available until controlled evaluation with known hidden state
- KMeans labels are behavioral proxies; model accuracy metrics measure cluster separation, not true SoC recovery
- Training script requires FastF1 network access; initial artifact must be generated and committed separately from CI
