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
    # classes_ sorted alphabetically: HIGH=col0, LOW=col1, MEDIUM=col2
    model = RivalObservationModel(_fake_clf([0.1, 0.6, 0.3]), {})
    ev = model.predict_evidence(np.zeros((1, 7)))
    assert set(ev.keys()) == set(ENERGY_LABELS)
    assert abs(sum(ev.values()) - 1.0) < 1e-6


def test_predict_evidence_maps_classes_correctly():
    from app.ml.rival_observation_model import RivalObservationModel
    # classes_: HIGH=col0, LOW=col1, MEDIUM=col2
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


class _BadClf:
    """Picklable stub with wrong feature count."""
    n_features_in_ = 99
    classes_ = np.array(["HIGH", "LOW", "MEDIUM"])


def test_load_or_none_rejects_feature_count_mismatch(tmp_path, monkeypatch):
    import app.ml.rival_observation_model as mod
    import joblib
    model_path = tmp_path / "bad.joblib"
    joblib.dump(_BadClf(), model_path)
    monkeypatch.setattr(mod, "_MODEL_PATH", model_path)
    monkeypatch.setattr(mod, "_META_PATH", tmp_path / "bad.meta.json")
    result = mod.RivalObservationModel.load_or_none()
    assert result is None
