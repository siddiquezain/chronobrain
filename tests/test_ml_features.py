"""
Regression: the overtake-success model must be fed genuinely-computed features,
not hardcoded placeholders, and inference must stay deterministic.
"""

import numpy as np
import pytest

from app.ml.dataset import generate
from app.ml.features import FEATURE_NAMES, build_features
from app.ml.predict import model_info, predict_probability


def test_feature_set_is_the_expected_six_real_features():
    assert FEATURE_NAMES == [
        "gap_to_car_ahead_s",
        "gap_trend_s_per_lap",
        "our_soc_mj",
        "our_speed_kmh",
        "overtake_mode_eligible",
        "rival_terminal_speed_kmh",
    ]
    assert "drs_available" not in FEATURE_NAMES
    # the fabricated ones are gone
    for gone in ("slipstream_factor", "tyre_age_laps", "straight_distance_m", "closing_speed_mps"):
        assert gone not in FEATURE_NAMES


def test_build_features_shape_and_missing_handling():
    f = build_features(
        gap_to_car_ahead_s=None, gap_trend_s_per_lap=None,
        our_soc_mj=5.0, our_speed_kmh=300.0, overtake_mode_eligible=None,
        rival_terminal_speed_kmh=None,
    )
    assert f.shape == (1, 6)
    assert f[0, 0] > 0            # gap default, not 0
    assert f[0, 1] == 0.0        # trend unknown -> neutral
    assert f[0, 4] == 0.0        # overtake_mode_eligible None -> 0


def test_dataset_matches_feature_width():
    X, y = generate()
    assert X.shape[1] == len(FEATURE_NAMES)
    assert set(np.unique(y)).issubset({0, 1})


def test_metadata_records_the_feature_contract():
    info = model_info()
    assert info.get("feature_names") == FEATURE_NAMES
    assert info.get("n_features", len(FEATURE_NAMES)) == 6


def test_prediction_is_deterministic_and_bounded():
    f = build_features(0.5, -0.3, 6.5, 305.0, True, 300.0)
    p1 = predict_probability(f)
    p2 = predict_probability(f)
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_closing_gap_and_energy_raise_the_probability():
    far = build_features(3.0, 0.2, 2.0, 280.0, False, 335.0)
    close = build_features(0.4, -0.4, 8.0, 320.0, True, 300.0)
    assert predict_probability(close) > predict_probability(far)


def test_wrong_feature_width_raises():
    with pytest.raises(ValueError):
        predict_probability(np.zeros((1, 8)))
