"""Shared pytest fixtures for the ChronoPace intelligence engine."""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def default_seed():
    return 42


@pytest.fixture(scope="session")
def rng(default_seed):
    return np.random.default_rng(default_seed)
