import pytest
import polars as pl
from scipy.stats import uniform

from abctools import abc_methods, plot_utils, toy_model, manager
from abctools.abc_classes import SimulationBundle

## ======================================#
## Seed assignment---
## ======================================#

@pytest.fixture
def initialize_baseline():
    return {"x": 100}

@pytest.fixture
def seed():
    return 0

@pytest.fixture
def empty_bundle(seed, initialize_baseline):
    empty_bundle = manager.call_experiment("empty_config", 
                                        "generate_seed", 
                                        project_seed=seed,
                                        initializer = initialize_baseline)
    return empty_bundle.inputs

def test_empty_bundle(empty_bundle):
    manual_bundle = pl.DataFrame({"simulation": 0, "randomSeed": seed})
    pl.testing.assert_frame_equal(empty_bundle, manual_bundle)
