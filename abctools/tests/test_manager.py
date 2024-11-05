import polars as pl
import pytest
from polars.testing import assert_frame_equal

# from abctools import abc_methods, manager, plot_utils, toy_model
from abctools import manager
from abctools.abc_classes import SimulationBundle

# from scipy.stats import uniform


## ======================================#
## Seed assignment---
## ======================================#


def initialize_baseline():
    return {"x": 100}


@pytest.fixture
def fixed_seed():
    yield 0


@pytest.fixture
def empty_bundle_df(fixed_seed):
    empty_bundle = manager.call_experiment(
        "empty_config",
        "generate_seed",
        project_seed=fixed_seed,
        initializer=initialize_baseline,
    )
    yield empty_bundle.inputs


def test_empty_bundle(empty_bundle_df, fixed_seed):
    manual_bundle_df = pl.DataFrame(
        {"simulation": 0, "randomSeed": fixed_seed}
    )

    assert_frame_equal(empty_bundle_df, manual_bundle_df)
