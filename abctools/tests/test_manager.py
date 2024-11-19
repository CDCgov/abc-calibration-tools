import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

# from abctools import abc_methods, manager, plot_utils, toy_model
from abctools import abc_manager
from abctools.abc_classes import SimulationBundle

# from scipy.stats import uniform

## ======================================#
## Bundle creation tests---
## ======================================#


def initialize_baseline():
    return {"x": 100}


@pytest.fixture
def fixed_seed():
    yield 0


@pytest.fixture
def empty_bundle(fixed_seed):
    empty_bundle = abc_manager.call_experiment(
        config="empty_config",
        experiment_mode="generate_seed",
        project_seed=fixed_seed,
        initializer=initialize_baseline,
    )
    yield empty_bundle


@pytest.fixture
def manual_bundle(fixed_seed):
    df = pl.DataFrame({"simulation": 0, "randomSeed": fixed_seed})

    bundle = SimulationBundle(
        inputs=df, step_number=0, baseline_params=initialize_baseline
    )
    yield bundle


def test_bundle_generation(empty_bundle):
    assert isinstance(empty_bundle, SimulationBundle)


def test_empty_bundle(empty_bundle, manual_bundle):
    assert_frame_equal(empty_bundle.inputs, manual_bundle.inputs)


def test_passed_bundle(fixed_seed, manual_bundle):
    original_status = manual_bundle.status
    copied_bundle = abc_manager.call_experiment(
        config="empty_config",
        experiment_mode="generate_seed",
        project_seed=fixed_seed,
        bundle=manual_bundle,
    )

    assert_frame_equal(copied_bundle.inputs, manual_bundle.inputs)
    assert copied_bundle.baseline_params == manual_bundle.baseline_params
    assert copied_bundle.step_number == manual_bundle.step_number
    assert copied_bundle.status != original_status
    assert manual_bundle.status == "duplicated"


## ======================================#
## Seed generation tests---
## ======================================#


@pytest.fixture
def empty_random_bundle():
    bundle = abc_manager.call_experiment(
        config="empty_config",
        experiment_mode="generate_seed",
        project_seed=None,
        initializer=initialize_baseline,
    )
    yield bundle


@pytest.fixture
def random_seed(empty_random_bundle):
    yield empty_random_bundle.inputs["randomSeed"]


def test_seed_notfixed(random_seed, fixed_seed):
    assert random_seed[0] != fixed_seed


def test_randomseed_pass(random_seed):
    fixed_bundle = abc_manager.call_experiment(
        config="empty_config",
        experiment_mode="generate_seed",
        project_seed=random_seed[0],
        initializer=initialize_baseline,
    )

    assert_series_equal(fixed_bundle.inputs["randomSeed"], random_seed)
