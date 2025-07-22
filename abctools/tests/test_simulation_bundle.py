import polars as pl
from abctools.abc_classes import SimulationBundle

def test_accept_stochastic():
    inputs = pl.DataFrame({
        "simulation": range(10),
        "seed": [0] * 10
    }).with_columns(
        (pl.col("simulation") * 0.5).floor().alias("parameter1")
    )

    bundle = SimulationBundle(
        inputs=inputs,
        step_number=0,
        baseline_params={},
        seed_variable_name="seed"
    )

    bundle.distances = pl.DataFrame({
        "simulation": range(10),
    }).with_columns(
        (pl.col("simulation") * 0.1).alias("distance")
    )
    bundle.accept_stochastic(tolerance=0.4)
    accepted_totals = bundle.accepted.select(
        pl.sum("acceptance_weight", "accept_bool")
    )

    assert accepted_totals["acceptance_weight"].item() == accepted_totals["accept_bool"].item()