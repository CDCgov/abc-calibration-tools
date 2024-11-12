import polars as pl
from gcm_python_wrappers import wrappers

from abctools import abc_methods, manager, toy_model
from abctools.abc_classes import SimulationBundle

## ======================================#
## User defined model functions---
## ======================================#

prior_sampler_distros = {
    "averageContactsPerDay": abc_methods.get_truncated_normal(
        mean=2, sd=1, low=0.01, upp=25
    ),
    "infectiousPeriod": abc_methods.get_truncated_normal(
        mean=4, sd=2.5, low=0.01, upp=25
    ),
}


# Simulation setup compatible with GCM baseScenario YAML files
def run_toy_model(params: dict):
    """Runs toy stochastic SIR model"""

    # Get baseScenario params out of sub-dictionary
    compressed_params = params["baseScenario"]
    for par, value in params.items():
        if par != "baseScenario":
            compressed_params[par] = value

    params = compressed_params

    # Configure compartments
    N = params["population"]
    I0 = int(params["population"] * params["initialPrevalence"])
    S0 = N - I0
    R0 = 0
    initial_state = (S0, I0, R0)

    # Configure time
    tmax = params["totalDays"]

    # Configure rates
    beta = 1 / params["averageContactsPerDay"]
    gamma = 1 / params["infectiousPeriod"]
    rates = (beta, gamma)

    # Configure random seed
    if "randomSeed" in params:
        random_seed = params["randomSeed"]
    else:
        random_seed = None

    # Run model
    # Note: I am not using 'susceptible' for anything in this example
    time_points, _, infected, recovered = toy_model.ctmc_gillespie_model(
        initial_state, rates, tmax, t=0, random_seed=random_seed
    )

    # Format as dataframe
    results_df = pl.DataFrame(
        {"time": time_points, "infected": infected, "recovered": recovered},
        schema=[
            ("time", pl.Float32),
            ("infected", pl.Int64),
            ("recovered", pl.Int64),
        ],
    )
    return results_df


# Simulation running wrapper to iterate over sim-specific parameters
def sim_runner(input_bundle: SimulationBundle) -> SimulationBundle:
    """
    Function to run the simulation parameters within a bundle and return a bundle with results
    """

    results_dict = {}
    for param_set in input_bundle.full_params_df.rows(named=True):
        results_dict[param_set["simulation"]] = run_toy_model(param_set)

    input_bundle.results = results_dict

    return input_bundle


# Summary function to return peak time and total number infected
def summarize_sims(df: pl.DataFrame) -> pl.DataFrame:
    """User-defined function to calculate infection metrics, in this case time to peak infection and total infected"""
    # Find the time of peak infection (first instance, if multiple)
    time_to_peak_infection = (
        df.sort("infected", descending=True).select(pl.first("time"))
    ).item()

    # Calculate total infected by taking the maximum value from the 'recovered' column
    total_infected = df.get_column("recovered").max()

    # return pl.DataFrame with metrics
    metrics = pl.DataFrame(
        {
            "time_to_peak_infection": time_to_peak_infection,
            "total_infected": total_infected,
        }
    )
    return metrics


# Difference function for distance calculation
def distance_difference(df: pl.DataFrame, target_data: pl.DataFrame) -> float:
    """User-defined distance function to calculate the difference between simulation and target data"""
    # Concatenate data frames and caclualte difference
    diff = df - target_data

    sum_distance = sum(diff / target_data)

    return sum_distance


## ======================================#
## Setup environment---
## ======================================#
dir = "./abctools/examples/abc-smc"
wrappers.delete_experiment_items(dir, "", "")

tolerance = [
    500,
    250,
    100,
    50,
    20,
    10000,
]

## ======================================#
## Generate target---
## ======================================#

target_seed = 1234
target_bundle = manager.call_experiment(
    config="./abctools/examples/abc-smc/config.yaml",
    experiment_mode="generate_target",
    write=["simulations", "summaries"],
    project_seed=target_seed,
    wd=dir,
    runner=sim_runner,
    summarizer=summarize_sims,
)

## ======================================#
## Generate first bundle---
## ======================================#
n_init = 1000
seed = 12345


sample_bundle = manager.call_experiment(
    config="./abctools/examples/abc-smc/config.yaml",
    experiment_mode="draw_samples",
    write=["simulations", "summaries"],
    project_seed=seed,
    wd=dir,
    random_sampler=prior_sampler_distros,
    sampler_method="sobol",
    runner=sim_runner,
    summarizer=summarize_sims,
    replicates=n_init,
)

sample_bundle.calculate_distances(
    target_data=target_bundle.summary_metrics[0],
    distance_function=distance_difference,
    use_summary_metrics=True,
)

sample_bundle.accept_reject()
