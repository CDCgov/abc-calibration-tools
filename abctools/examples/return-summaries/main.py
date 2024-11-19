import polars as pl

# from abctools import abc_methods, manager, plot_utils, toy_model
from abctools import abc_manager, abc_methods, toy_model
from abctools.abc_classes import SimulationBundle

seed = 1234
replicates = 10

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
    # Configure compartments
    N = params["baseScenario"]["population"]
    I0 = int(
        params["baseScenario"]["population"]
        * params["baseScenario"]["initialPrevalence"]
    )
    S0 = N - I0
    R0 = 0
    initial_state = (S0, I0, R0)

    # Configure time
    tmax = params["baseScenario"]["totalDays"]

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


## ======================================#
## Running the experiment---
## ======================================#

experiment_bundle = abc_manager.call_experiment(
    config="./abctools/examples/return-summaries/config.yaml",
    experiment_name="summarize",
    project_seed=seed,
    wd="./abctools/examples/return-summaries",
    random_sampler=prior_sampler_distros,
    sampler_method="sobol",
    runner=sim_runner,
    summarizer=summarize_sims,
    replicates=replicates,
)
