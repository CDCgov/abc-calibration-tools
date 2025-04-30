import math
import os
import random
import unittest

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import uniform

from abctools import abc_methods, plot_utils, toy_model
from abctools.abc_classes import SimulationBundle

# Set random seed
random_seed = 12345
random.seed(random_seed)
np.random.seed(random.randint(0, 2**32 - 1))


def run_toy_model(params: dict):
    """Runs toy stochastic SIR model"""
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


def run_experiment_sequence(
    simulations_df: pl.DataFrame, summary_function=None
) -> pl.DataFrame:
    """Takes in Polars DataFrame of simulation (full) parameter data and outputs complete or summarized results from a stochastic SIR model"""
    results_list = []
    for simulation_params in simulations_df.rows(named=True):
        results = run_toy_model(simulation_params)

        if summary_function:
            summarized_results = summary_function(results)
            results_list.append(
                {
                    "simulation": simulation_params["simulation"],
                    "summary_metrics": summarized_results,
                }
            )
        else:
            results = results.with_columns(
                pl.lit(simulation_params["simulation"])
                .alias("simulation")
                .cast(pl.Int64)
            )
            results_list.append(results)

    if summary_function:
        return pl.DataFrame(results_list)
    else:
        return pl.concat(results_list)


def calculate_infection_metrics(df):
    """User-defined function to calculate infection metrics, in this case time to peak infection and total infected"""

    # Find the time of peak infection (first instance, if multiple)
    time_to_peak_infection = (
        df.sort("infected", descending=True).select(pl.first("time"))
    ).item()

    # Calculate total infected by taking the maximum value from the 'recovered' column
    total_infected = df.get_column("recovered").max()

    # Create a DataFrame with the metrics
    metrics_data = {
        "time_to_peak_infection": [time_to_peak_infection],
        "total_infected": [total_infected],
    }

    metrics_df = pl.DataFrame(metrics_data)

    return metrics_df


def calculate_distance(
    results_data: pl.DataFrame, target_data: pl.DataFrame
) -> float:
    """User-defined function to measure Euclidean distance from target"""

    # Extract values from DataFrames
    time_to_peak_infection_results = results_data[
        "time_to_peak_infection"
    ].item()
    total_infected_results = results_data["total_infected"].item()
    time_to_peak_infection_target = target_data[
        "time_to_peak_infection"
    ].item()
    total_infected_target = target_data["total_infected"].item()

    # Calculate differences
    time_diff = time_to_peak_infection_results - time_to_peak_infection_target
    infected_diff = total_infected_results - total_infected_target

    # Compute Euclidean distance
    distance = math.sqrt(time_diff**2 + infected_diff**2)

    return distance


class TestABCPipeline(unittest.TestCase):
    def setUp(self):
        # Number of simulations (per step/iteration/generation)
        self.n_init = 100  # Number of simulations to initialize each step
        self.n_required = (
            30  # Number of accepted simulations required to proceed
        )

        # Number of steps/iterations/generations
        self.n_steps = 6

        # Set stochasticity
        self.stochastic = True

        # Baseline parameters
        self.baseline_params = {
            "population": 10000,
            "totalDays": 100,
            "initialPrevalence": 0.03,
        }

        # Define prior distributions for the experiment parameters
        self.experiment_params_prior_dist = {
            "averageContactsPerDay": abc_methods.get_truncated_normal(
                mean=2, sd=1, low=0.01, upp=25
            ),
            "infectiousPeriod": abc_methods.get_truncated_normal(
                mean=4, sd=2.5, low=0.01, upp=25
            ),
        }

        # Generate target_data
        self.target_params = {
            "averageContactsPerDay": 1.7,
            "infectiousPeriod": 5,
        }
        target_params_dict = (
            self.baseline_params | self.target_params | {"randomSeed": 142}
        )
        self.target_data = run_toy_model(target_params_dict)
        self.target_metrics = calculate_infection_metrics(self.target_data)
        print(self.target_data.head(10))

        # Set tolerance level
        # Note, the final is high because it's just a draw from the approximated posterior distribution
        self.tolerance = [
            500,
            250,
            100,
            50,
            20,
            10000,
        ]  # todo: turn into percentages

        # Define perturbation kernels
        self.perturbation_kernels = {
            "averageContactsPerDay": uniform(-0.05, 0.1),
            "infectiousPeriod": uniform(-0.1, 0.2),
        }

    def test_pipeline(self):
        # Initialize empty dictionary to keep track of SimulationBundle objectes for each ABC step (iteration/generation)
        sim_bundles = {}

        for step_number in range(self.n_steps):
            # For the initialization step (step 0), sample from the priors
            # For steps 1+, the current sim_bundle is regenerated at the end of the previous loop
            if step_number == 0:
                with self.subTest("Initialize Samples"):
                    input_df = abc_methods.draw_simulation_parameters(
                        params_inputs=self.experiment_params_prior_dist,
                        n_parameter_sets=self.n_init,
                        add_random_seed=self.stochastic,
                        seed=random.randint(0, 2**32 - 1),
                    )
                    self.assertEqual(input_df.shape[0], self.n_init)
            else:
                # Only perturb if it's not the final step
                if step_number != (self.n_steps - 1):
                    with self.subTest(
                        f"Resample, step #{step_number} (includes perturbation and validation checks against prior distributions"
                    ):
                        input_df = abc_methods.resample(
                            sim_bundles[step_number - 1].accepted,
                            n_samples=self.n_init,
                            replicates_per_sample=1,
                            perturbation_kernels=self.perturbation_kernels,
                            prior_distributions=self.experiment_params_prior_dist,
                            weights=sim_bundles[step_number - 1].weights,
                        )
                        self.assertEqual(input_df.shape[0], self.n_init)
                else:
                    with self.subTest(f"Resample, final step #{step_number}"):
                        input_df = abc_methods.resample(
                            sim_bundles[step_number - 1].accepted,
                            n_samples=self.n_init,
                            replicates_per_sample=1,
                            weights=sim_bundles[step_number - 1].weights,
                        )
                        self.assertEqual(input_df.shape[0], self.n_init)

            print(f"Step {step_number}, trying {len(input_df)} samples")

            # Create the simulation bundle for the current step
            sim_bundle = SimulationBundle(
                inputs=input_df,
                step_number=step_number,
                baseline_params=self.baseline_params,
            )

            with self.subTest(f"Run Model, step #{step_number}"):
                results = run_experiment_sequence(sim_bundle.full_params_df)
                sim_bundle.add_results(results, merge_params=False)
                self.assertEqual(
                    sim_bundle.results.n_unique(subset=["simulation"]),
                    self.n_init,
                )
                self.assertIn("time", sim_bundle.results[0].columns)
                self.assertIn("infected", sim_bundle.results[0].columns)
                self.assertIn("recovered", sim_bundle.results[0].columns)

            with self.subTest(
                f"Calculate Summary Metrics, step #{step_number}"
            ):
                sim_bundle.calculate_summary_metrics(
                    calculate_infection_metrics
                )

                self.assertIsInstance(sim_bundle.summary_metrics, pl.DataFrame)
                self.assertEqual(len(sim_bundle.summary_metrics), self.n_init)
                # todo: could ensure time to peak infection is in reasonable range

            with self.subTest(f"Calculate Distances, step #{step_number}"):
                sim_bundle.calculate_distances(
                    self.target_metrics,
                    calculate_distance,
                    use_summary_metrics=True,
                )

                # Ensure all distances are >=0
                for distance in sim_bundle.distances["distance"]:
                    self.assertGreaterEqual(distance, 0)

            with self.subTest(
                f"Accept or Reject Simulations through accept_stochastic, step #{step_number}"
            ):
                sim_bundle.accept_stochastic(self.tolerance[step_number])

                # Ensure at least one simulation is accepted
                self.assertGreaterEqual(len(sim_bundle.accepted), 1)

            with self.subTest(
                f"Check Accepted Simulations and Run/Check/Merge Additional if Needed, step #{step_number}"
            ):
                # Continue adding simulations until the required number is accepted
                fractional_acceptance = sim_bundle.n_accepted / self.n_init
                print(f"{fractional_acceptance*100:.1f}% of samples accepted")
                while sim_bundle.n_accepted < self.n_required:
                    # Calculate how many more simulations need to be initialized
                    n_additional = int(
                        (self.n_required - sim_bundle.n_accepted)
                        * 1.5
                        / fractional_acceptance
                    )

                    # Initialize more samples as a new SimulationBundle additional_sim_bundle
                    if step_number == 0:
                        additional_input_df = abc_methods.draw_simulation_parameters(
                            params_inputs=self.experiment_params_prior_dist,
                            n_parameter_sets=n_additional,
                            add_random_seed=self.stochastic,
                            starting_simulation_number=sim_bundle.n_simulations,
                            seed=random.randint(0, 2**32 - 1),
                        )
                    else:
                        additional_input_df = abc_methods.resample(
                            sim_bundles[step_number - 1].accepted,
                            n_samples=n_additional,
                            perturbation_kernels=self.perturbation_kernels,
                            prior_distributions=self.experiment_params_prior_dist,
                            weights=sim_bundles[step_number - 1].weights,
                            starting_simulation_number=sim_bundle.n_simulations,
                        )

                    print(
                        f"Step {step_number}, trying {len(additional_input_df)} additional samples"
                    )

                    # Create additional SimulationBundle (will be merged into the current step's sim_bundle once evaluated)
                    additional_sim_bundle = SimulationBundle(
                        inputs=additional_input_df,
                        step_number=step_number,
                        baseline_params=self.baseline_params,
                    )

                    # Run model for additional simulations
                    additional_results = run_experiment_sequence(
                        additional_sim_bundle.full_params_df
                    )
                    additional_sim_bundle.add_results(
                        additional_results, merge_params=False
                    )

                    # Calculate summary metrics for the additional simulations
                    additional_sim_bundle.calculate_summary_metrics(
                        calculate_infection_metrics
                    )

                    # Calculate distances for the additional simulations
                    additional_sim_bundle.calculate_distances(
                        self.target_metrics,
                        calculate_distance,
                        use_summary_metrics=True,
                    )

                    # Accept or reject the additional simulations based on tolerance criteria
                    additional_sim_bundle.accept_stochastic(
                        self.tolerance[step_number]
                    )

                    # Merge the new SimulationBundle into current sim_bundle
                    sim_bundle.merge_with(
                        additional_sim_bundle
                    )  # Notes: -pay attention to simulation numbers, -keep track of number of additional runs/merges

                # Check if there are enough accepted simulations
                print(
                    f"Step {step_number}, {sim_bundle.n_accepted} samples accepted"
                )
                self.assertGreaterEqual(sim_bundle.n_accepted, self.n_required)

            with self.subTest(
                f"Collate accepted simulations, step #{step_number}"
            ):
                # Collate results from distance calculations
                sim_bundle.collate_accept_results()

                # Ensure the accepted logical column is present
                self.assertIn("accept_bool", sim_bundle.accept_results.columns)

                # Ensure the sum of TRUE counts in logical column is the number of accepted sims
                self.assertEqual(
                    sim_bundle.accept_results["accept_bool"].sum(),
                    sim_bundle.n_accepted,
                )

                print()
                # Check that distance values with accepted_sim == True are less than or equal to tolerance
                accept_above_max = sim_bundle.accept_results.filter(
                    pl.col("accept_bool")
                ).filter(pl.col("distance") > self.tolerance[step_number])
                self.assertTrue(accept_above_max.is_empty())

            with self.subTest(f"Calculate weights, step #{step_number}"):
                if step_number == 0:
                    # Uniform weights on the initial step
                    total_accepted = len(sim_bundle.accepted)
                    weights = sim_bundle.accepted.select(
                        pl.col("simulation"),
                        (pl.lit(1.0) / total_accepted).alias("weight"),
                    )
                else:
                    prev_step_accepted = sim_bundles[step_number - 1].accepted
                    prev_step_weights = sim_bundles[step_number - 1].weights

                    weights = abc_methods.calculate_weights_abcsmc(
                        current_accepted=sim_bundle.accepted,
                        prev_step_accepted=prev_step_accepted,
                        prev_weights=prev_step_weights,
                        prior_distributions=self.experiment_params_prior_dist,
                        perturbation_kernels=self.perturbation_kernels,
                        normalize=True,
                    )

                sim_bundle.weights = weights
                total_weight = weights["weight"].sum()
                self.assertAlmostEqual(total_weight, 1.0, places=5)

            # Add current sim_bundle to sim_bundles dictionary
            sim_bundles[step_number] = sim_bundle
            del sim_bundle

        ### Plots
        with self.subTest("Make timeseries plots"):
            output_folder = "abctools/tests/figs"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for step_number, sim_bundle in sim_bundles.items():
                unique_simulations = (
                    sim_bundle.results.select("simulation")
                    .unique()
                    .to_series()
                    .to_list()
                )

                data_list = [
                    sim_bundle.results.filter(pl.col("simulation") == sim)
                    for sim in unique_simulations
                ]

                x_col = "time"
                y_col = "infected"
                plot_args_list = [{"color": "blue", "alpha": 0.5}] * len(
                    data_list
                )

                # Add target data
                data_list.append(self.target_data)
                plot_args_list.append({"color": "red", "alpha": 0.8})

                # Label axes
                xlabel = "Time (days)"
                ylabel = "Infections"

                # Plot
                fig = plot_utils.plot_xy_data(
                    data_list, x_col, y_col, plot_args_list, xlabel, ylabel
                )
                file_out = os.path.join(
                    output_folder,
                    f"infection_timeseries_step_{step_number}.jpg",
                )
                fig.savefig(file_out)

        with self.subTest("Make results plots"):
            for step_number, sim_bundle in sim_bundles.items():
                data_list = []
                for summary_metrics in sim_bundle.summary_metrics.iter_rows(
                    named=True
                ):
                    data_list.append(
                        pl.DataFrame(
                            {
                                "time_to_peak_infection": [
                                    summary_metrics["time_to_peak_infection"]
                                ],
                                "total_infected": [
                                    summary_metrics["total_infected"]
                                ],
                            }
                        )
                    )
                x_col = "time_to_peak_infection"
                y_col = "total_infected"
                plot_args_list = [
                    {"color": "blue", "alpha": 0.10, "marker": "o"}
                ] * len(data_list)

                # Add target data
                data_list.append(self.target_metrics)
                plot_args_list.append(
                    {"color": "red", "alpha": 0.9, "marker": "o"}
                )
                # Label axes
                xlabel = "Time to peak infections"
                ylabel = "Total Infections"

                # Plot
                fig = plot_utils.plot_xy_data(
                    data_list, x_col, y_col, plot_args_list, xlabel, ylabel
                )
                file_out = os.path.join(
                    output_folder,
                    f"target_metrics_step_{step_number}.jpg",
                )
                fig.savefig(file_out)

            with self.subTest("Make parameter histograms"):
                output_folder = "abctools/tests/figs/parameters"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                for step_number, sim_bundle in sim_bundles.items():
                    for experiment_param in sim_bundle.experiment_params:
                        df = sim_bundle.inputs
                        col = experiment_param
                        vline_value = self.target_params[experiment_param]

                        plt.hist(df[col])
                        plt.axvline(
                            x=vline_value,
                            color="red",
                        )
                        plt.xlabel(experiment_param)
                        plt.ylabel("frequency")

                        file_out = os.path.join(
                            output_folder,
                            f"{experiment_param}_hist_step_{step_number}.jpg",
                        )
                        plt.savefig(file_out)
                        plt.close()

        with self.subTest("Posterior predictive check"):
            # TODO: add posterior predictive check
            pass
            # Add assertion


if __name__ == "__main__":
    unittest.main()
