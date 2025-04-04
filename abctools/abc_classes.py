import logging
import os
import pickle

import polars as pl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimulationBundle:
    """
    A class to keep track of an iteration/generation of simulations (particles)
    for ABC/SMC.

    Attributes:
        inputs (pl.DataFrame): Input parameters for the simulations.
        results (pl.DataFrame): Results for the simulations, initialized as an empty DataFrame.
        step_number (int): Keeps track of the ABC step (a.k.a. generation/iteration)
        baseline_params (dict): Unchanging parameters needed for the simulation
        experiment_params (list): Derived from 'inputs'--list of experiment parameter names
        status (str): Current status in the ABC process
        distances (pl.DataFrame): Calculated distances from target
        accepted (pl.DataFrame): Accepted simulations with experiment parameters
        n_accepted (int): Calculated from 'accepted'--number of accepted simulations
        weights (pl.DataFrame): Simulation weights for resampling
        merge_history (dict): History of merges with other SimulationBundle objects
        summary_metrics (pl.DataFrame): Summary metrics calculated for each simulation
        acceptance_weights (pl.DataFrame): Weights for accepted simulations
        accept_results (pl.DataFrame): DataFrame of accepted results
    """

    def __init__(
        self,
        inputs: pl.DataFrame,
        step_number: int,
        baseline_params: dict,
        status: str = "initialized",
    ):
        """
        Initialize a new instance of SimulationsBundle.

        Args:
            inputs (pl.DataFrame): Input parameters for the simulations (optionally including randomSeed).
            step_number (int): Step/iteration/generation number.
            baseline_params (dict): The baseline parameters for the simulations.
            status (str): Current status of the object. Defaults to "initialized".
        """

        # Public variables
        self.inputs = inputs
        self.status = status
        self.merge_history = {}
        self.results = pl.DataFrame()
        self.distances = pl.DataFrame()
        self.accepted = pl.DataFrame()
        self.acceptance_weights = pl.DataFrame()
        self.weights = pl.DataFrame()
        self.summary_metrics = pl.DataFrame()

        # Private variables
        self._step_number = step_number
        self._baseline_params = baseline_params
        self._experiment_params = [
            col
            for col in inputs.columns
            if col not in ["simulation", "randomSeed"]
        ]

    @property
    def step_number(self) -> int:
        """Getter for _step_number."""
        return self._step_number

    @property
    def n_simulations(self) -> int:
        """Getter for _n_simulations."""
        return self.inputs["simulation"].n_unique()

    @property
    def baseline_params(self) -> list:
        """Getter for _baseline_params."""
        return self._baseline_params

    @property
    def experiment_params(self) -> list:
        """Getter for _experiment_params."""
        return self._experiment_params

    @property
    def n_accepted(self) -> int:
        """Getter for number of accepted simulations"""
        return self.accepted.shape[0]

    @property
    def writer_input_dict(self) -> dict:
        """Getter that outputs a dictionary with simulation details. Needed by gcm_python_wrappers.wrappers.gcm_experiments_writer"""
        return {
            "baseline_parameters": self._baseline_params,
            "experiment_parameters": self._experiment_params,
            "simulation_parameter_values": self.inputs,
        }

    @property
    def full_params_df(self) -> pl.DataFrame:
        """Getter that outputs a Polars DataFrame with the full parameters list (simulation number, random seed, baseline parameters, and experimental parameters)"""
        full_params_df = self.inputs
        for colname, value in self._baseline_params.items():
            full_params_df = full_params_df.with_columns(
                pl.lit(value).alias(colname)
            )
        return full_params_df

    def __getstate__(self):
        """
        Specifies what gets pickled when the save_state method is called.

        Returns:
            state (dict): The object's state without the 'results' attribute.
        """
        # Copy object's __dict__
        state = self.__dict__.copy()

        # Remove 'results'
        if "results" in state:
            del state["results"]

        return state

    def save_state(self, folder_path: str, filename: str):
        """
        Saves the current state of the simulation bundle to a file using pickle,
        excluding 'results'.

        Args:
            folder_path (str): The path to the folder where state should be saved.
            filename (str): The name of the file to save state into.

        Returns:
            None
        """
        # Check if folder exists, and create it if it doesn't
        os.makedirs(folder_path, exist_ok=True)

        # Create full path for the output file
        full_path = os.path.join(folder_path, filename)

        # Use 'with' statement to ensure that file is properly closed after writing
        with open(full_path, "wb") as file:
            # Pickle only selected parts of the object and write it to file
            pickle.dump(self.__getstate__(), file)

    def add_results(self, results_df, recover_params=True):
        """
        Adds results to the SimulationBundle object.

        Args:
            results_df (pl.DataFrame): The results DataFrame to be added.

        Returns:
            None
        """
        # Ensure results_df is a Polars DataFrame with all of the same values in the 'simulation' column as inputs
        # Note: there may be more than one row per simulation in results_df
        if not isinstance(results_df, pl.DataFrame):
            raise TypeError("results_df must be a Polars DataFrame.")

        if not results_df["simulation"].is_in(self.inputs["simulation"]).all():
            raise ValueError(
                "results_df must contain all simulation numbers from inputs."
            )

        # Recover params if applicable
        if recover_params:
            self.recover_params()

        self.results = results_df

    def recover_params(self):
        """
        Updates self.results by merging in columns from self.inputs onto self.results based on the 'simulation' column.

        Returns:
            None
        """
        if self.results is None:
            raise ValueError(
                "self.results is not set. Cannot recover parameters without results."
            )

        # Perform a left join to add input parameters to results based on 'simulation'
        merged_results = self.results.join(
            self.inputs, on=["simulation"], how="left"
        )

        # Ensure the DataFrame is unique on 'simulation'
        merged_results = merged_results.unique(subset=["simulation"])

        # Update self.results with merged data
        self.results = merged_results

    def calculate_summary_metrics(self, summary_function):
        """
        Applies a user-defined function to calculate summary metrics for each simulation.

        Args:
            summary_function (callable): A function that takes in per-simulation results (a Polars DataFrame, typically) and returns summary metrics.

        Returns:
            None
        """
        if self.results is None:
            raise ValueError("No results available to summarize.")

        self.summary_metrics = pl.DataFrame()

        grouped_results = self.results.group_by("simulation").map_groups(
            summary_function
        )

        for sim_number in grouped_results["simulation"]:
            metrics_df = grouped_results.filter(
                pl.col("simulation") == sim_number
            )
            self.summary_metrics = self.summary_metrics.vstack(metrics_df)

    def calculate_distances(
        self, target_data, distance_function, use_summary_metrics=True
    ):
        """
        Calculates distances between simulation results and target data using a user-defined distance function.

        Args:
            target_data (tuple): Target data to compare against.
            distance_function (callable): A user-defined function that takes results_data and target_data and returns a distance.
            use_summary_metrics (bool): Whether to use summary metrics or raw results. Defaults to True if summary metrics have been calculated.

        Returns:
            None
        """

        # Check if summary metrics should be used
        if use_summary_metrics and not self.summary_metrics.is_empty():
            data_to_use = self.summary_metrics
        else:
            data_to_use = self.results

        # Calculate distances using the chosen data
        distances_list = []

        for sim_number, sim_data in data_to_use.group_by("simulation"):
            sim_number = sim_number[0]  # Extract the simulation number
            if "simulation" in sim_data.columns:
                sim_data = sim_data.drop("simulation")
            distance = distance_function(sim_data, target_data)
            distances_list.append(
                {"simulation": sim_number, "distance": distance}
            )

        self.distances = pl.DataFrame(distances_list)

    def accept_reject(self, tolerance):
        """
        Accepts or rejects simulations based on the calculated distances and given tolerance level.

        Args:
            tolerance (float): The tolerance level for accepting simulations.

        Returns:
            None
        """

        # Ensure distances have been calculated
        if self.distances.is_empty():
            raise ValueError("Distances have not been calculated.")

        # Filter distances based on tolerance
        accepted_simulations = self.distances.clone().filter(
            pl.col("distance") <= tolerance
        )

        # Join with inputs to get accepted parameters
        self.accepted = accepted_simulations.join(
            self.inputs, on="simulation"
        ).drop("distance")

        # Drop 'randomSeed' columns if present
        if "randomSeed" in self.accepted.columns:
            self.accepted = self.accepted.drop("randomSeed")

        # Create acceptance weights DataFrame
        self.acceptance_weights = self.accepted.with_columns(
            pl.lit(1.0).alias("acceptance_weight")
        )

    def accept_stochastic(
        self,
        tolerance,
    ):
        """
        Accepts the minimum simulation of each parameter set with greater than zero replicates under the tolerance level
        Sets the acceptance_weight proportion for each parameter set

        Args:
            tolerance (float): The tolerance level for accepting simulations.

        Returns:
            None

        Raises:
            ValueError: If distances have not been previously calculated.
        """

        if self.distances.is_empty():
            raise ValueError("Distances have not been calculated.")

        self.accept_reject(tolerance)
        self.collate_accept_results()

        # Group by parameters besides simulation and random seed
        particle_prop_accepted = self.accept_results.group_by(
            self.inputs.drop(["simulation", "randomSeed"]).columns
        ).agg(
            pl.col("accept_bool").mean().alias("acceptance_weight"),
            pl.col("simulation").min().alias("simulation"),
        )

        # filter particles with accepted count > 0
        particle_prop_accepted = particle_prop_accepted.filter(
            pl.col("acceptance_weight") > 0
        )

        accepted_list = []
        acceptance_weights_list = []

        # Create acceptance weights (to be included in the weights assignment) and params dict
        for row in particle_prop_accepted.rows(named=True):
            acceptance_weights_list.append(
                {
                    "simulation": row["simulation"],
                    "acceptance_weight": row["acceptance_weight"],
                }
            )

            # Extract parameters and convert to dictionary
            params_dict = (
                self.inputs.filter(pl.col("simulation") == row["simulation"])
                .drop(["simulation", "randomSeed"])
                .to_dicts()[0]
            )

            # Add simulation ID to the parameters dictionary
            params_dict["simulation"] = row["simulation"]

            accepted_list.append(params_dict)

        self.acceptance_weights = pl.DataFrame(acceptance_weights_list)
        self.accepted = pl.DataFrame(accepted_list)

    def accept_proportion(self, proportion):
        """
        Accepts a specified proportion of simulations with the smallest distances.
        This method ranks all simulations by their distance values in ascending order
        and selects the top-performing simulations up to the specified proportion.

        Args:
            proportion (float): The proportion of top simulations to accept based on their distances.
                                For example, 0.1 for the top 10%, or 0.25 for the top 25%.

        Returns:
            None

        Raises:
            ValueError: If distances have not been previously calculated.
        """

        # Ensure distances have been calculated
        if self.distances.is_empty():
            raise ValueError("Distances have not been calculated.")

        # Calculate the number of simulations to accept based on the given proportion (minimum of 1)
        num_to_accept = max(1, int(self.distances.shape[0] * proportion))

        # Sort simulations by distance in ascending order and select the best ones
        sorted_distances = self.distances.sort("distance").head(num_to_accept)

        accepted_list = []
        acceptance_weights_list = []

        # Accept only the top-performing simulations as determined by the specified proportion
        for row in sorted_distances.rows(named=True):
            # Retrieve and clean input parameters for each accepted simulation
            accepted_params = self.inputs.filter(
                pl.col("simulation") == row["simulation"]
            )
            if "simulation" in accepted_params.columns:
                accepted_params = accepted_params.drop("simulation")
            if "randomSeed" in accepted_params.columns:
                accepted_params = accepted_params.drop("randomSeed")

            # Store parameters of accepted simulations in an attribute for later use or analysis
            accepted_list.append(
                {"simulation": row["simulation"], "params": accepted_params}
            )
            acceptance_weights_list.append(
                {"simulation": row["simulation"], "acceptance_weight": 1.0}
            )

        self.accepted = pl.DataFrame(accepted_list)
        self.acceptance_weights = pl.DataFrame(acceptance_weights_list)

    def collate_accept_results(self):
        """
        Makes a single DataFrame attribute from ABC results of accepted, distance, and inputs.
        Args:
            self
        Returns:
            None
        Raises:
            ValueError: if accept is not previously calculated
        """

        # Ensure accept is already calculated
        if self.accepted.is_empty():
            raise ValueError("Accept has not been calculated.")

        # Dummy mapper to join distances and accept with inputs
        mapper = pl.DataFrame(
            {
                "simulation": self.distances["simulation"],
                "distance": self.distances["distance"],
            }
        )

        # Joining results with inputs
        distance_results = self.inputs.join(
            mapper, on="simulation", how="left"
        )

        # Adding logical column whether an input was accepted and storing as self attribute
        accepted_sims = self.accepted["simulation"].to_list()
        self.accept_results = distance_results.with_columns(
            pl.col("simulation").is_in(accepted_sims).alias("accept_bool")
        )

    def merge_with(self, other_bundle):
        """
        Merges another SimulationBundle object into this one by combining their inputs,
        results, summary metrics, distances, and accepted simulations.

        Args:
            other_bundle (SimulationBundle): Another SimulationBundle instance to merge with this one.

        Returns:
            None
        """

        # Merge inputs DataFrames directly
        merged_inputs = self.inputs.vstack(other_bundle.inputs)

        # Check for duplicate simulation numbers after merging
        if (
            merged_inputs["simulation"].unique().len()
            != merged_inputs["simulation"].len()
        ):
            raise ValueError(
                "Duplicate simulation numbers found after merging. Merge aborted."
            )

        # If no duplicates are found, proceed with updating self.inputs
        self.inputs = merged_inputs

        # Merge results as DataFrames
        self.results = self.results.vstack(other_bundle.results)

        # Merge distances DataFrames directly
        self.distances = self.distances.vstack(other_bundle.distances).unique(
            subset=["simulation"]
        )

        # Merge accepted simulations DataFrames directly
        self.accepted = self.accepted.vstack(other_bundle.accepted).unique(
            subset=["simulation"]
        )

        # Merge acceptance weights DataFrames directly
        self.acceptance_weights = self.acceptance_weights.vstack(
            other_bundle.acceptance_weights
        ).unique(subset=["simulation"])

        # Merge summary metrics DataFrames directly
        self.summary_metrics = self.summary_metrics.vstack(
            other_bundle.summary_metrics
        ).unique(subset=["simulation"])

        # Record the merge event in the history
        current_merge_index = len(self.merge_history) + 1
        number_merged = len(other_bundle.inputs)

        self.merge_history[current_merge_index] = number_merged
