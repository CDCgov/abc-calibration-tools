import os
import pickle

import polars as pl


class SimulationBundle:
    """
    A class to keep track of an iteration/generation of simulations (particles)
    for ABC/SMC.

    Attributes:
        inputs (pl.DataFrame): Input parameters for the simulations.
        results (pl.DataFrame): Results for the simulations, initialized as an empty DataFrame.
        step_number (int): Keeps track of the ABC step (a.k.a. generation/iteration)
        baseline_params (dict): Unchanging parameters needed for the simulation
        experiment_params (list): Calculated from 'inputs'--list of experiment parameter names
        status (str): Current status in the ABC process
        distances (dict): Calculated distances from target
        accepted (dict): Accepted simulations with experiment parameters
        n_accepted (int): Calculated from 'accepted'--number of accepted simulations
        weights (dict): Simulation weights for resampling
        merge_history (dict): History of merges with other SimulationBundle objects
        summary_metrics (dict): Summary metrics calculated for each simulation
        acceptance_weights (dict): Weights for accepted simulations
        accept_results (pl.DataFrame): DataFrame of acceptance results
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
        self.weights = None
        self.results = pl.DataFrame()  # Initialize results as an empty DataFrame

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
        return len(self.accepted)

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

    def recover_params(self):
        """
        Updates self.results by merging in columns from self.inputs onto self.results based on the 'simulation' and 'randomSeed' columns.

        Returns:
            None
        """
        if self.results is None:
            raise ValueError(
                "self.results is not set. Cannot recover parameters without results."
            )

        # Ensure results is a Polars DataFrame
        if not isinstance(self.results, pl.DataFrame):
            raise TypeError(
                "self.results must be a Polars DataFrame."
            )

        # Perform a left join to add input parameters to results based on 'simulation' and 'randomSeed'
        merged_results = self.results.join(
            self.inputs, on=["simulation", "randomSeed"], how="left"
        )

        # Ensure the DataFrame is unique on 'simulation' and 'randomSeed'
        merged_results = merged_results.unique(subset=["simulation", "randomSeed"])

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

        self.summary_metrics = {}

        grouped_results = self.results.groupby("simulation").apply(
            summary_function
        )
        for sim_number in grouped_results["simulation"]:
            self.summary_metrics[sim_number] = grouped_results.filter(
                pl.col("simulation") == sim_number
            ).to_dict(False)

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
        if use_summary_metrics and hasattr(self, "summary_metrics"):
            data_to_use = self.summary_metrics
        else:
            data_to_use = {
                row["simulation"]: row for row in self.results.to_dict()
            }

        # Calculate distances using the chosen data
        self.distances = {}

        for sim_number, sim_data in data_to_use.items():
            distance = distance_function(sim_data, target_data)
            self.distances[sim_number] = distance

    def accept_reject(self, tolerance):
        """
        Accepts or rejects simulations based on the calculated distances and given tolerance level.

        Args:
            tolerance (float): The tolerance level for accepting simulations.

        Returns:
            None
        """

        # Ensure distances have been calculated
        if not hasattr(self, "distances"):
            raise ValueError("Distances have not been calculated.")

        # Initialize accepted simulations dictionary
        self.accepted = {}
        self.acceptance_weights = {}

        # Iterate over simulations and accept those within tolerance
        for sim_number, distance in self.distances.items():
            if distance <= tolerance:
                # Filter inputs for current simulation and remove 'simulation' and 'randomSeed' columns if present
                accepted_params = self.inputs.filter(
                    pl.col("simulation") == sim_number
                )
                if "simulation" in accepted_params.columns:
                    accepted_params = accepted_params.drop("simulation")
                if "randomSeed" in accepted_params.columns:
                    accepted_params = accepted_params.drop("randomSeed")

                # Add filtered parameters to the dictionary of accepted simulations
                self.accepted[sim_number] = accepted_params
                self.acceptance_weights[sim_number] = 1.0

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

        if not hasattr(self, "distances"):
            raise ValueError("Distances have not been calculated.")

        self.accept_reject(tolerance)
        self.collate_accept_results()

        # Group by parameters besides simulation and random seed
        particle_prop_accepted = self.accept_results.groupby(
            self.inputs.drop(["simulation", "randomSeed"]).columns
        ).agg(
            pl.col("accept_bool").mean().alias("acceptance_weight"),
            pl.col("simulation").min().alias("simulation"),
        )

        # filter particles with accepted count > 0
        particle_prop_accepted = particle_prop_accepted.filter(
            pl.col("acceptance_weight") > 0
        )

        self.acceptance_weights = {}
        self.accepted = {}
        # Create acceptance weights (to be included in the weights assignment) and params dict
        for row in particle_prop_accepted.rows(named=True):
            self.acceptance_weights.update(
                {row["simulation"]: row["acceptance_weight"]}
            )
            self.accepted.update(
                {
                    row["simulation"]: self.inputs.filter(
                        pl.col("simulation") == row["simulation"]
                    ).drop(["simulation", "randomSeed"])
                }
            )

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
        if not hasattr(self, "distances"):
            raise ValueError("Distances have not been calculated.")

        # Calculate the number of simulations to accept based on the given proportion (minimum of 1)
        num_to_accept = max(1, int(len(self.distances) * proportion))

        # Sort simulations by distance in ascending order and select the best ones
        sorted_simulations = sorted(
            self.distances.items(), key=lambda item: item[1]
        )

        # Initialize/clear accepted simulations dictionary
        self.accepted = {}
        self.acceptance_weights = {}

        # Accept only the top-performing simulations as determined by the specified proportion
        for sim_number, distance in sorted_simulations[:num_to_accept]:
            # Retrieve and clean input parameters for each accepted simulation
            accepted_params = self.inputs.filter(
                pl.col("simulation") == sim_number
            )
            if "simulation" in accepted_params.columns:
                accepted_params = accepted_params.drop("simulation")
            if "randomSeed" in accepted_params.columns:
                accepted_params = accepted_params.drop("randomSeed")

            # Store parameters of accepted simulations in an attribute for later use or analysis
            self.accepted[sim_number] = accepted_params
            self.acceptance_weights[sim_number] = 1.0

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
        if not hasattr(self, "accepted"):
            raise ValueError("Accept has not been calculated.")

        # Dummy mapper to join distances and accept with inputs
        mapper = pl.DataFrame(
            {
                "simulation": self.distances.keys(),
                "distance": self.distances.values(),
            }
        )

        # Joining results with inputs
        distance_results = self.inputs.join(
            mapper, on="simulation", how="left"
        )

        # Adding logical column whether an input was accepted and storing as self attribute
        accepted_sims = list(int(k) for k in self.accepted.keys())
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
        self.results = self.results.vstack(other_bundle.results).unique(subset=["simulation", "randomSeed"])

        # Merge distances dictionaries directly
        self.distances.update(other_bundle.distances)

        # Merge accepted simulations dictionaries directly
        self.accepted.update(other_bundle.accepted)

        # Merge acceptance weights dictionaries directly
        self.acceptance_weights.update(other_bundle.acceptance_weights)

        # Record the merge event in the history
        current_merge_index = len(self.merge_history) + 1
        number_merged = len(other_bundle.inputs)

        self.merge_history[current_merge_index] = number_merged
