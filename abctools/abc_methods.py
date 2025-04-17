import logging
import random
import warnings

import numpy as np
import polars as pl
from scipy.stats import qmc, truncnorm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def draw_simulation_parameters(
    params_inputs: dict,
    n_parameter_sets: int,
    method: str = "sobol",
    add_random_seed: bool = True,
    add_simulation_id: bool = True,
    starting_simulation_number: int = 0,
    seed=None,
    replicates_per_particle=1,
) -> pl.DataFrame:
    """
    Draw samples of parameters for simulations based on the specified method.

    Args:
        params_inputs (dict): Dictionary containing parameters and their distributions.
        n_parameter_sets (int): Number of simulations to perform.
        method (str): Sampling method ("random", "sobol", or "latin_hypercube").
        add_random_seed (bool): If True, adds a 'randomSeed' column with randomly generated numbers.
        add_simulation_id (bool): If True, adds a 'simulation' column with simulation IDs starting from `starting_simulation_number`.
        starting_simulation_number (int): The number at which to start numbering simulations. Defaults to 0.
        seed (int): Random seed passed in to ensure consistency between runs.
        replicates_per_particle (int): Number of replicates to generate per parameter set.

    Returns:
        pd.DataFrame: DataFrame containing arrays of sampled values for each parameter,
                      possibly including 'random_seed' and 'simulation' columns.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_params = len(params_inputs)

    if method == "random":
        # Random sampling for each distribution
        samples = {
            param_key: dist_obj.rvs(size=n_parameter_sets)
            for param_key, dist_obj in params_inputs.items()
        }

    elif method == "sobol" or method == "latin_hypercube":
        warnings.filterwarnings("ignore", category=UserWarning)

        # Create the appropriate sampler based on the chosen method
        sampler = (
            qmc.Sobol(d=num_params, seed=seed)
            if method == "sobol"
            else qmc.LatinHypercube(d=num_params, seed=seed)
        )

        # Generate uniform samples across all dimensions
        uniform_samples = sampler.random(n=n_parameter_sets)

        # Transform uniform samples using ppf for each distribution
        samples_transformed = np.column_stack(
            [
                dist.ppf(uniform_samples[:, i])
                for i, dist in enumerate(params_inputs.values())
            ]
        )

        # Convert array of samples into dictionary format
        samples = {
            param_key: samples_transformed[:, i]
            for i, param_key in enumerate(params_inputs.keys())
        }

    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    # Convert to Polars DataFrame
    n_simulations = n_parameter_sets * replicates_per_particle
    simulation_parameters_df = pl.DataFrame(samples).select(
        pl.all().repeat_by(replicates_per_particle).flatten()
    )

    if add_random_seed:
        # Add a random seed column with integers between 0 and 2^32 - 1
        seed_column = [
            random.randint(0, 2**32) for _ in range(n_simulations)
        ]
        simulation_parameters_df = simulation_parameters_df.with_columns(
            pl.Series("randomSeed", seed_column)
        )

    # If specified, add a simulation ID column with integers from `starting_simulation_number` to `starting_simulation_number + n_parameter_sets - 1`
    if add_simulation_id:
        # Generate sequence using arange and offset by the starting number
        simulation_id_sequence = np.arange(
            starting_simulation_number,
            starting_simulation_number + n_simulations,
        )

        # Reorder columns to make 'simulation' the first one
        simulation_parameters_df = pl.concat(
            [
                pl.DataFrame({"simulation": simulation_id_sequence}),
                simulation_parameters_df,
            ],
            how="horizontal",
        )

    return simulation_parameters_df


def resample(
    accepted_simulations: pl.DataFrame,
    n_samples: int,
    replicates_per_sample: int = 1,
    perturbation_kernels: dict = None,
    prior_distributions: dict = None,
    weights: pl.DataFrame = None,
    add_random_seed: bool = True,
    starting_simulation_number: int = 0,
    seed: int = None,
) -> pl.DataFrame:
    """
    Resamples parameters from accepted simulations with optional perturbation and reweighting.

    Args:
        accepted_simulations (pl.DataFrame): DataFrame of accepted simulations with parameters.
        n_samples (int): Number of additional samples to generate.
        replicates_per_sample (int): Number of replicates to generate per sample.
        perturbation_kernels (dict): Dictionary of perturbation kernels for each parameter.
        prior_distributions (dict): Dictionary of prior distributions for each parameter.
        weights (pl.DataFrame or None): Optional DataFrame of weights for each accepted simulation. If None, uniform weighting is assumed.
        add_random_seed (bool): If True, adds a 'random_seed' column with randomly generated numbers.
        starting_simulation_number (int): Starting number for new simulation IDs.
        seed (int): Random seed passed in to ensure consistency between runs.

    Returns:
        pl.DataFrame: DataFrame containing resampled and possibly perturbed parameters.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Prepare list to hold new samples
    new_samples = []

    # Normalize weights if provided (just to be safe)
    if not weights.is_empty():
        total_weight = weights["weight"].sum()
        weights = weights.with_columns(
            (pl.col("weight") / total_weight).alias("weight")
        )
        sim_numbers = weights["simulation"].to_list()
        weights = weights["weight"].to_list()
    else:
        sim_numbers = accepted_simulations["simulation"].to_list()
        weights = [1 / len(sim_numbers)] * len(sim_numbers)

    # Resampling loop
    for _ in range(n_samples):
        # Select a random index based on weights
        chosen_index = random.choices(sim_numbers, weights=weights, k=1)[0]

        # Retrieve the chosen parameters
        chosen_params = accepted_simulations.filter(
            pl.col("simulation") == chosen_index
        ).to_dicts()[0]

        # Perturb each parameter using its respective kernel and check against prior distribution
        selected_params = {}

        # Perturb and test against prior distribution, if specified
        for param_name, value in chosen_params.items():
            if param_name == "simulation":
                continue
            if perturbation_kernels and prior_distributions:
                kernel_dist = perturbation_kernels.get(param_name)
                prior_dist = prior_distributions.get(param_name)

                if kernel_dist is not None:
                    # Apply perturbation until a valid sample within the prior distribution is obtained
                    while True:
                        perturbed_value = value + kernel_dist.rvs()
                        if (
                            prior_dist.pdf(perturbed_value) > 0
                        ):  # Check if within prior distribution
                            break

                    selected_params[param_name] = perturbed_value

                else:
                    raise ValueError(
                        f"No perturbation kernel provided for {param_name}."
                    )
            else:
                selected_params[param_name] = value

        new_samples.append(selected_params)

    n_simulations = n_samples * replicates_per_sample

    # Create resampled dataframe
    resampled_df = pl.DataFrame(new_samples)

    # Ensure all columns have appropriate data types
    resampled_df = resampled_df.with_columns(
        [
            pl.col(col_name).cast(pl.Float64)
            if resampled_df[col_name].dtype == pl.Object
            else pl.col(col_name)
            for col_name in resampled_df.columns
        ]
    )

    # Repeat rows by replicates_per_sample and flatten the DataFrame
    resampled_df = resampled_df.select(
        pl.all().repeat_by(replicates_per_sample).flatten()
    )

    # Add 'simulation' column starting at desired number
    resampled_df = resampled_df.with_columns(
        pl.Series(
            "simulation",
            np.arange(
                starting_simulation_number,
                starting_simulation_number + n_simulations,
            ),
        )
    )

    if add_random_seed:
        # Add a random seed column with integers between 0 and 2^32 - 1
        seed_column = [
            random.randint(0, 2**32 - 1) for _ in range(n_simulations)
        ]
        resampled_df = resampled_df.with_columns(
            pl.Series("randomSeed", seed_column)
        )

    return resampled_df


def calculate_weights_abcsmc(
    current_accepted: pl.DataFrame,
    prev_step_accepted: pl.DataFrame,
    prev_weights: pl.DataFrame,
    stochastic_acceptance_weights: pl.DataFrame,
    prior_distributions: dict,
    perturbation_kernels: dict,
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Calculate weights for simulations in steps t > 0 of an ABC SMC algorithm (Toni et al. 2009)

    Args:
        current_accepted (pl.DataFrame): Accepted simulations for the current step.
        prev_step_accepted (pl.DataFrame): Accepted simulations from the previous step.
        prev_weights (pl.DataFrame): Weights for each simulation from the previous step.
        stochastic_acceptance_weights (pl.DataFrame): DataFrame of stochastic acceptance weights for each simulation.
        prior_distributions (dict): Dictionary containing prior distribution objects for each parameter.
        perturbation_kernels (dict): Dictionary containing scipy.stats distributions used as perturbation kernels for each parameter.
        normalize (bool): If True, normalize the weights so they sum to 1.

    Returns:
        pl.DataFrame: DataFrame of calculated weights for each simulation in current_accepted.
    """

    # Ensure prev_weights weight column is numeric
    prev_weights = prev_weights.with_columns(pl.col("weight").cast(pl.Float64))

    # Initialize list to store new weights
    new_weights = []

    # Loop over all accepted simulations in current step
    for row in current_accepted.iter_rows(named=True):
        sim_number = row["simulation"]
        params = row

        # Calculate numerator
        numerator = 1.0
        for param_name, param_value in params.items():
            if param_name == "simulation":
                continue
            prior_dist = prior_distributions[param_name]
            numerator *= prior_dist.pdf(param_value)

        # Calculate denominator: weighted sum over all previous particles' contribution
        denominator = 0.0
        for prev_row in prev_step_accepted.iter_rows(named=True):
            prev_sim_number = prev_row["simulation"]
            prev_params = prev_row

            kernel_product = 1.0
            for param_name in params.keys():
                if param_name == "simulation":
                    continue
                if param_name in perturbation_kernels:
                    kernel_dist = perturbation_kernels[param_name]
                    kernel_product *= kernel_dist.pdf(
                        params[param_name] - prev_params[param_name]
                    )

            prev_weight = prev_weights.filter(
                pl.col("simulation") == prev_sim_number
            )["weight"][0]
            denominator += prev_weight * kernel_product

        # Avoid division by zero; if denominator is zero set weight to zero directly
        weight = numerator / denominator if denominator != 0 else 0

        # Retrieve stochastic acceptance weight from DataFrame
        stochastic_weight = stochastic_acceptance_weights.filter(
            pl.col("simulation") == sim_number
        )["acceptance_weight"][0]

        # Store calculated weight
        new_weights.append(
            {"simulation": sim_number, "weight": weight * stochastic_weight}
        )

    weights_df = pl.DataFrame(new_weights)

    if normalize:
        # Ensure weights column is numeric before normalization
        weights_df = weights_df.with_columns(pl.col("weight").cast(pl.Float64))

        # Normalize weights so they sum up to 1
        total_new_weight = weights_df["weight"].sum()

        if total_new_weight == 0:
            raise ValueError(
                "Total weight is zero after normalization. Check input data and distributions."
            )

        weights_df = weights_df.with_columns(
            (pl.col("weight") / total_new_weight).alias("weight")
        )

    return weights_df


def get_truncated_normal(mean, sd, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
