# import polars as pl

# from abctools import abc_methods, manager, plot_utils, toy_model
from abctools import manager, abc_methods
# from abctools.abc_classes import SimulationBundle

seed = 12345
replicates = 10

prior_sampler_distros = {
    "averageContactsPerDay": abc_methods.get_truncated_normal(
        mean=2, sd=1, low=0.01, upp=25
    ),
    "infectiousPeriod": abc_methods.get_truncated_normal(
        mean=4, sd=2.5, low=0.01, upp=25
    ),
}

experiment_bundle = manager.call_experiment(
    config = "./abctools/examples/return-summaries/config.yaml",
    experiment_mode = "summarize",
    protocol = ("write"),
    project_seed = seed,
    wd = "./abctools/examples/return-summaries",
    random_sampler = prior_sampler_distros,
    sampler_method = "sobol",
    replicates = replicates
)
