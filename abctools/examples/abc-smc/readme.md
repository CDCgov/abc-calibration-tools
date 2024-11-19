# Overview
This example is for a simple ABC-SMC routine with few parmaeters in a toy stochastic SIR model, emulating the test of `abc_classes` provided through `test_abc_pipeline.py`

### Running the example
This example can be run by using
```shell
poetry run python abctools/examples/abc-smc/main.py
```

# Workflow
## User-defined functions
The script takes in a user-defined set of functions that determine how the parameters are drawn, how the models are run, and where simulations are stored. The first section of user-defined functions specifies how to collect simulations, how to collect summary metrics, and the rules for drawing parmaeters

Parameters are drawn during the first (`prior_sampler_distros`) and subsequent (`perturbation_kernels`) rounds of ABC-SMC.
The `toy_model` function uses these drawn parameters and a base set is defined for generating a single simulation and `sim_runner` is defined to collect multiple simulations and store them as the results value of a SimulationBundle object. The results are then summarized by the `summarize_sims` function. These summary metrics are then compared to some target summary metrics through the `distance_difference` function.

## Environment and Target
The environmental set up is used to establish the working directory of the example, the tolerance steps being used during ABC-SMC, and the number of samples to be simulated per tolerance step. The tolerances can be of arbitrary length and acceptance criteria can be tuned for calibration performance.

We then use the manager wrapper to `call_experiment` for generating a synthetic target data set. We use the `baseScenario` configuration as baseline parameters nd specify a target seed. We store both the simulations (called through `runner`) and the summaries (called through `summarizer`) by including them in the `write` array. All writing takes place in the example by specifying the working directory `wd`.

## ABC-SMC
We then run the ABC-SMC routine by iterating over call experiments for each tolerance step. We now include a `random_sampler` to draw from the prior distributions. For subsesquent ABC-SMC steps, we pass on the SimulationBundle of the previous ABC-SMC step, which has already been post-processed.

The post-processing after each draw and simulation run and summary is to cacluatle weights and collect the accepted simulation scores. The modified SimulationBundle of the given step is then stored into a dictionary of SimulationBundles.

## Figures
One brief figure is generated per parameter sampled according to the ABC-SMC. These are the sequential histgorams of draws provided per step.

# Out of scope
- This example will be followed by future implementations that develop calibration wrappers and options with save points
- Different types of tolerances, such as percentages or replicating the re-draw until hitting particular numbers of accepted simulations
