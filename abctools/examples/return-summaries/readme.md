# Return simulations summary
Basic example of manager wrapper to make a directory, store simulations, summarize the simulations, and store the summarized outputs

## Goal
Purpose of this example is to show a simple implementation of the manager wrapper and how user defined calls are generated. The majority of the `main.py` file is used to create the user-specified simulation functions and parameter draw functions. 

## Calling experiments through the manager
There are only three required parameter inputs
1. `config` - configuration file to specify parameter inputs
2. `experiment_mode` - user-defined mode of the experiment that determines directory management, doesn't require using a string already known to the 
3. `write` - tuple that tells the manager what outputs to write to files

The returned output of `manager.call_experiment` is a `SimulationBundle` object that contains the information from the simulations and summaries so that no information has to be written for further analysis.

## Implementation
This experiment call only seeks to generate a certain number of simulations, store those results, then summarize the simulations, and store them as well.

Since we want to store all of the simulated and summarized products, we set `write` to `['simulations','summaries']` and provide a working directory `wd` in which to write the product sub-directories. We set the `experiment_mode` to a useful name for organizing this experiment within the example.

The user defines a `sim_runner` function that is accepted asthea callable input `runner`. This function iterates over parameters defined in a simulation bundle parameter dictionary and uses the `run_toy_model` to execute a simple exact stochastic SIR simulation. The number of simulations being run is detemrined by the `replicates` input and the initial project-level seed is provided for simulation consistency.

The `summarizer` input then takes a user-specified function to return `DataFrame`s from the simulated outputs, even if they are not printed to an output directory.

Finally, the manager then returns the `SimulationBundle` object for further analysis.
