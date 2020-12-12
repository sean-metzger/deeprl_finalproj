#!/usr/bin/env python
"""Example of using PBT with RLlib.
Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).
Note that Tune in general does not need 8 GPUs, and this is just a more
computationally demanding example.
"""

import random

from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

if __name__ == "__main__":

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=3,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "env_config.alpha": np.random.uniform(0, .01)
        }
    )
        #custom_explore_fn=explore)

    analysis = tune.run(
        "PPO",
        name="pbt_test_two",
        scheduler=pbt,
        num_samples=4,
        metric="evaluation_frequency.",
        mode="max",
        verbose=1,
        config={
            "env": "LunarLander-v2",
            "env_config":{
                {'alpha':tune.uniform(0,.01)}
            },
            "evaluation_config": { 
                    "env_conifg":{'alpha':0}
            },
            'evaluation_frequency':1,
            "num_workers": 4,
            "num_gpus": 0,
            "lr": 1e-3,
        })