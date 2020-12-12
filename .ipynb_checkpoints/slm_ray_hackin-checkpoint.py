"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved


import copy
import logging
import json
import math
import os
import random
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
# parser.add_argument("--run", type=str, default="PPO")
# parser.add_argument("--torch", action="store_true")
# parser.add_argument("--as-test", action="store_true")
# parser.add_argument("--stop-iters", type=int, default=50)
# parser.add_argument("--stop-timesteps", type=int, default=100000)
# parser.add_argument("--stop-reward", type=float, default=0.1)
parser.add_argument("--torch", action="store_true")
parser.add_argument('--smoke_test', action='store_true')
parser.add_argument('--alpha_max', type=float, default=0.1)
parser.add_argument('--explore_function', type=str, default='random')
parser.add_argument('--name', type=str, default='pbt')

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.schedulers import PopulationBasedTraining
    
def train_RL(config, checkpoint_dir=None): 
    
    print('called TRAINRL', 'checkpoint_dir', checkpoint_dir, 'reporter', reporter)
    # Get the assist amount.
    
    # Take a train step with the policy
    
    # Take an eval step with the policy without the eval
    step = 0
    config=config
    alpha = config.pop('alpha', None)
    config['evaluation_config'] = {
                           'env_config':{'alpha':0}
                       }
    config['evaluation_interval']=1
    config['env_config']={'alpha':alpha}
    agent = PPOTrainer(env='LunarLander-v2', config=config) # initialize agent...
    
    if checkpoint_dir is not None: 
        for _ in range(10): 
            print('----')
        print('loading from checkpoint. Checkpoint dir:', checkpoint_dir)
        alpha = config.pop('alpha', alpha)
        config['evaluation_config'] = {
                               'env_config':{'alpha':0}
                           }
        config['evaluation_interval']=1
        config['env_config']={'alpha':alpha}
        path = checkpoint_dir #os.path.join(checkpoint_dir, "checkpoint")
        print('path from before')
        path = checkpoint_dir[:-2]
        files = os.listdir(path)
        print(files)
        file = [f for f in files if 'checkpoint' in f and not '.' in f][0]
        print('file', file)
        step = int(file.split('-')[1])
        path = os.path.join(path, file)
        print('new path', path)
        agent= PPOTrainer(env='LunarLander-v2', config=config)
        agent.restore(path)
        
        
    while True:
        
        result = agent.train()
        res_me = result
        res_me['eval_rew_mean'] = result['evaluation']['episode_reward_mean']
        with tune.checkpoint_dir(step=step) as checkpoint_dir: 
            path = checkpoint_dir.split('/')
            path = path[:-1]
            path = ('/').join(path) #os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = agent.save(path)
            print("checkpoint saved at", checkpoint)
            print('path passed into agent save', path)
        
        step+=1
        tune.report(**res_me)
        


def slm_explore(config): 
    new_config = copy.deepcopy(config)
    new_config['lr'] = .001
    new_config['alpha'] = 0
    return new_config

    
scheduler = PopulationBasedTraining(
    time_attr='training_iteration', 
    perturbation_interval=1,
    custom_explore_fn=slm_explore
)


class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 5 if args.smoke_test else 10
        if not self.should_stop and result["episode_reward_mean"] > 1000:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= max_iter

    def stop_all(self):
        return self.should_stop

stopper = CustomStopper()
from ray.tune import CLIReporter

reporter=CLIReporter()
reporter.add_metric_column('eval_rew_mean')


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    config = {
        'lr':0.001,
        'alpha': 0, #tune.uniform(0, args.alpha_max),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "framework": "torch" if args.torch else "tf",
    }
    name = '%s_%s_alpha_max_%.3f' %(args.name, args.explore_function, args.alpha_max)
#     resources = PPOTrainer.default_resource_request(config).to_json()
    analysis = tune.run(train_RL, 
#             resources_per_trial=resources,
            config=config, 
            metric="eval_rew_mean",
            mode='max',
            name=name,
            scheduler=scheduler, 
            verbose=1,
            stop=stopper, 
            checkpoint_score_attr="eval_rew_mean",
            keep_checkpoints_num=5,
            num_samples=5,
            progress_reporter=reporter
    )
    
    df = analysis.results_df
    df.to_pickle('/userdata/smetzger/cs285/final_proj_ray/runs/%s.pkl' %name)