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

parser.add_argument("--torch", action="store_true")
parser.add_argument('--smoke_test', action='store_true')
parser.add_argument('--alpha_max', type=float, default=0.1)
parser.add_argument('--explore_function', type=str, default='original')
parser.add_argument('--name', type=str, default='pbt')
parser.add_argument('--num_samples', type=int, default=16)
parser.add_argument('--perturbation_interval', type=int, default=1)

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.schedulers import PopulationBasedTraining
    
def train_RL(config, checkpoint_dir=None): 
    """
    This is the PBT code to train the agent. It evaluates the agents on the env without any assist. 
    """
    
#     print('called TRAINRL', 'checkpoint_dir', checkpoint_dir, 'reporter', reporter)
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
#         for _ in range(10): 
#             print('----')
#         print('loading from checkpoint. Checkpoint dir:', checkpoint_dir)
        alpha = config.pop('alpha', alpha)
        config['evaluation_config'] = {
                               'env_config':{'alpha':0}
                           }
        config['evaluation_interval']=1
        config['env_config']={'alpha':alpha}
        path = checkpoint_dir #os.path.join(checkpoint_dir, "checkpoint")
#         print('path from before')
        path = checkpoint_dir[:-2]
        files = os.listdir(path)
        try:
            file = [f for f in files if 'checkpoint' in f and not '.' in f][0]
#             print('file', file)
            step = int(file.split('-')[1])
            path = os.path.join(path, file)
#             print('new path', path)
            agent= PPOTrainer(env='LunarLander-v2', config=config)
            agent.restore(path)
            
        except Exception: 
            for _ in range(10):
                print('resetting, no chkpoint')
        
        
    while True:
        
        result = agent.train()
        res_me = result
        res_me['eval_rew_mean'] = result['evaluation']['episode_reward_mean']
        res_me['eval_rew_std'] = np.std(result['evaluation']['hist_stats']['episode_reward'])
        res_me['cur_alpha'] = config.pop('alpha', alpha)
        with tune.checkpoint_dir(step=step) as checkpoint_dir: 
            path = checkpoint_dir.split('/')
            path = path[:-1]
            path = ('/').join(path) #os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = agent.save(path)
#             print("checkpoint saved at", checkpoint)
#             print('path passed into agent save', path)
        
        step+=1
        tune.report(**res_me)
        


def decrease_alpha(config): 
    new_config = copy.deepcopy(config)
    new_config['lr'] = 5e-5
    new_config['alpha'] = config['alpha']*5/6
    return new_config

def decrease_alpha_free_lr(config):
    new_config = copy.deepcopy(config)
    new_config['lr'] = 5e-5
    new_config['alpha'] = config['alpha']*5/6
    return new_config

def original(config): 
    return new_config

    



class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 5 if args.smoke_test else 100
        if not self.should_stop and result["episode_reward_mean"] > 1000:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= max_iter

    def stop_all(self):
        return self.should_stop

stopper = CustomStopper()
from ray.tune import CLIReporter

reporter=CLIReporter()
reporter.add_metric_column('eval_rew_mean')
reporter.add_metric_column('eval_rew_std')

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    config = {
        'lr':5e-5,
        'alpha': tune.uniform(0, args.alpha_max),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "framework": "torch" if args.torch else "tf",
    }
    name = '%s_%s_alpha_max_%.3f_perturb_%d' %(args.name, args.explore_function, args.alpha_max, args.perturbation_interval)
    resources = PPOTrainer.default_resource_request(config).to_json()

    
    # Get the custom exploration function.
    print('USING EXPLORE FUNCTION:', args.explore_function)
    
    if args.explore_function == 'original':
        slm_explore= original
    elif args.explore_function == 'decrease_alpha':
        slm_explore =decrease_alpha
    elif args.explore_function == 'decrease_alpha_free_lr':
        slm_explore = decrease_alpha_free_lr
    
    # Start the scheduler.
    scheduler = PopulationBasedTraining(
        time_attr='training_iteration', 
        perturbation_interval=args.perturbation_interval,
        custom_explore_fn=slm_explore,
    )
    
    analysis = tune.run(train_RL, 
            resources_per_trial=resources,
            config=config, 
            metric="eval_rew_mean",
            mode='max',
            name=name,     
            scheduler=scheduler, 
            verbose=1,
            stop=stopper, 
            checkpoint_score_attr="eval_rew_mean",
            keep_checkpoints_num=args.num_samples,
            num_samples=args.num_samples,
            progress_reporter=reporter
    )