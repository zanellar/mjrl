import os
import yaml 
import wandb  
import numpy as np
import torch
 
from mjrl.envs.panda_posctrl_posreach import Environment
from mjrl.utils.paths import LOGS_PATH, PARAMS_PATH, WEIGHTS_PATH
from mjrl.utils.evalwrap import EnvEvalWrapper
from mjrl.scripts.trainer import Trainer 

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser() 
parser.add_argument("--add_run", action="store_true", help="Add a run to an existing sweep")
args = parser.parse_args()
 
# Set Wandb Service timeout to 10 minutes
os.environ["WANDB__SERVICE_WAIT"] = "600"

# project name (used in wandb)
project = 'mjrl' 
 
# Load training config file (yaml)
config_file_path = os.path.join(PARAMS_PATH, "panda_posctrl_posreach", "sweep_tqc.yaml")
with open(config_file_path, "r") as stream: 
        config = yaml.safe_load(stream)  
  
# Initialize sweep or add a run to an existing sweep
if not args.add_run: 

    # Initialize sweep  
    sweep_id = wandb.sweep( sweep = config,  project = project)      

    # Save sweep_id in a file
    with open("_tmp_sweep_id.txt", 'w') as file:
        file.write(sweep_id)

else:  

    # Read sweep_id from file
    with open("_tmp_sweep_id.txt", 'r') as file:
        sweep_id = str(file.read())

# Set reward id
reward_id = Environment.POSITIVE_REWARD

# Initialize trainer
trainer = Trainer(
    env = Environment(
        max_episode_length=config["settings"]["expl_episode_horizon"], 
        render_mode="rgb_array",
        reward_id=reward_id
    ),  
    enveval = EnvEvalWrapper(
        Environment(
            max_episode_length=config["settings"]["eval_episode_horizon"], 
            render_mode="rgb_array",
            reward_id=reward_id
        ),
        settings = config["settings"],
        vars = ["dist"]
    ), 
    # enveval = Environment(
    #     max_episode_length=config["settings"]["eval_episode_horizon"], 
    #     render_mode="rgb_array"
    # ), 
    config = config, 
    sweep = True
)

# Start sweep job 
wandb.agent(sweep_id = sweep_id, function = trainer.run, project = project)
  
# Finish sweep job
wandb.finish()
  
 