import os
import yaml 
import wandb  
import numpy as np
import torch
 
from mjrl.envs.panda_posctrl_posreach_energylimit1 import Environment
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

# Exploration environment settings  
env_settings = dict(
    energy_margin = 10,  
    motor_torque_coefficient = 0,
    k_task = 2,
    k_energy = 0.1,
    tank_min_threshold = 5,  
    energy_tank_init = 1000,
    eval_env = False
)

# Evaluation environment settings
eval_env_settings = env_settings.copy()
eval_env_settings["eval_env"] = True 

# Set reward id
reward_id = Environment.POSITIVE_REWARD
  
trainer = Trainer(
    env = Environment(
        max_episode_length=config["settings"]["expl_episode_horizon"], 
        render_mode="human",
        debug = False,
        settings = env_settings,
        reward_id = reward_id
    ),  
    enveval = EnvEvalWrapper(
        Environment(
            max_episode_length=config["settings"]["eval_episode_horizon"], 
            render_mode="human",
            debug = False,
            settings = eval_env_settings,
            reward_id = reward_id
        ),
        settings = config["settings"],
        vars = ["dist"],
        logs=["avg"]
    ),
    config = config , 
    sweep = True
) 

# Start sweep job 
wandb.agent(sweep_id = sweep_id, function = trainer.run, project = project)
  
# Finish sweep job
wandb.finish()
  
 