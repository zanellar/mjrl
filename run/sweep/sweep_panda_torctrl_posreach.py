import os
import yaml 
import wandb  
import numpy as np
import torch
 
from mjrl.envs.panda_torctrl_posreach import Environment
from mjrl.utils.paths import LOGS_PATH, PARAMS_PATH, WEIGHTS_PATH
from mjrl.utils.evalwrap import EnvEvalWrapper
from mjrl.scripts.trainer import Trainer 
    
# Load training config file (yaml)
config_file_path = os.path.join(PARAMS_PATH, "panda_torctrl_posreach", "sweep_sac.yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream)  

# Initialize sweep by passing in config
sweep_id = wandb.sweep(
  sweep = config, 
  project = 'mjrl' 
)      

# Save sweep_id in a file
with open("_tmp_sweep_id.txt", 'w') as file:
    file.write(sweep_id)

trainer = Trainer(
    env = Environment(
        max_episode_length=config["settings"]["expl_episode_horizon"], 
        render_mode="rgb_array"
    ),  
    enveval = EnvEvalWrapper(
        Environment(
            max_episode_length=config["settings"]["eval_episode_horizon"], 
            render_mode="rgb_array"
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
wandb.agent(sweep_id, function=trainer.run)
  
# Finish sweep job
wandb.finish()
  
 