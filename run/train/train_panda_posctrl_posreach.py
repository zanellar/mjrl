import os
import yaml 
import wandb 
import numpy as np
import torch
 
from mjrl.envs.panda_posctrl_posreach import Environment
from mjrl.utils.paths import LOGS_PATH, PARAMS_PATH, WEIGHTS_PATH
from mjrl.scripts.trainer import Trainer
from mjrl.utils.argsutils import Dict2Args 
from mjrl.utils.evalwrap import EnvEvalWrapper
    
# Load training config file (yaml)
config_file_path = os.path.join(PARAMS_PATH, "panda_posctrl_posreach", "train_tqc.yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream)   
 
# Set random seed
torch.manual_seed(config["parameters"]["seed"])
np.random.seed(config["parameters"]["seed"])

# Wandb initialization
run = wandb.init(
    sync_tensorboard = True, 
    project = "mjrl",
    # monitor_gym = True, # automatically upload gym environements' videos
    # save_code = True,
    # mode = "online"  
    mode = "disabled"  
) 
  
trainer = Trainer(
    env = Environment(
        max_episode_length=config["settings"]["expl_episode_horizon"], 
        render_mode="human",
        debug = False
    ),  
    enveval = EnvEvalWrapper(
        Environment(
            max_episode_length=config["settings"]["eval_episode_horizon"], 
            render_mode="human"
        ),
        settings = config["settings"],
        vars = ["dist"] 
    ),
    config = config 
) 
 
trainer.run()

wandb.finish()