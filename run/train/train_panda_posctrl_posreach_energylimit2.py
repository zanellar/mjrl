import os
import yaml 
import wandb 
import numpy as np
import torch
 
from mjrl.envs.panda_posctrl_posreach_energylimit0 import Environment
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

# Environment settings
env_settings = dict(
    energy_margin = 5,  
    motor_torque_coefficient = 0,
    k_task = 1,
    k_energy = 1,
    history_length = 5
)
# Set reward id
reward_id = Environment.POSITIVE_REWARD
  
trainer = Trainer(
    env = Environment(
        max_episode_length=config["settings"]["expl_episode_horizon"], 
        render_mode="human",
        debug = True,
        settings = env_settings,
        reward_id = reward_id
    ),  
    enveval = EnvEvalWrapper(
        Environment(
            max_episode_length=config["settings"]["eval_episode_horizon"], 
            render_mode="human",
            debug = False,
            settings = env_settings,
            reward_id = reward_id
        ),
        settings = config["settings"],
        vars = ["dist"] 
    ),
    config = config 
) 
 
trainer.run()

wandb.finish()