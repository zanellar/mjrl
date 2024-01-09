import os
import yaml 
import wandb 
 
from mjrl.envs.panda_position_reach import Environment
from mjrl.utils.paths import LOGS_PATH, PARAMS_PATH, WEIGHTS_PATH
from mjrl.scripts.trainer import Trainer
from mjrl.utils.argsutils import Dict2Args 
    
# Load training config file (yaml)
config_file_path = os.path.join(PARAMS_PATH, "sac_panda_pos_reach.yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream)   

# Wandb initialization
run = wandb.init(
    sync_tensorboard = True, 
    project = "mjrl",
    # monitor_gym = True, # automatically upload gym environements' videos
    # save_code = True,
    mode = "online" # disable wandb
) 
  
trainer = Trainer(
    env = Environment(
        max_episode_length=config["settings"]["expl_episode_horizon"], 
        render_mode="human",
        debug = False
    ),  
    enveval = Environment(
        max_episode_length=config["settings"]["eval_episode_horizon"], 
        render_mode="human"
    ),
    config = config 
) 
 
trainer.run()

wandb.finish()