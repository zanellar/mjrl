import os 
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from mjrl.scripts.args.pkgpaths import PkgPath
from mjrl.envrl.pendulum import Pendulum


AGENT_WEIGHTS_PATH = os.path.join(PkgPath.OUT_TEST_FOLDER,"/home/riccardo/projects/mjrlenvs/data/train/pendulum/3/checkpoints/SAC_1_0/all_cps/rl_model_162500_steps.zip")
EPISODE_HORIZON = 500 # timesteps 
RENDERING = True  
NUM_EVAL_EPISODES = 1 

########################################################################
########################################################################
########################################################################

########################################################################

if not os.path.exists(os.path.join(PkgPath.OUT_TEST_FOLDER,)):
    os.makedirs(os.path.join(PkgPath.OUT_TEST_FOLDER,))

with open(os.path.join(PkgPath.OUT_TEST_FOLDER,"output.txt"), 'w') as f: 
    line = "timestep,energy_tank,energy_exchanged" 
    f.write(line)
    f.close()
 
########################################################################
  
env = Pendulum(
            max_episode_length=EPISODE_HORIZON, 
            init_joint_config = [0]
        )
 
env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
env = DummyVecEnv([lambda : env])     # Needed for all environments (e.g. used for mulit-processing)
env = VecNormalize(env)               # Needed for improving training when using MuJoCo envs? 

########################################################################


model = SAC.load(
    path=os.path.join(PkgPath.OUT_TRAIN_FOLDER,AGENT_WEIGHTS_PATH), 
    env=env
    )

env.reset()   
print(env)
model.learn(
    total_timesteps = EPISODE_HORIZON*10, 
    log_interval = 1,  
    tb_log_name = "SAC_1_0_continue",
    callback = EvalCallback(
                        env, 
                        eval_freq = EPISODE_HORIZON,
                        n_eval_episodes = 1, 
                        deterministic = True, 
                        render = True
                    )
)  