import os 
import json 
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, DDPG, TD3, PPO
from sb3_contrib import TQC
 
from mjrl.envs.panda_posctrl_posreach_energyeval import Environment
from mjrl.utils.paths import LOGS_PATH, PARAMS_PATH, WEIGHTS_PATH, RESULTS_PATH
from mjrl.scripts.trainer import Trainer
from mjrl.utils.argsutils import Dict2Args 
from mjrl.utils.evalwrap import EnvEvalWrapper

# Load training config file (yaml)
config_file_path = os.path.join(PARAMS_PATH, "panda_posctrl_posreach", "test_tqc.yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream)   
 
# Set random seed
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

model_id = config["model_id"]

# Environment settings
env = Environment(
    max_episode_length=config["episode_horizon"], 
    render_mode="human",
    debug = False,
    settings={"motor_torque_coefficient": config["motor_torque_coefficient"]}
)  

# Load the best model
best_model_weights_path = os.path.join(WEIGHTS_PATH, "panda_posctrl_posreach", model_id, model_id, "best_model", "best_model.zip")
model = TQC.load(best_model_weights_path)
 
# Initialize measurements
dist_list = []
returns_list = []
episode_return = 0
step_energy_consumption = []
step_reward = []
energy_consumption_list = []
episode_energy_consumption = 0

# Test the model
obs, _ = env.reset()   


for ep_index in range(config["n_episodes"]): 
    for step_index in range(config["episode_horizon"]):  

        # Predict action
        action, _ = model.predict(observation=obs, deterministic=True)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)   
        
        # Measurements
        step_energy_consumption.append(info["energy_exiting"])
        episode_energy_consumption += info["energy_exiting"]
        step_reward.append(reward.item())
        episode_return += reward.item()
        dist_list.append(info["dist"])
 
        if config["render"]:
            env.render()
        if terminated or truncated: # BUG not working cam selection 
            print(f"Episode {ep_index} {'TRUNCATED' if truncated else 'terminated'} after {step_index} steps with energy consumption {episode_energy_consumption}")
            # break

    # End of episode
    obs, _ = env.reset()   
    returns_list.append(episode_return) 
    episode_return = 0 
    energy_consumption_list.append(episode_energy_consumption.tolist())
  
    episode_energy_consumption = 0

if config["save"]:
    file_path =  os.path.join(RESULTS_PATH, f"{model_id}.txt") 
    with open(file_path, 'w') as file:  
        file.write(returns_list)  
  
# print max energy consumption   
print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(f"Max Energy Consumption per step: {max(step_energy_consumption)}")
print(f"Min Energy Consumption per step: {min(step_energy_consumption)}")
print(f"Average Energy Consumption per step: {np.mean(step_energy_consumption)}")

# plot step energy consumption as bar chart
fig = plt.figure()
plt.fill_between(range(len(step_energy_consumption)), step_energy_consumption, color='blue', alpha=0.5)
plt.fill_between(range(len(step_energy_consumption)), step_reward, color='green', alpha=0.5) 
for i in range(config["n_episodes"]):
    episode_dist = dist_list[i*config["episode_horizon"]:(i+1)*config["episode_horizon"]]
    avg10last_distances_episode = np.mean(episode_dist[-10:])
    if avg10last_distances_episode < config["success_threshold"]:
        plt.plot(range(i*config["episode_horizon"], (i+1)*config["episode_horizon"]), episode_dist, color='black', alpha=0.5)
    else:  
        plt.plot(range(i*config["episode_horizon"], (i+1)*config["episode_horizon"]), episode_dist, color='red', alpha=0.2)
for i in range(config["n_episodes"]):
    plt.axvline(x=i*config["episode_horizon"], color='k', linestyle='-', alpha=0.2)
plt.xlabel('Step')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption per Step')
 
# same plot at above but every episode is overlaid with different color in the interval config["episode_horizon"]
fig = plt.figure()
for i in range(config["n_episodes"]):
    plt.fill_between(range(config["episode_horizon"]), step_energy_consumption[i*config["episode_horizon"]:(i+1)*config["episode_horizon"]], alpha=0.3)
    plt.plot(range(config["episode_horizon"]), dist_list[i*config["episode_horizon"]:(i+1)*config["episode_horizon"]], color='black', alpha=0.5)
plt.xlabel('Step')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption per Step')

# plot histogram of energy consumption
fig = plt.figure()
step_energy_consumption_nozeros = [x for x in step_energy_consumption if x > 0.001]
plt.hist(step_energy_consumption_nozeros, bins=200)
plt.xlabel('Energy Consumption')
plt.ylabel('Frequency [#steps]')
plt.title('Energy Consumption Histogram per Step')

# plot energy consumption per episode as histogram
fig = plt.figure()
plt.hist(energy_consumption_list, bins=200)
plt.xlabel('Energy Consumption')
plt.ylabel('Frequency [#episodes]')
plt.title('Energy Consumption per Episode Histogram')

plt.show()

