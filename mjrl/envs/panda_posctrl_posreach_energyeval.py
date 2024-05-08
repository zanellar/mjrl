import math
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from mjrl.scripts.mjenv import MjEnv 
from mjrl.utils.gym import EnvGymBase

from mjrl.envs.panda_posctrl_posreach import Environment as PandaPosCtrlPosReachEnv
 
class Environment(PandaPosCtrlPosReachEnv): 
    
  def __init__(self, 
              init_joint_config = "random", 
              max_episode_length = 5000, 
              render_mode = "human",
              reward_id = 0,
              debug = False,
              log = 0,
              settings = None,
              ):
     
    super(Environment, self).__init__(
      init_joint_config = init_joint_config, 
      max_episode_length = max_episode_length, 
      render_mode = render_mode,
      reward_id = reward_id,
      debug = debug,
      log = log
    ) 

    self.motor_torque_coefficient = settings["motor_torque_coefficient"]
 
  def reset(self, random_goal=True, seed=0):  
    self.obs, info = super().reset(seed=seed)

    # Additional info on energy consumption
    self.energy_out = 0 
    info["energy_exiting"] = self.energy_out 
  
    # reset position and torque values
    self.qpos = np.array(self.sim.get_state()[0:7])
    self.qtor = np.array(self.sim.get_joints_ft()[0:7])

    return (self.obs, info)
  
  def step(self, action):  
    qpos_new = np.array(self.sim.get_state()[0:7])
    qtor_new = np.array(self.sim.get_joints_ft()[0:7])  # TODO: get torque from self.sim
 
    # Energy Exiting 
    self.energy_out = 0
    for i in range(len(qtor_new)):
      delta_energy = self.qtor[i]*(qpos_new[i] - self.qpos[i])
      if delta_energy >= 0:
        self.energy_out += delta_energy
      else:
        self.energy_out += self.motor_torque_coefficient*self.qtor[i]**2
   
    # Update qpos and qtor
    self.qpos = qpos_new
    self.qtor = qtor_new

    # Execute action and update RL variables
    self.obs, self.reward, self.terminated, self.truncated, info = super().step(action)

    # Additional info on energy consumption
    info["energy_exiting"] = self.energy_out 
 
    return self.obs, self.reward, self.terminated, self.truncated, info

  def render(self, mode=None): 
    super().render(mode=mode)

  def get_obs(self):   
    return super().get_obs()