import math
import wandb
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from mjrl.scripts.mjenv import MjEnv 
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
    
    # Energy  
    self.energy_margin = settings["energy_margin"] 
    self.energy_tank_init = settings["energy_tank_init"]
    self.motor_torque_coefficient = settings["motor_torque_coefficient"]
    self.k_task = settings["k_task"]
    self.k_energy = settings["k_energy"] 
    self.energy_out = 0
    self.energy_tank = self.energy_tank_init
    self.tank_min_threshold = settings["tank_min_threshold"]
    self.eval_env = settings["eval_env"]
    self.energy_excess_ct = 0
  
    super(Environment, self).__init__(
      init_joint_config = init_joint_config, 
      max_episode_length = max_episode_length, 
      render_mode = render_mode,
      reward_id = reward_id,
      debug = debug,
      log = log
    )   

  def get_reward(self, info):  

    # Task metrics
    self.eef_pos = self.sim.get_obj_pos("end_effector")  
    self.dist = np.linalg.norm(self.eef_pos-self.goal, axis = -1)   
    sim_state = self.sim.get_state()
    qpos = sim_state[0:7]
    qvel = sim_state[7:14] 

    # Energy metrics
    qpos_new = np.array(self.sim.get_state()[0:7])
    qtor_new = np.array(self.sim.get_joints_ft()[0:7])   
    energy_out = self._get_energy_out(qpos_new, qtor_new)
    penality_energy_tank =  (self.energy_tank_init - self.energy_tank)/(self.energy_tank_init + self.energy_tank) # defined in [0,1]
    penality_energy_out = energy_out/self.energy_margin # defined in [0,1+]
    energy_penalty = penality_energy_tank + penality_energy_out

    # Negative Reward
    if self.reward_id == self.NEGATIVE_REWARD:
 
      reward = - self.k_task*self.dist - self.k_energy*energy_penalty  

    # Positive Reward
    elif self.reward_id == self.POSITIVE_REWARD: 

      reward = 1/(0.01 + self.k_task*self.dist + self.k_energy*energy_penalty)   

    # Additional energy penalty
    if energy_out > self.energy_margin: 
      reward = -10*abs(reward)

    return reward
 
  def reset(self, random_goal=True, seed=0):  
    self.obs, self.info = super().reset(seed=seed, random_goal=random_goal)
   
    # Additional info on energy consumption
    self.energy_out = 0 
    self.info["energy_exiting"] = self.energy_out 
   
    # reset energy  
    self.qpos = np.array(self.sim.get_state()[0:7])
    self.qtor = np.array(self.sim.get_joints_ft()[0:7]) 

    self.energy_tank = self.energy_margin

    return (self.obs, self.info)
  
  def _get_energy_out(self, qpos_new, qtor_new):  

    # Energy Exiting 
    energy_out = 0
    for i in range(len(qtor_new)):
      delta_energy = self.qtor[i]*(qpos_new[i] - self.qpos[i])
      if delta_energy >= 0:
        energy_out += delta_energy
      else:
        energy_out += self.motor_torque_coefficient*self.qtor[i]**2  
   
    return energy_out
 
  def step(self, action):  


    if self.energy_out < self.energy_margin and self.energy_tank > self.tank_min_threshold: 

      # Execute action and update RL variables
      self.obs, self.reward, self.terminated, self.truncated, self.info = super().step(action)

      qpos_new = np.array(self.sim.get_state()[0:7])
      qtor_new = np.array(self.sim.get_joints_ft()[0:7])  

      # Energy Exiting
      self.energy_out = self._get_energy_out(qpos_new, qtor_new)

      # Update energy tank
      self.energy_tank -= self.energy_out

      # Update qpos and qtor
      self.qpos = qpos_new
      self.qtor = qtor_new

    else:  

      # Truncate episode if energy exceeds margin
      self.obs = self.get_obs()
      self.reward = self.get_reward(self.info)
      self.terminated = False
      self.truncated = True 
      self.energy_excess_ct += 1

    if self.eval_env:
      wandb.log({f"eval/energy_budget": self.energy_margin - self.energy_out})

    if self.truncated or self.terminated:
      if self.eval_env:
        wandb.log({f"eval/final_energy_budget": self.energy_margin - self.energy_out})
        wandb.log({f"eval/final_energy_tank": self.energy_tank - self.tank_min_threshold}) 
        wandb.log({f"eval/energy_excess_ct": self.energy_excess_ct})
      else:
        wandb.log({f"rollout/energy_excess_ct": self.energy_excess_ct})
        wandb.log({f"rollout/energy_excess_ct": self.energy_excess_ct})
        wandb.log({f"rollout/final_energy_budget": self.energy_margin - self.energy_out})

    # Additional info on energy consumption 
    self.info["energy_exiting"] = self.energy_out  
    
    # Debug 
    if self.debug:
      print(f"energy_exiting={self.energy_out}, energy_tank={self.energy_tank}")
  
    return self.obs, self.reward, self.terminated, self.truncated, self.info
 

  def get_obs(self):  
    nominal_obs = super().get_obs() 

    obs = np.concatenate([
      nominal_obs,
      np.array([self.energy_tank/self.energy_tank_init])
    ]).astype(np.float32)  
    return obs