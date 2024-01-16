 
import gymnasium as gym  
import numpy as np
from gymnasium import spaces

class EnvGymBase(gym.Env): 

  '''
  Base class for all gym environments
  '''

  def __init__(self,  
              debug = False
              ):
    super(EnvGymBase, self).__init__()

    self.obs = None
    self.action = None
    self.reward = None
  
  def reset(self, seed=None): 
    super().reset(seed=seed)
    pass
  
  def get_reward(self):  
    pass

  def set_goal(self, goal):
    pass
     
  def step(self, action):  
    pass

  def render(self, mode=None): 
    pass

  def close(self):
    pass

  def get_obs(self):
    pass
 
  def seed(self, seed=None):
    pass

  def get_sample(self):
    return self.obs, self.action, self.reward


class EnvGymGoalBase(EnvGymBase): 

  def __init__(self,  
              debug = False
              ):
    super(EnvGymGoalBase, self).__init__()

  def compute_reward(self):  
    pass