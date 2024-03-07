
import numpy as np
import math
from gymnasium import spaces

from mjrl.scripts.mjenv import MjEnv 
from mjrl.utils.gym import EnvGymBase
 
class Environment(EnvGymBase): 

  NEGATIVE_REWARD = 0
  POSITIVE_REWARD = 1

  def __init__(self, 
              max_episode_length=5000, 
              init_joint_config = [0], 
              init_joint_config_std_noise = 0,
              render_mode = "rgb_array",
              debug = False,
              log = 0, 
              hard_reset = False,
              reward_id = 0,
              ):
    super(Environment, self).__init__()
 
    # Settings
    self.debug = debug   
    self.log = log
    self.render_mode = render_mode 
    self.hard_reset = hard_reset
    self.reward_id = reward_id
    
    # Simulator
    self.sim = MjEnv( 
      env_name="pendulum_torctrl",  
      max_episode_length=max_episode_length,
      init_joint_config=init_joint_config,
      init_joint_config_std_noise=init_joint_config_std_noise
      ) 
 
    # Actions  
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.sim.action_shape, dtype=np.float32)   

    # Observations 
    self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) 
    
    # RL variables   
    self.action = None
    self.obs = None
    self.reward = None 
    self.terminated = False
    self.truncated = False
    self.info = {}

    # Metrics
    self.err_pos = None

    # Reset
    self.reset()

  def reset(self, goal=None, seed=0 ):
    """ 
    :return: (np.array) 
    """   
    super().reset(seed=seed)

    # reset simulation
    self.sim.reset(hard_reset=self.hard_reset )

    # reset RL variables
    self.obs = self.get_obs() 
    self.action = None
    self.terminated = False
    self.truncated = False
    info = {}

    # reset metrics
    self.err_pos = 1. - self.obs[0]

    return (self.obs, info) 

  def get_reward(self):   
    sin_pos, cos_pos,  tanh_vel = self.get_obs() 
    self.err_pos = 1. - sin_pos  
    torque = self.action[0]
    if self.reward_id == self.NEGATIVE_REWARD:
      reward = -abs(self.err_pos) -0.1*abs(tanh_vel) -0.01*abs(torque)
    elif self.reward_id == self.POSITIVE_REWARD:
      reward = 1/(1. + abs(self.err_pos) + 0.1*abs(tanh_vel) + 0.01*abs(torque)) 
    return reward

  def _check_episode_truncate(self):
    return False
  
  def step(self, action):   
    self.action = action  
    _, terminated = self.sim.execute(self.action) 
    self.reward = self.get_reward( )  
    self.obs = self.get_obs()
    self.terminated = terminated
    self.info = {}
    self.truncated = self._check_episode_truncate() if not terminated else False 
 
    return self.obs, self.reward, self.terminated, self.truncated, self.info

  def get_obs(self): 
    qpos, qvel = self.sim.get_state()
    self.obs = np.array([
      math.sin(qpos),
      math.cos(qpos),
      math.tanh(qvel)
    ])   
    return self.obs

  def render(self, mode=None): 
    self.sim.render()

  def get_sample(self):
    return self.obs, self.action, self.reward, self.terminated, self.truncated, self.info