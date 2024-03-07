import math
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from mjrl.scripts.mjenv import MjEnv 
from mjrl.utils.gym import EnvGymBase

class Environment(EnvGymBase): 
  
  TARGET_RANGE = [[ 0.2, 0.8], [-0.3, 0.3], [0.4, 0.8]]
  # WORKSPACE_BARRIERS = [[0.0, 1], [-0.5, 0.5], [0.2, 1.0]]
  # WORKSPACE_BARRIERS = [[-1.5, 1.5], [-1.5, 1.5], [0.2, 1.5]]
  WORKSPACE_BARRIERS = None
  SUCCESS_THRESHOLD = 0.05
  
  NEGATIVE_REWARD = 0
  POSITIVE_REWARD = 1

  def __init__(self, 
              init_joint_config = "random", 
              max_episode_length = 5000, 
              render_mode = "human",
              reward_id = 0,
              debug = False,
              log = 0
              ):
    super(Environment, self).__init__()
    
    # Settings
    self.debug = debug   
    self.reward_id = reward_id
    self.log = log
    self.render_mode = render_mode
  
    # Simulator
    self.sim = MjEnv(
      env_name = "panda_torctrl",   
      max_episode_length = max_episode_length,
      init_joint_config = init_joint_config 
    )

    # get position of the workspace 
    # if self.WORKSPACE_BARRIERS is not None: 
      # ws_pos = np.zeros(3) 
      # ws_size = np.zeros(3)
      # for i in range(len(self.WORKSPACE_BARRIERS)):
      #   ws_pos[i] = (self.WORKSPACE_BARRIERS[i][0] + self.WORKSPACE_BARRIERS[i][1])/2
      #   ws_size[i] = (self.WORKSPACE_BARRIERS[i][1] - self.WORKSPACE_BARRIERS[i][0])/2
      # self.sim.set_site_pos("workspace",ws_pos)
      # self.sim.set_site_size("workspace",ws_size) 

    # Actions  
    self.action_space = spaces.Box(low=-1, high=1, shape=self.sim.action_shape, dtype=np.float32)   

    # Observation    
    self.observation_space =  spaces.Box(low=-1, high=1, shape=self.get_obs().shape, dtype=np.float32)

    # RL variables   
    self.action = None
    self.obs = None
    self.reward = None 
    self.terminated = False
    self.truncated = False
    self.info = {}

    # Initialize  
    self.num_successful_episodes = 0
    self.goal = self.sim.get_obj_pos("target_point")
    
    # Metrics
    self.dist = None
 
    # Reset
    self.reset()

  def _check_episode_truncate(self):
    '''
    Check if the episode should be truncated. 
    Check if the robot goes out of the workspace.
    ''' 
    if self.WORKSPACE_BARRIERS is not None:  
      for i in range(len(self.eef_pos)):
        minv = self.WORKSPACE_BARRIERS[i][0]
        maxv = self.WORKSPACE_BARRIERS[i][1]
        if self.eef_pos[i]<minv or self.eef_pos[i]>maxv:
          if self.log > 0:
            print(f"EFF out of the workspace along axis {i}:  {self.eef_pos[i]} not in range [{minv},{maxv}]") 
          # exit("@@@@@@")
          return True
        
    # reaches the target
    # if np.linalg.norm(self.eef_pos-self.goal, axis = -1) < self.SUCCESS_THRESHOLD:
    #   self.num_successful_episodes += 1
    #   #print(f"{np.linalg.norm(self.eef_pos-self.goal, axis = -1)}<{self.SUCCESS_THRESHOLD}")
    #   return True 
 
    # the robot hits something
    # TODO
  
    return False
 
  def get_reward(self, info):  
    self.eef_pos = self.sim.get_obj_pos("end_effector")  
    self.dist = np.linalg.norm(self.eef_pos-self.goal, axis = -1)   
    sim_state = self.sim.get_state()
    qpos = sim_state[0:7]
    qvel = sim_state[7:14]
    qtor = self.action
    if self.reward_id == self.NEGATIVE_REWARD:
      # reward = - self.dist - 0.1*np.linalg.norm(np.tanh(qvel)) - 0.01*np.linalg.norm(qtor) 
      reward = - self.dist - 0.03*np.linalg.norm(np.tanh(qvel)) - 0.03*np.linalg.norm(qtor)/(0.15+self.dist)  - 0.05*np.linalg.norm(qtor)/abs(1.15-min(1,self.dist))   
    elif self.reward_id == self.POSITIVE_REWARD:
      reward = 1/(1. + self.dist + 0.1*np.linalg.norm(np.tanh(qvel)) + 0.01*np.linalg.norm(qtor) ) 
    return reward

  def set_goal(self, goal):
    self.sim.set_site_pos("target_point",goal)
    self.goal = self.sim.get_obj_pos("target_point")

  def reset(self, random_goal=True, seed=0):  
    super().reset(seed=seed)

    # set goal
    if random_goal:
      new_goal = np.random.rand(len(self.goal))
      trange = self.TARGET_RANGE
      for i, r in enumerate(trange):  
        new_goal[i] = r[0] + (new_goal[i])*(r[1]-r[0])
      self.set_goal(new_goal)

    # reset simulation
    self.sim.reset(hard_reset=True)   

    # reset RL variables
    self.obs = self.get_obs() 
    self.action = None
    self.terminated = False
    self.truncated = False
    info = {}

    # reset metrics
    self.eef_pos = self.sim.get_obj_pos("end_effector")
    self.dist = np.linalg.norm(self.eef_pos - self.goal, axis = -1)

    return (self.obs, info)
 
  def step(self, action):  
    info = {}   
    self.action = action  
    _, self.terminated = self.sim.execute(self.action)    
    self.reward = self.get_reward(info) 
    self.obs = self.get_obs() 
    self.truncated = self._check_episode_truncate() if not self.terminated else False 
    
    if self.debug:
      print(f"action={self.action}, reward={self.reward}, terminated={self.terminated}, truncated={self.truncated}")
      self.render() 

    return self.obs, self.reward, self.terminated, self.truncated, info

  def render(self, mode=None): 
    self.sim.render()
    pass 

  def get_obs(self):  
    sim_state = self.sim.get_state()
    qpos = sim_state[0:7]
    qvel = sim_state[7:14]
    eef = np.array(sim_state[14:17])
    target = np.array(sim_state[17:20])
  
    obs = np.concatenate([
      np.sin(qpos),
      np.cos(qpos),
      np.tanh(qvel), 
      np.tanh(eef-target)
    ]).astype(np.float32)  
    return obs