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

  def __init__(self, 
              init_joint_config = "random", 
              max_episode_length = 5000, 
              render_mode = "human",
              debug = False,
              log = 0
              ):
    super(Environment, self).__init__()
    
    self.debug = debug   
    self.log = log
    self.render_mode = render_mode
  
    # Env params
    self.sim = MjEnv(
      env_name = "panda_position",   
      max_episode_length = max_episode_length,
      init_joint_config = init_joint_config,
      actuators_type = MjEnv.POSITION_ACTUATOR,
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

    self.num_successful_episodes = 0
    self.reward = 0 
    self.action = None

    # Goal  
    self.goal = self.sim.get_obj_pos("target_point")
    self.achieved_goal = self.sim.get_obj_pos("end_effector")  

    # Actions  
    self.action_space = spaces.Box(low=-0.05, high=0.05, shape=self.sim.action_shape, dtype=np.float32)   

    # Observation space  
    self.observation_space =  spaces.Box(low=-1, high=1, shape=self.get_obs().shape, dtype=np.float32)

    self.reset()

  def _check_episode_truncate(self):
    '''
    Check if the episode should be truncated. 
    Check if the robot goes out of the workspace.
    '''
    target_pos = self.sim.get_obj_pos("target_point")
    eef_pos = self.sim.get_obj_pos("end_effector")  
    if self.WORKSPACE_BARRIERS is not None:  
      for i in range(len(eef_pos)):
        minv = self.WORKSPACE_BARRIERS[i][0]
        maxv = self.WORKSPACE_BARRIERS[i][1]
        if eef_pos[i]<minv or eef_pos[i]>maxv:
          if self.log > 0:
            print(f"EFF out of the workspace along axis {i}:  {eef_pos[i]} not in range [{minv},{maxv}]") 
          # exit("@@@@@@")
          return True
        
    # reaches the target
    # if np.linalg.norm(eef_pos-target_pos, axis = -1) < self.SUCCESS_THRESHOLD:
    #   self.num_successful_episodes += 1
    #   #print(f"{np.linalg.norm(eef_pos-target_pos, axis = -1)}<{self.SUCCESS_THRESHOLD}")
    #   return True 


    # the robot hits something
    # TODO

  
    # the robot hits something
    # TODO

    return False
 
  def get_reward(self, info): 
    target_pos = self.sim.get_obj_pos("target_point")
    eef_pos = self.sim.get_obj_pos("end_effector")  
    dist = np.linalg.norm(eef_pos-target_pos, axis = -1)   
    # r = 1/(dist+0.01)
    r = -dist 
    #print(r)
    return r

  def set_goal(self, goal):
    self.sim.set_site_pos("target_point",goal)
    self.goal = self.sim.get_obj_pos("target_point")

  def reset(self, hard_reset=True, random_goal=True, seed=0): 
    super().reset(seed=seed)
    if random_goal:
      new_goal = np.random.rand(len(self.goal))
      trange = self.TARGET_RANGE
      for i, r in enumerate(trange):  
        new_goal[i] = r[0] + (new_goal[i])*(r[1]-r[0])
      self.set_goal(new_goal)
    self.sim.reset(hard_reset=hard_reset) 
    self.obs = self.get_obs() 
    info = {}
    return (self.obs, info)
 
  def step(self, action):  
    info = {}   
    self.action = action  
    _, terminated = self.sim.execute(self.action)    
    self.reward = self.get_reward(info) 
    self.obs = self.get_obs() 
    truncated = self._check_episode_truncate() if not terminated else False 
    
    if self.debug:
      print(f"action={self.action}, reward={self.reward}, terminated={terminated}, truncated={truncated}")
      self.render() 

    return self.obs, self.reward, terminated, truncated, info

  def render(self, mode=None): 
    self.sim.render()
    pass 

  def get_obs(self):  
    sim_state = self.sim.get_state()
    qpos = sim_state[0:7]
    eef = np.array(sim_state[7:10])
    target = np.array(sim_state[10:13])

    eef_target_error = eef-target

    obs = np.concatenate([
      np.sin(qpos),
      np.cos(qpos), 
      np.tanh(eef_target_error),
    ]).astype(np.float32)  
    return obs