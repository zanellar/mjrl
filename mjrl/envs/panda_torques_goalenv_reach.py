import numpy as np
import gymnasium as gym 
from gymnasium import spaces 

from mjrl.scripts.mjenv import MjEnv
from mjrl.utils.gym import EnvGymGoalBase
 
class PandaTorquesGoalReach(EnvGymGoalBase): 
  
  TARGET_RANGE = [[0.3,0.6],[-0.5,0.5],[0.3, 0.8]]
  WORKSPACE_BARRIERS = [[0.0,0.9],[-0.9,0.9],[0.2, 1.0]]

  def __init__(self, 
              max_episode_length = 5000, 
              init_joint_config = 'random',
              render_mode = "rgb_array",
              debug = False,
              log = 0
              ):
    super(PandaTorquesGoalReach, self).__init__()
    
    self.debug = debug   
    self.log = log
    self.render_mode = render_mode
  
    
    # Env params
    self.sim = MjEnv(
      env_name = "panda_torque",
      max_episode_length=max_episode_length,
      init_joint_config=init_joint_config 
    )

    self.num_successful_episodes = 0
    self.reward = 0
    self.obs = None
    self.action = None

    # Goal  
    self.goal = self.sim.get_obj_pos("target_point") 
    eef_pos = self.sim.get_obj_pos("end_effector")  

    # Actions  
    self.action_space = spaces.Box(low=-1, high=1., shape=self.sim.action_shape, dtype=np.float32)   

    # Observation space  
    self.observation_space = spaces.Dict(dict(
            observation = spaces.Box(low=-1, high=1, shape=(7*2+7+3,), dtype=np.float32),
            desired_goal = spaces.Box(low=-1, high=1.2, shape=self.goal.shape, dtype=np.float32),
            achieved_goal = spaces.Box(low=-1, high=1.2, shape=eef_pos.shape, dtype=np.float32),
        )) 

  def _check_episode_truncate(self):
    eef_pos = self.obs["achieved_goal"]
    target_pos = self.obs["desired_goal"]

    # # the gripper goes out of the desired work space
    # for i in range(len(eef_pos)):
    #   minv = self.WORKSPACE_BARRIERS[i][0]
    #   maxv = self.WORKSPACE_BARRIERS[i][1]
    #   if eef_pos[i]<minv or eef_pos[i]>maxv:
    #     print(f"dim {i}:  {eef_pos[i]} out of [{minv},{maxv}]")
    #     return True

    # # reaches the target
    # epsilon = 0.05
    # if np.linalg.norm(eef_pos-target_pos, axis = -1)<epsilon:
    #   self.num_successful_episodes += 1
    #   print(f"{np.linalg.norm(eef_pos-target_pos, axis = -1)}<{epsilon}")
    #   return True 

    # the robot hits something
    # TODO

    return False
 
  def compute_reward(self ):  
    dist = np.linalg.norm(self.obs["achieved_goal"] - self.obs["desired_goal"] , axis = -1)   
    # dq = abs(sum(self.obs["observation"][7:14]))
    r = - dist #- dq
    # r = -(dist >= 0.05).astype(np.float32) 
    return r

  def set_goal(self, goal):
    self.sim.set_site_pos("target_point",goal)
    self.goal = self.sim.get_obj_pos("target_point")

  def reset(self, hard_reset=True, goal="random", seed=0): 
    ''' @goal: "random", None, new_goal (list)'''
    super().reset(seed=seed)
    if goal=="random": 
      new_goal = np.random.rand(len(self.goal))
      trange = self.TARGET_RANGE
      for i, r in enumerate(trange):  
        new_goal[i] = r[0] + (new_goal[i])*(r[1]-r[0]) 
    elif goal is None:
      new_goal = self.goal
    else: 
      new_goal = goal

    self.set_goal(new_goal)

    self.sim.reset(hard_reset=hard_reset) 
    return self.get_obs() 
 
  def step(self, action):  
    info = {}   
    self.action = action  
    _, terminated = self.sim.execute(self.action)  
    self.obs = self.get_obs()  
    self.reward = self.compute_reward() 
 
    if not terminated: 
      truncated = self._check_episode_truncate() 

    return self.obs, self.reward, terminated, truncated, info

  def render(self, mode=None): 
    self.sim.render()
 
  def get_obs(self):  
    sim_state = self.sim.get_state()
    q = sim_state[0:7]
    dq = sim_state[7:14]
    eef = np.array(sim_state[14:17])
    target = np.array(sim_state[17:])

    obs = np.array([
      np.sin(q),
      np.cos(q),
      np.tanh(dq),
      eef-target
    ])  
 
    self.obs = dict(
        observation = obs, 
        achieved_goal = np.array(eef), 
        desired_goal = np.array(target)
    )

    return self.obs