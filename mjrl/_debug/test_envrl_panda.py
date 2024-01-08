import sys
sys.path.append('/home/kvn/Super Duper Code/panda_mujoco/')
 
from mjrl.envrl.panda_torques_goalenv_reach import PandaTorquesGoalReach

env = PandaTorquesGoalReach(
  max_episode_length=5000, 
  init_joint_config = [0, -1, 0, -3, 0, 0, 0, 0, 0]
) 

obs = env.reset()
for i in range(10000):
    action = [0, 0.2, 0, 0.7, 0, 0, 0]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    print(obs, reward, terminated, truncated)
    if terminated or truncated: 
      obs = env.reset()    
      
env.close()
