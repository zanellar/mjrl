import sys
sys.path.append('/home/kvn/Super Duper Code/panda_mujoco/')
 
from mjrl.envs.panda_position_reach import Environment

env = Environment(
  max_episode_length=5000, 
  init_joint_config = [0, -1, 0, -3, 0, 0, 0, 0, 0],
  log=1
) 

obs = env.reset()
for i in range(1000):
    action = [0, 0.2, 0, 0.7, 0, 0, 0]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    print(obs, reward, terminated, truncated)
    if terminated: 
      obs = env.reset()   

    if truncated:
      obs = env.reset()   
      print("Truncated")
      # input("Press Enter to continue...")
      
env.close()
