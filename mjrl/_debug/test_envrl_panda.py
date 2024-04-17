 
from mjrl.envs.panda_posctrl_posreach import Environment

env = Environment(
  max_episode_length=5000, 
  init_joint_config = [0, -1, 0, -3, 0, 0, 0, 0, 0],
  log=1
) 

obs = env.reset()
for i in range(1000):
    action = [0, 0.05, 0.05, 0.05, 0.05, 0, 0]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    print("@@@@@@@@@@@@@@@@@@:")
    print("obs:", obs)
    print("action:", action)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    # input("Press Enter to continue...")

    if terminated: 
      obs = env.reset()   

    if truncated:
      obs = env.reset()   
      print("Truncated")
      # input("Press Enter to continue...")
      
env.close()
