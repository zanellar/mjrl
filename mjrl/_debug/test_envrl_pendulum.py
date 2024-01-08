 
from mjrl.envrl.pendulum import Pendulum

env = Pendulum(
  max_episode_length=500, 
  init_joint_config = [0]
) 

obs = env.reset()
for i in range(10000):
    action = [0.3]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    print(f"obs={obs}, reward={reward}, terminated = {terminated}, truncated={truncated}")
    if terminated or truncated: 
      obs = env.reset() 
      
env.close()
