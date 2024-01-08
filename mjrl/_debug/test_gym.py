from gym.envs.mujoco.reacher import ReacherEnv
import gymnasium as gym  

# env = ReacherEnv()
env = gym.make('HalfCheetah-v2')

joints = env.sim.get_state()[1][:2]

env.reset()
for i in range(60):
    _obs, _reward, _terminated, _truncated, _info = env.step([0,0,0,0,0,0]) 
    env.render()
    print(env.sim.get_state()[1][-6:])
    # print(_terminated)