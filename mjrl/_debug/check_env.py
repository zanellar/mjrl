
from stable_baselines3.common.env_checker import check_env 
from mjrl.envs.panda_position_reach import PandaPositionReach


env = PandaPositionReach()
check_env(env) 