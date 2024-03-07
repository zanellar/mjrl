
from stable_baselines3.common.env_checker import check_env 
from mjrl.envs.panda_posctrl_posreach import PandaPositionReach


env = PandaPositionReach()
check_env(env) 