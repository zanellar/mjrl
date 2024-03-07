from traceback import print_tb

import sys
sys.path.append('/home/kvn/Super Duper Code/panda_mujoco/')
from mjrl.scripts.env.mjenv import MjEnv
import numpy as np


'''
Test of ur5 environment using `mjenv` wrapper 
'''

 
#######################################################################

init_joint_config = [0, -1, 0, -3, 0, 0, 0, 0, 0]

env = MjEnv(
    env_name="panda_posctrl",   
    max_episode_length=1000,
    # init_joint_config= 'random'
    init_joint_config=init_joint_config 
    ) 
 
env.set_site_pos("target_point",[0.5, 0, 1])

test_action = [2.0, 0, 0, 0, 0, 0, 0]

while True:
    env.render()
    state, done = env.execute(test_action) 
    print(f"\n@@@ s={state}" )  

 