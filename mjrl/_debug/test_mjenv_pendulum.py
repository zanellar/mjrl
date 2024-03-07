from traceback import print_tb

from mjrl.scripts.env.mjenv import MjEnv
import numpy as np


'''
Test of ur5 environment using `mjenv` wrapper 
'''

 
#######################################################################

init_joint_config = [0]

env = MjEnv(
    env_name="pendulum_torctrl",   
    max_episode_length=100,
    # init_joint_config= 'random'
    init_joint_config=init_joint_config 
    ) 
 
test_action = [2.0]

while True:
    env.render()
    state, done = env.execute(test_action)    
    print(f"\n@@@ state={state} " )  

 