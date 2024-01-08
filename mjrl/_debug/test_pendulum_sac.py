import os
import numpy as np  
from mjrl.scripts.eval.tester import TestRun
from mjrl.scripts.args.pkgpaths import PkgPath
from mjrl.run.pendulum_sac import Args

########################################################################

agent = "SAC_1_0"

tester = TestRun(Args)
 
# tester.loadmodel(run_id)
    # tester.registercallback()
    mean, std = tester.eval_returns_model(n_eval_episodes=5, render=False)
    print(mean, std)
tester.plot(show=False)
    # tester.infer()

########################################################################
########################################################################
########################################################################

########################################################################

# if not os.path.exists(os.path.join(PkgPath.OUT_TEST_FOLDER,)):
#     os.makedirs(os.path.join(PkgPath.OUT_TEST_FOLDER,))

# with open(os.path.join(PkgPath.OUT_TEST_FOLDER,"output.txt"), 'w') as f: 
#     line = "timestep,energy_tank,energy_exchanged" 
#     f.write(line)
#     f.close()

########################################################################

# def energycallback(locals_, _globals): 
#     energy_tank = locals_["info"]["energy_tank"]
#     print(energy_tank)
#     energy_exchanged = locals_["info"]["energy_exchanged"]
#     t = locals_["current_lengths"][0]
#     with open(os.path.join(PkgPath.OUT_TEST_FOLDER,"output.txt"), 'a') as f: 
#         line = f"\n{t},{energy_tank},{energy_exchanged}" 
#         f.write(line)
#         f.close()
  
