from tkinter import W
from traceback import print_tb
import mujoco_py
import os
import math
from mjrl.scripts.args.pkgpaths import PkgPath

'''
Test of the environment using `mujoco-py`
'''
 
xml_path = os.path.join(PkgPath.ENV_DESC_FOLDER,"pendulum_torctrl/arena.xml") 

# model
model = mujoco_py.load_model_from_path(xml_path)
# simulator
sim = mujoco_py.MjSim(model)

# simulate 1 step
sim.step()
 

# render
viewer = mujoco_py.MjViewer(sim)

for i in range(10000000):
    print(sim.data.qpos)
    viewer.render()  
    sim.step()  
 