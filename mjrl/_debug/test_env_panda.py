from tkinter import W
from traceback import print_tb
import mujoco_py
import os
import math

'''
Test of the environment using `mujoco-py`
'''

# mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'xmls', 'ur5.xml')
xml_path = 'home/kvn/Super Duper Code/panda_mujoco/panda_mujoco/assets/panda.xml'

# model
model = mujoco_py.load_model_from_path('/home/kvn/Super Duper Code/panda_mujoco/panda_mujoco/assets/panda.xml')
# simulator
sim = mujoco_py.MjSim(model)

# simulate 1 step
sim.step()

# environment state
state = sim.get_state()  # sim.data.qpos
t = state.time
q = state.qpos#[sim.model.get_joint_qpos_addr("target_point")]
dq = state.qvel
print("\nTime= {}".format(t))
print("\nJoints Positions = {}".format(q))
print("\nJoints Velocities = {}".format(dq))

# cartesian states
T = sim.data.get_site_xpos("end_effector")  # w.r.t. world (not coincident with robot base)
R = sim.data.get_site_xmat("end_effector")
# r = sim.data.get_site_quat("end_effector")
print("\nEEF Position = {}".format(T))
print("\nEEF Rotation Matrix = \n{}".format(R))
# print("\nEEF Quaternion = \n{}".format(r))

# set initial pose
sim.data.qpos[0] = 0
sim.data.qpos[1] = -math.pi/4
sim.data.qpos[2] = 0
sim.data.qpos[3] = -3*math.pi/4
sim.data.qpos[4] = 0
sim.data.qpos[5] = 0
sim.data.qpos[6] = 0
 

for i in range(len(sim.data.qpos)) :
    sim.data.ctrl[i] = sim.data.qpos[i]


# render
viewer = mujoco_py.MjViewer(sim)

for i in range(10000000):
    viewer.render() 
    sim.data.ctrl[3] += 0.001
    sim.step() 
    if sim.data.qpos[3] > -math.pi/2 :
        break


for i in range(10000000):
    viewer.render() 
    sim.data.ctrl[1] += 0.001
    sim.step() 

    if sim.data.qpos[1] > math.pi/4 :
        break



for i in range(10000000):
    viewer.render() 
    sim.data.ctrl[0] = 0
    sim.data.ctrl[1] = -math.pi/4
    sim.data.ctrl[2] = 0
    sim.data.ctrl[3] = -3*math.pi/4
    sim.data.ctrl[4] = 0
    sim.data.ctrl[5] = 0
    sim.data.ctrl[6] = 0
    sim.step() 







# for i in range(1000000):
#     viewer.render()
#     T = sim.data.get_site_xpos("end_effector")  # w.r.t. world (not coincident with robot base)
#     R = sim.data.get_site_xmat("end_effector")
#     # r = sim.data.get_site_quat("end_effector")
#     print("\nEEF Position = {}".format(T))
#     print("\nEEF Rotation Matrix = \n{}".format(R)) 

# for i in range(1000):
#     # sim.data.qpos[0] = 0
#     # sim.data.qpos[1] = 0 
#     # sim.data.qpos[2] = 0 
#     sim.data.qpos[3] = -2
#     # sim.data.qpos[4] = 0 
#     # sim.data.qpos[5] = 0
#     # sim.data.qpos[6] = 0
#     viewer.render()
#     sim.step()
#     print("\n  2_joints= {}".format(sim.data.get_joint_qpos("panda_joint4")))

# state = sim.get_state()
# print("\nJoints={}".format(state.qpos))

# # control motors
# sim.data.ctrl[3] = -1.5
# sim.data.ctrl[2] = -1.5
# for i in range(5000): 
#     sim.data.qpos[0] = 0
#     sim.data.qpos[1] = 0 
#     sim.data.qpos[2] = 0 
#     sim.data.ctrl[3] = +10
#     sim.data.qpos[4] = 0 
#     sim.data.qpos[5] = 0
#     sim.data.qpos[6] = 0 
#     viewer.render()
#     sim.step() 
#     # print("\n EEF velocity= {}".format(sim.data.get_site_xvelp("end_effector"))) 
#     print("\n  3_joints= {}".format(sim.data.get_joint_qpos("panda_joint4")))
