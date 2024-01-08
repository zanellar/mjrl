import mujoco_py
import numpy as np
import random
import json
import os 
import glfw

from mjrl.utils.paths import ENVS_PATH

''' 
DOCUMENTATION: 
https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/generated/wrappers.pxi


'''

    # self._sim.model.actuator_ctrlrange
    # self._sim.model.actuator_names

    # self._sim.model.jnt_range
    # self._sim.model.joint_names

    # self._sim.model.site_range
    # self._sim.model.site_names
  

class MjEnv(object):

    def __init__(
        self, 
        env_name,
        controller=None,   
        specs=None, 
        init_joint_config=None, 
        init_joint_config_std_noise=0,
        folder_path=None,  
        max_episode_length=None 
        ):
        '''
        This class is used to create a MuJoCo environment.

        

        @param env_name: (string) name of an environment   
        @param init_joint_config: if None it starts from model defined default position | joints (list) starts from a specific joints config | "random" (string) starts from random position
        @param folder_path: (string) path to the folder where the xml file is contained
        @param episode_terminator: instance of class EpisodeTerminator with method done(state, time) that returns a True when end the current episode

        --- variables ---

        state: (list) RL environment states (specification in variable states_specification)
        action: (list) agent action (specification in variable actions_specification) 
        done: (bool) episode termination
        states_shape: (tuple) states shape
        states_type: (tuple) states type
        actions_shape: (tuple) actions shape
        actions_type: (tuple) actions type
        states_specification: (dict) {'shape': (tuple) states_shape, 'type': (str) states_type}
        actions_specification: (dict)('shape': (tuple) actions_shape, 'type': (str) actions_type}


        Example:

            env = MjEnv(
                env_name="ur5",  
                max_episode_length=5000,
                init_joint_config=[0, 0, 0, 0, 0, 0])

            actions = [0,0,0,0,0,0]
            state, done = env.execute(actions)
            env.render()

        '''

        self.env_name = env_name
        self.controller = controller
        self.specs = specs
        
        env_data_folder = ENVS_PATH if folder_path is None else folder_path
 
        ##### SPECIFICATIONS #### 
        param_path = os.path.join(env_data_folder, self.env_name, "specs.json")  
  
        with open(param_path, 'r') as fi:
            param_spec = json.loads(fi.read()) 
            self._states_specs = param_spec["states"] 
            self._actions_specs = param_spec["actions"] 
            self._env_params = param_spec["environment"] 

        num_states = np.sum([int(sdata["dim"]) for sname, sdata in self._states_specs.items()], dtype=int)  
        self.state_shape = (num_states,) 
        num_actions = len(list(self._actions_specs.keys()))
        self.action_shape = (num_actions,)  

        self.states_specification = dict(shape=tuple(self.state_shape), type="float")
        self.actions_specification = dict(shape=tuple(self.action_shape), type="float")
 

        ##### MODEL #### 
        xml_path = os.path.join(env_data_folder, self.env_name, "arena.xml")  
        self._mjmodel = mujoco_py.load_model_from_path(xml_path)

        ##### SIMULATOR ####

        # sim & ctrl frequencies
        self.simulation_frequency = 1./self._mjmodel.opt.timestep
        self.control_frequency = self._env_params["control_frequency"]
        nsubsteps = round(self.simulation_frequency/self._env_params["control_frequency"])
        print(f"nsubsteps={nsubsteps}")
        assert nsubsteps > 0, f"expected 'control_frequency'({self.control_frequency})<='simulation_frequency'({self.simulation_frequency}) " 

        self._sim = mujoco_py.MjSim(self._mjmodel, nsubsteps=nsubsteps)

        # viewer 
        self._viewer = None
    
        # starting position
        self._init_joint_config = init_joint_config
        self._init_joint_config_std_noise = init_joint_config_std_noise
        if self._init_joint_config is not None:
            self.set_joints_pos(self._init_joint_config)

        # --- support variables --
        self._render_ct = 0
        self._episode_max_time = max_episode_length
        self._site_forced = {}
        self._body_forced = {}

        # --- public variables ---
        self.fixed_frame_name = self._env_params["fixed_frame"] 
        self.fixed_frame = self.env_fixed_frame(self.fixed_frame_name)
        self.episode_time = 0
        self.episode_index = 0
        self.state = self.get_state()  # state
        self.action = None  # action 
        self.done = False
 
    def _refresh(self):
        for name,pos in self._site_forced.items():
            self.set_site_pos(name,pos) 
        for name,pos in self._body_forced.items():
            self.set_body_pos(name,pos) 
 
    def execute(self, action):
        ''' Takes a sorted list of actions (float), that are the torques to the motors'''

        self.action = action 

        for i in range(len(action)):
            self._sim.data.ctrl[i] = self.action[i]
        
        self._sim.step()

        self._refresh()
        self.episode_time += 1
        self.state = self.get_state() 
        if self._episode_max_time is not None:
            if self.episode_time >= self._episode_max_time:
                self.done = True
                return self.state, self.done 
          
        return self.state, self.done
 

    def reset(self, hard_reset=True, initi_pos=None ):
        ''' reset the episode simulation.
        The robot default initial position is the one specified in the constructor.
        Otherwise it can start from a new position `initi_pos`'''

        if hard_reset:  
            nsubsteps = round(self.simulation_frequency/self._env_params["control_frequency"])
            self._sim = mujoco_py.MjSim(self._mjmodel, nsubsteps=nsubsteps)
            if self._viewer is not None:
                glfw.destroy_window(self._viewer.window) 
                self._viewer = mujoco_py.MjViewer(self._sim)
 
        self.episode_index += 1
        self.episode_time = 0
        self.done = False
        self._sim.reset()
        if initi_pos is not None:
            self.set_joints_pos(initi_pos)
        elif self._init_joint_config is not None:
            initi_pos = self._init_joint_config 
            self.set_joints_pos(initi_pos) 

    def render(self, episode_step=0, do_first=False):
        ''' render and visualize 1 time step of the simulation. Must be called in a loop.
            episode_step is the number of episodes to skip'''
 
        # we create the viewer in the first render call
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)

        if self.done:
            self._render_ct += 1
            if self._render_ct > int(episode_step):
                self._render_ct = 0

        if self._render_ct == int(episode_step) or (do_first and self.episode_index == 0):
            self._viewer.render()
            return True
        else:
            return False
  
    def set_joints_pos(self, joints): # BUG controllare dove usato... 
        joint_ranges = self._sim.model.jnt_range
        joint_names = self._sim.model.joint_names 
        for i, name in enumerate(joint_names):   
            joint_index = self._sim.model.get_joint_qpos_addr(name)
            r = joint_ranges[joint_names.index(name)]
            if joints == 'random':  
                jval = random.uniform(r[0], r[1])
            else: 
                dis = np.random.randn()*self._init_joint_config_std_noise 
                jval = joints[i] + dis
                if jval<r[0]:
                    jval = r[0]
                if jval>r[1]:
                    jval = r[1] 
            self._sim.data.qpos[joint_index] = jval
        self._sim.forward()
        # self._sim.step()

    def set_site_pos(self, name, pos):
        ''' 
        set the position of a site in the simulation
        @param name: name of the site
        @param pos: position of the site
        '''
        self._site_forced[name] = pos
        site_id = self._sim.model.site_name2id(name)
        self._sim.data.site_xpos[site_id] = pos

    def set_body_pos(self, name, pos):
        '''
        set the position of a body in the simulation
        @param name: name of the body
        @param pos: position of the body
        '''
        self._body_forced[name] = pos
        body_id = self._sim.model.body_name2id(name)
        self._sim.data.body_xpos[body_id] = pos
        
    def get_obj_pos(self, objname, objtype="site", objvar="xpos"):  
        '''
        get the position of an object in the simulation
        @param objname: name of the object
        @param objtype: type of the object (site, body, geom, joint, ...)
        @param objvar: variable of the object (xpos, xmat, ...)
        '''
        attr_get_value = getattr(self._sim.data,f"get_{objtype}_{objvar}") 
        pos = attr_get_value(objname)  
        return pos.copy()

    def get_joints_pos(self, ids=None):   
        '''
        Get the position of the joints
        '''
        if ids is None:
            joints_pos = self._sim.data.qpos
        else:
            joints_pos = self._sim.data.qpos[ids]
        return joints_pos.copy()

    def get_joints_vel(self, ids=None):   
        '''
        Get the velocity of the joints
        '''

        if ids is None:
            joints_vel = self._sim.data.qvel
        else:
            joints_vel = self._sim.data.qvel[ids]
        return joints_vel.copy()
    

    def get_joints_ft(self, ids=None):   
        '''
        Get the force/torque of the joints
        '''

        if ids is None:
            joints_ft = self._sim.data.qfrc_actuator
        else:
            joints_ft = self._sim.data.qfrc_actuator[ids]
        return joints_ft.copy()
 
    def get_state(self):
        ''' 
        Returns the states from the given list (states_list) 
        '''
        state = [] 
        sim_states = self._sim.get_state() 
        for sname, sdata in self._states_specs.items():

            sid = sdata["id"]   
            stype = sdata["type"]     
            svar = sdata["var"]     

            attr_get_value = getattr(self._sim.data,f"get_{stype}_{svar}") 
            sval = attr_get_value(sid)  
            if stype in ["body","site"]: 
                sval = list(np.array(sval)-np.array(self.fixed_frame[:3]))  # convert wrt fixed frame
                # TODO rotazione!!!!  

            sval = list(np.array(sval).flatten())   

            state.append(sval)
        state = [item for sublist in state for item in sublist]  
        return state.copy()
   
    def env_fixed_frame(self, name=None):
        ''' 
        Returns the fixed reference frame as a list [x,y,z,qw,qx,qy,qz]
        @param name: name of the body to be used as reference frame
        '''
        frame = []
        if name is None:
            frame = [0, 0, 0, 1, 0, 0, 0]
        else:
            pos = self._sim.data.get_body_xpos(name)
            quat = self._sim.data.get_body_xquat(name)
            for p in pos:
                frame.append(p)
            for q in quat:
                frame.append(q)
        self.fixed_frame = frame
        return frame
