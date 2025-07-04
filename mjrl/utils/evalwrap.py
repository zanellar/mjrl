import wandb 
import numpy as np
import gymnasium as gym 
from gymnasium import spaces

from mjrl.utils.gym import EnvGymBase
from mjrl.utils.argsutils import Dict2Args
 
class EnvEvalWrapper(EnvGymBase):

    def __init__(self, env, vars=[], only_final=[], settings={}, dim_hist_mean_final_values=10, logs=["min", "max", "avg"]):
        '''
        env: the original environment 
        vars: list of vars to log (name of the variable in the env). The variable must be a scalar. 
        only_final: list of boolean whether to log only the final value of the var or not corresponding to the vars list
        settings: dictionary of settings
        dim_hist_mean_final_values: the dimension of the history of the mean final values
        '''
        super(EnvEvalWrapper, self).__init__()
        
        self.settings = Dict2Args(settings)

        self._env = env 
        self.vars = vars
        self.only_final = only_final
        self.logs = logs

        # initialize the the moving average of the mean final value
        self._hist_mean_final_values = {}
        self.dim_hist_mean_final_values = dim_hist_mean_final_values
        for var in self.vars:
            self._hist_mean_final_values[var] = [None]*dim_hist_mean_final_values

        # Initialize the eval values
        self.index = 0
        self.eval_values = {}
        for var in self.vars:
            self.eval_values[var] = []
        self.only_final = self.only_final if len(self.only_final) == len(self.vars) else [True]*len(self.vars)
 
        # Actions and observations spaces
        self.action_space = self._env.action_space 
        self.observation_space = self._env.observation_space
  
    def get_reward(self):  
        return self._env.get_reward()

    def set_goal(self, goal):
        self._env.set_goal(goal)
         
    def render(self, mode=None): 
        self._env.render(mode=mode)

    def close(self):
        self._env.close()

    def get_obs(self):
        self.obs = self._env.get_obs()
    
    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def get_sample(self): 
        return self._env.get_sample()
  
    def reset(self, seed=None): 
        super().reset(seed=seed) 
        
        # reset values 
        self.eval_values = {}
        for var in self.vars:
            self.eval_values[var] = []

        self._final_eval_values = {} 
        for var in self.vars:
            self._final_eval_values[var] = []
 
        return self._env.reset(seed=seed)
    
    def step(self, action):   
        '''
        Step the original environment with the given action and log the values of the vars to wandb
        '''
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        for i, var in enumerate(self.vars):
            if not self.only_final[i]: 
                self.eval_values[var].append(getattr(self._env, var))
        
        if terminated or truncated:

            self.index += 1
            # reset when wandb run is finished
            # TODO

            for i, var in enumerate(self.vars):  
                value = getattr(self._env, var)
                print(f"eval/{var}: {value}")
                self._final_eval_values[var].append(value)

                if not self.only_final[i]:  
                    data_plot = [[_x, _y] for (_x, _y) in zip(range(len(self.eval_values[var])), self.eval_values[var])]
                    table = wandb.Table(data=data_plot, columns=["steps", f"{var}_{self.index}"])
                    line_plot = wandb.plot.line(table, x="steps", y=f"{var}_{self.index}", title=f"{var}_{self.index}") 
                    wandb.log({f"evalall/{var}_{self.index}": line_plot})

            if self.index % self.settings.num_eval_episodes == 0:
                for var in self.vars:
                    avg_final_value = np.mean(self._final_eval_values[var]) 
                    min_final_value = np.min(self._final_eval_values[var])
                    max_final_value = np.max(self._final_eval_values[var])
                    if "min" in self.logs:
                        wandb.log({f"eval/min_final_{var}": min_final_value})
                        print(f"\n ----- eval/min_final_{var}: {min_final_value} ----- ")
                    if "max" in self.logs:
                        wandb.log({f"eval/max_final_{var}": max_final_value})
                        print(f"\n ----- eval/max_final_{var}: {max_final_value} ----- ")
                    if "avg" in self.logs: 
                        wandb.log({f"eval/avg_final_{var}": avg_final_value}) 
                        print(f"\n ----- eval/avg_final_{var}: {avg_final_value} ----- ") 

                    self._hist_mean_final_values[var].pop(0)
                    self._hist_mean_final_values[var].append(avg_final_value)
                    if None not in self._hist_mean_final_values[var]: 
                        mean_hist_avg_final_value = np.mean(self._hist_mean_final_values[var])
                        wandb.log({f"eval/mean{self.dim_hist_mean_final_values}_avg_final_{var}": mean_hist_avg_final_value}) 


        return obs, reward, terminated, truncated, info