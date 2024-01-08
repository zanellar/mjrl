
import os   
import json
from stable_baselines3.common.callbacks import BaseCallback

############################################################################################

class BaseLogCallback(BaseCallback):
    def __init__(self, prefix="log"):
        super().__init__()   
        self.prefix = prefix

    def set(self, folder_path, file_name, num_rollouts_episode=1):
        '''
        Set the log file path and the number of rollouts per episode
        folder_path: path to the folder where the log file will be saved
        file_name: name of the log file
        num_rollouts_episode: number of rollouts per episode
        '''
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.file_path = os.path.join(self.folder_path, f"{self.prefix}_{file_name}.json")
        self.num_rollouts_episode = num_rollouts_episode 
        self.episodes = 0 

    def _on_training_start(self) -> None:  
        self.rollouts = 0   
        return True
    
    def _is_episode_end(self):
        return self.rollouts % self.num_rollouts_episode == 0

    def _on_rollout_start(self) -> None:
        self.rollouts_return = 0 
        if self._is_episode_end():
            # reset start episode data
            pass 
        return True

    def _on_step(self) -> bool:
        # do something in the step
        return True
    
    def _on_rollout_end(self) -> None:  
        self.rollouts += 1 
        if self._is_episode_end():
            # process end episode data
            pass 
            self.episodes += 1 
        return True

    def _on_training_end(self) -> bool: 
        return True
 
         
############################################################################################

class SaveTrainingLogsCallback(BaseLogCallback):
    def __init__(self, save_all=False):
        super().__init__() 
        self.save_all = save_all 
        self.episodes_return = 0
        self.rollouts_return = 0 

    def _on_training_start(self) -> None:
        self.episodes_data = dict(steps=[], obs=[], actions=[], rewards=[])
        self.training_data = dict(episodes_data=[], episodes_return=[], rollouts_return=[])
        self.rollouts = 0   
        return True

    def _on_rollout_start(self) -> None: 
        return True

    def _on_step(self) -> bool:
        sample = self.training_env.env_method("get_sample") 
        obs, action, reward, terminated, truncated, info = sample[0]
        
        if terminated or truncated:
            return True

        if self.save_all:
            self.episodes_data["steps"].append(self.num_timesteps)
            self.episodes_data["obs"].append(obs.tolist())
            self.episodes_data["actions"].append(action.tolist())
            self.episodes_data["rewards"].append(reward) 
        self.rollouts_return += reward
        return True
    
    def _on_rollout_end(self) -> None:  
        self.rollouts += 1
        self.episodes_return += self.rollouts_return 
        if self._is_episode_end():
            self.training_data["episodes_return"].append(self.episodes_return) 
            if self.save_all:
                self.training_data["episodes_data"].append(self.episodes_data)  
            self.episodes += 1 
            self.episodes_return = 0  
        self.rollouts_return = 0  
        return True

    def _on_training_end(self) -> bool: 
        print("@@@@@@@@@@@ TRAINING END @@@@@@@@@@@@@@") 
        self.training_data["num_tot_steps"]= self.num_timesteps
        with open(self.file_path, 'w') as f:
            json.dump(self.training_data, f)
  