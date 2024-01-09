import numpy as np
import os 
import torch    
import wandb 

from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback,CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise, ActionNoise
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.logger import configure 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder 



from mjrl.utils.paths import ENVS_PATH, WEIGHTS_PATH, PLOTS_PATH, PARAMS_PATH, LOGS_PATH
from mjrl.scripts.cbs import SaveTrainingLogsCallback
from mjrl.utils.argsutils import Dict2Args
  
class Trainer():

    def __init__(self, config, env, enveval=None, sweep=False):

        self.params = Dict2Args(config["parameters"])
        self.settings = Dict2Args(config["settings"])
        print("parameters: ",self.params)
        print("settings: ", self.settings)

        self.sweep = sweep
        self.env = env 
        if enveval is None:
            self.enveval = env
        else:
            self.enveval = enveval
 
        # torch.manual_seed(config.seed)  
        # np.random.seed(config.seed)
         
    def run(self):    
 
        if self.sweep:
            wandb.init(
                sync_tensorboard = True,
                # monitor_gym = True,  
                # save_code = False,
                # reinit = True
                )
            self.params = wandb.config  
    
        # if self.settings.env_eval_rendering: # BUG
        #     def make_env(): 
        #         return Monitor(self.env)    
        #     self.env = DummyVecEnv([make_env])  
        #     self.env = VecVideoRecorder(self.env, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)  # record videos

        ###################### PATHS ########################
        run_name = wandb.run.name

        try: 
            weights_path = os.path.join(self.settings.weights_path, self.settings.env_id, run_name) 
        except:
            weights_path = os.path.join(WEIGHTS_PATH, self.settings.env_id, run_name)
        
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

        best_model_folder_path = os.path.join(weights_path, run_name,"best_model")
        if not os.path.exists(best_model_folder_path):
            os.makedirs(best_model_folder_path)
 
        # # normalized_env_save_path = os.path.join(weights_path, "normalized_env.pickle") 
        # cblogs_path = os.path.join(weights_path, "cblogs")  
  
        ###################### AGENT ######################## 

        if self.params.exploration_noise is None: 
            action_noise = None 
        else: 
            if self.params.exploration_noise=="walk":
                noise_func = OrnsteinUhlenbeckActionNoise 
            elif self.params.exploration_noise=="gaussian":
                noise_func = NormalActionNoise   
            else:
                noise_func =  ActionNoise
            action_noise = noise_func(
                mean = np.zeros(self.env.action_space.shape), 
                sigma = self.params.exploration_sigma*np.ones(self.env.action_space.shape)  
            )  
            
        if self.settings.algorithm.lower()=="sac":
            agent = SAC(
                policy = 'MlpPolicy',
                env = self.env,  
                buffer_size = self.params.buffer_size,
                batch_size = self.params.batch_size,
                learning_rate = self.params.learning_rate,
                gamma = self.params.gamma,
                tau = self.params.tau,
                action_noise = action_noise,
                learning_starts = self.params.learning_starts,
                train_freq = (int(self.params.train_freq.split("_")[0]), self.params.train_freq.split("_")[1]),
                gradient_steps = self.params.gradient_steps,
                seed = self.params.seed,
                #
                use_sde_at_warmup = self.params.use_sde_at_warmup,
                use_sde = self.params.use_sde,
                sde_sample_freq = self.params.sde_sample_freq,
                target_update_interval = self.params.target_update_interval,
                ent_coef = self.params.ent_coef,
                policy_kwargs = self.params.policy_kwargs,
                #
                verbose = 1 
            ) 
  
        new_logger = configure(LOGS_PATH, ["stdout", "csv", "tensorboard"])
        agent.set_logger(new_logger)

        ###################### CALLBACKS ######################## 

        callbackslist = [] 

        # ###### SAVE TRAINING LOGS
        # save_training_logs_cb = SaveTrainingLogsCallback(save_all = self.settings.save_all_training_logs)  
        # save_training_logs_cb.set(
        #     folder_path = cblogs_path,
        #     file_name = name,
        #     num_rollouts_episode = int(self.settings.expl_episode_horizon/config["train_freq"][0])
        # ) 

        # callbackslist.append(save_training_logs_cb)
 
        callbackslist.append(
            WandbCallback( 
                verbose = 2
            )
        )

        ###### EARLY STOPPING 
        if self.settings.early_stopping: 
            early_stop_callback = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals = self.settings.max_no_improvement_evals, 
                    min_evals = self.settings.min_evals, 
                    verbose = 1
            ) 
        else:
            early_stop_callback = None

        ###### EVALUATION 
        callbackslist.append(
            EvalCallback(
                self.enveval, 
                best_model_save_path = best_model_folder_path, 
                eval_freq = self.settings.eval_model_freq,
                n_eval_episodes = self.settings.num_eval_episodes, 
                deterministic = True, 
                render = False,
                callback_after_eval = early_stop_callback 
            ) 
        )  # BUG: need to comment "sync_envs_normalization" function in EvalCallback._on_step() method 
        
        ###### CUSTOM CALLBACKS
        # if len(self.settings.callbacks)>0:
        #     for external_callback in self.settings.callbacks:
        #         external_callback.set(
        #             folder_path = cblogs_path,
        #             file_name = name,
        #             num_rollouts_episode = int(self.settings.expl_episode_horizon/config["train_freq"][0])
        #         ) 
        #         callbackslist.append(external_callback)

        callbacks = CallbackList(callbackslist)
    

        ###################### LEARNING ######################## 

        self.env.reset()    
        agent.learn(
            total_timesteps = self.settings.training_horizon, 
            log_interval = 1,  
            progress_bar = True, 
            callback = callbacks
        )     

        # if self.settings.normalize_env is not None: 
        #     if not normalized_env_save_path:
        #         os.makedirs(normalized_env_save_path)
        #     enveval.save(normalized_env_save_path)



        ##############################################  

        # Evaluate the best model  
        # best_model_file_path = os.path.join(best_model_folder_path,"best_model.zip")
        # best_model = agent_class.load(best_model_file_path)  
        # mean_reward, std_reward = evaluate_policy(
        #                             best_model,  
        #                             enveval, 
        #                             n_eval_episodes=self.settings.num_eval_episodes_best_model, 
        #                             render=False, 
        #                             deterministic=True
        #                         )   

        # save_training_data.add(name, config, res=[mean_reward, std_reward])            
