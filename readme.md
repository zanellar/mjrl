 
## Installation ##

Setup virtual environment

```
conda env create -f condaenv.yml
```


Add the following line to .bashrc

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/[...]/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```


```
pip install -e .
``` 


## Mujoco ##
## XML
quaternions are in the format [qw,qx,qy,qz]

### Contact Forces bug
Issue: https://github.com/openai/mujoco-py/pull/487
Need to modify 'mujoco_py/mjviewer.py' in the conda env as follow:
https://github.com/openai/mujoco-py/pull/487/commits/ab026c1ff8df54841a549cfd39374b312e8f00dd


## Usage ##

### Create an environment

- create a folder in 'data/envdesc' with the files 'arena.xml'(simulation model description) and a 'specs.json'(state and action specifics)
- create a class in 'mjrlenvs/envrl/' that extends 'EnvGymBase' or 'EnvGymGoalBase' ('from mjrlenvs.scripts.env.envgymbase import *' ) and implement here the observation and the reward computation 


### Run 

create a configuration file like 'some_run.py' in 'mjrlenvs/train/run' then 

'''
python mjrl/run/some_run.py
'''

This can be either a single training or multiple trainings with same parametrs or a grid search. 

The best model, tensorboard output and a .txt file with parameters and mean reward can be found in 'mjrlenvs/data/test' 

## Usage ##

* fix trainer.py and runs with it
* fix panda


## Possible Errors Solved
* Error: 'Import error. Trying to rebuild mujoco_py.' or 'GLIBCXX_3.4.30 not found'
Solution: delete file '/home/riccardo/miniconda3/envs/rl0/lib/libstdc++.so.6'.  https://stackoverflow.com/questions/72205522/glibcxx-3-4-29-not-found

## TODOS
 
* testare salvataggio e ripristino pesi
* testare training pendolo 
* trovare un ordine sensato delle cartelle in mjrl
* creare dei file di debug per environments generici
* vectorized environment
* wandb sync gym videos
* 
