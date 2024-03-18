### Install MuJoCo & mujoco-py ##
1. Install Anaconda. Download it from this [link](https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh).
```
cd Downloads/
sudo chmod +x Anaconda3-2021.11-Linux-x86_64.sh
./Anaconda3-2021.11-Linux-x86_64.sh
```

2. Install git.
```
sudo apt install git
```

3. Install the Mujoco library.

    * Download the Mujoco library from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz).
    * Create a hidden folder :
    ```
    mkdir /home/<username>/.mujoco
    ```
    * Extract the library to the .mujoco folder.
    ```
    tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
    ```
    * Include these lines in .bashrc file.
    ```
    # Replace user-name with your username
    echo -e 'export LD_LIBRARY_PATH=/home/<username>/.mujoco/mujoco210/bin 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
    export PATH="$LD_LIBRARY_PATH:$PATH"  >> ~/.bashrc
    ```
    * Source bashrc.
    ```
    source ~/.bashrc
    ```
    * Test that the library is installed.
    ```
    cd ~/.mujoco/mujoco210/bin
    ./simulate ../model/humanoid.xml
    ```

4. Install mujoco-py.
```
conda create --name mjrl python=3.8
conda activate mjrl
sudo apt update
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt-get install libxslt1-dev zlib1g-dev libglew2.2 libglew-dev python3-pip

# Clone mujoco-py.
cd ~/.mujoco
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install "cython<3"
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip3 install -e . --no-cache
```
5. Reboot your machine.
```
sudo reboot
```
6. After reboot, run these commands to install additional packages.
```
conda activate mjrl
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
# If you get an error like: "ln: failed to create symbolic link '/usr/lib/x86_64-linux-gnu/libGL.so': File exists", it's okay to proceed
pip3 install -U 'mujoco-py<2.2,>=2.1'
```
7. Check if mujoco-py is properly installed.
```
cd ~/.mujoco/mujoco-py/examples
python3 setting_state.py
```

### Thrumbleshootting ###
1. `ImportError: /home/<username>/miniconda3/envs/mjrl/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)`
```
rm /home/<username>/miniconda3/envs/mjrl/bin/../lib/libstdc++.so.6
``` 

## Instal MjRL ##


```
conda env update --name mjrl --file environment.yml --prune
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
* creare dei file di debug per environments generici
* vectorized environment
* wandb sync gym videos
* use TQC
* https://github.com/Bargez908/cercagrigliona/blob/main/cercagrigliona/envs/panda_torques_reach_site_random.py
* https://github.com/Bargez908/cercagrigliona/blob/main/params/runs/panda_torques_tqc.yaml 
* https://github.com/Bargez908/cercagrigliona/blob/main/cercagrigliona/utils/train.py
