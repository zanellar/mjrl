
import os
import mjrl

  
_PACKAGE_PATH = mjrl.__path__[0]  

ENVS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "envs")  
WEIGHTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "weights")  
PLOTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "plots")  
PARAMS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "params")  
LOGS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "logs")
RESULTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "results")

