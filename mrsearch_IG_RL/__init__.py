import os
PATH_DIR = os.path.dirname(os.path.realpath(__file__))
CFG_DIR = PATH_DIR+"/cfg"
LOG_PATH = PATH_DIR+"/log/logs"

from mrsearch_IG_RL.envs import *
from mrsearch_IG_RL.cfg import base_cfg

from gym.envs.registration import register

register(
    # unique identifier for the env `name-version`
    id="rl_search-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mrsearch_IG_RL.envs:base_env",
    kwargs={
    'training' : True,
    'record' : False,
    'cfg': CFG_DIR+base_cfg
    },
)