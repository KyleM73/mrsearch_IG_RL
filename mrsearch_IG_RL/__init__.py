import os
PATH_DIR = os.path.dirname(os.path.realpath(__file__))

from mrsearch_IG_RL.envs import *
from mrsearch_IG_RL.cfg import *

from gym.envs.registration import register

register(
    # unique identifier for the env `name-version`
    id="rl_search-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mrsearch_IG_RL.envs.base_env:base_env",
    kwargs={
    'training':True,
    'cfg': PATH_DIR+"/cfg/base_env.yaml"
    },
)
