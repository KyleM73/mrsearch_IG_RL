import mrsearch_IG_RL.cfg as cfg
import mrsearch_IG_RL.external as ext

PATH_DIR = ext.os.path.dirname(ext.os.path.realpath(__file__))
CFG_DIR = PATH_DIR+"/cfg"
LOG_PATH = PATH_DIR+"/log/logs"

import mrsearch_IG_RL.envs

ext.gym.envs.registration.register(
    # unique identifier for the env `name-version`
    id="rl_search-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mrsearch_IG_RL.envs:base_env",
    kwargs={
    'training' : True,
    'record' : False,
    'boosted' : True,
    'cfg' : CFG_DIR+cfg.base_cfg,
    'plot' : False
    },
)

ext.gym.envs.registration.register(
    # unique identifier for the env `name-version`
    id="rl_search-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mrsearch_IG_RL.envs:base_env",
    kwargs={
    'training' : True,
    'record' : False,
    'boosted' : False,
    'cfg': CFG_DIR+cfg.base_cfg,
    'plot' : False
    },
)