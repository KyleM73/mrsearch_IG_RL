from .envs.base_env import base_env
from .cfg import *

from gym.envs.registration import register

register(
    # unique identifier for the env `name-version`
    id="rl_search-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mrsearch_IG_RL.envs.base_env:base_env"
)
