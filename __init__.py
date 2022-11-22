from gym.envs.registration import register

register(
    # unique identifier for the env `name-version`
    id="rl_search-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="mrsearch-IG-RL.envs.base_env:base_env"
)