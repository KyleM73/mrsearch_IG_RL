import os
import time
import datetime
import numpy as np

import pybullet as p

import gym
from gym.wrappers import Monitor as gymMonitor

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import mrsearch_IG_RL

def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

if __name__ == "__main__":

    env_id = "rl_search-v0"
    num_envs = 1
    train_steps = 1_000_000

    ## make parallel environments
    env = SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)],start_method='fork')

    ## train model
    policy_kwargs = {'net_arch':[128]}
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs,verbose=1,use_sde=True)
    start_time = time.time()
    model.learn(total_timesteps=train_steps)
    end_time = time.time()
    print("Model train time: "+str(datetime.timedelta(seconds=end_time-start_time)))


