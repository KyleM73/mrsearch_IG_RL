import time
import datetime
import torch
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import mrsearch_IG_RL
from mrsearch_IG_RL import LOG_PATH
from mrsearch_IG_RL.models import EntropyCnn

def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

if __name__ == "__main__":

    ## env params
    env_id = "rl_search-v0"
    num_envs = 32

    ## make parallel environments
    env = SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)],start_method='fork')

    ## training params
    train_steps = 1_000_000
    device = torch.device('mps')
    n_steps = 64
    buffer_size = n_steps * num_envs
    batch_size = 2048
    assert buffer_size % batch_size == 0

    ## logging params
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    tb_log = LOG_PATH+"/{}".format(dt)

    ## train model
    policy_kwargs = dict(features_extractor_class=EntropyCnn,net_arch=[64,64])
    model = PPO(
        "CnnPolicy",env,policy_kwargs=policy_kwargs,
        n_steps=n_steps,batch_size=batch_size,
        verbose=1,tensorboard_log=tb_log,use_sde=True,
        device=device)
    start_time = time.time()
    model.learn(total_timesteps=train_steps,tb_log_name=dt)
    end_time = time.time()
    print("Model train time: "+str(datetime.timedelta(seconds=end_time-start_time)))

    ## save  model
    save_path = '{}/{}'.format(tb_log,"model")
    model.save(save_path)


