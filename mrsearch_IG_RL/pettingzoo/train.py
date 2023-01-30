import numpy as np
import torch

import env
import policy
import ppo

## Device 
if torch.cuda.is_available():
    device = torch.device("cuda")
#elif torch.backends.mps.is_available(): #MPS is buggy
#    device = torch.device("mps")
else:
    device = torch.device("cpu")

## Create Env
cfg_file = "cfg.yaml"
output_dir = "train_1_30_23"
env = env.icm_env(headless=True,record=False,cfg=cfg_file,output_dir=output_dir)

## PPO Params
total_steps = 1_000_000
batch_size = 100
epochs = 5 #num times to repeat each episode
obs_size = 3
stack_size = env.n_frames
frame_size = (env.obs_w, env.obs_w)
n_steps = env.max_steps
n_agents = len(env.possible_agents)
beta = env.beta
loss_coef = env.lmbda
learning_rate = 0.001

net = policy.Network().to(device)

opt = torch.optim.Adam(net.parameters(), lr=learning_rate, eps=1e-5)

PPO(env=env, policy=net, optimizer=opt, max_steps=total_steps, epochs=epochs, 
    batch_size=batch_size, frame_size=frame_size, stack_size, num_agents=n_agents, 
    n_steps=n_steps, vec_obs_size=obs_size, loss_coef=loss_coef, device=device)



