repo for multi-robot RL search with information gains

## Setup
    / % conda create -n search python=3.9.7
    / % conda activate search
    / % git clone (...)
    (search) / % cd mrsearch_IG_RL
    (search) mrsearch_IG_RL % pip install -e .
    (search) mrsearch_IG_RL % mkdir -p mrsearch_IG_RL/log/videos && mkdir -p mrsearch_IG_RL/log/logs
## Train
go to ``mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml`` and set record: False

go to ``mrsearch_IG_RL/mrsearch_IG_RL/scripts/train.py`` and set the appopriate device on line 31 (options are ``'cpu'``,``'cuda'``,``'mps'``)

    (search) mrsearch_IG_RL % cd mrsearch_IG_RL
    (search) mrsearch_IG_RL/mrsearch_IG_RL % python scripts/train.py
to view tensorboard logs during traing, launch tensorboard in a separate terminal with:

    (search) mrsearch_IG_RL/mrsearch_IG_RL % tensorboard --logdir log/logs/{date}
### Train Tuning
there are three params in ``mrsearch_IG_RL/mrsearch_IG_RL/scripts/train.py`` tht control training performance:

    num_envs: the number of parallel environments
    n_steps: the number of steps each environment takes before updating the model weights
    batch_size: the total buffer size of the (s,a,r) tuples passed to train the model
with the constraint that ``buffer_size=num_envs*n_steps=batch_size``. with ``cpu`` training, training speed is set by cpu cache. the default params are:

    num_envs = 1
    n_steps = 64
    batch_size = 64
with ``gpu`` and ``mps`` the limiting factor is vRAM and RAM, respectively. the goal is to increase ``batch_size`` until hitting a memory error and then back off slightly. the limit on my macbookpro training on ``mps`` is:

    num_envs = 32
    n_steps = 64
    batch_size = 2048
training 1 million timesteps on my m1 macbookpro takes about an hour and a half.

on more powerful gpus, start by increasing ``num_envs`` until hitting max files open error. reduce ``num_envs`` by a factor of 2 after hitting the error and increase ``n_steps`` until hitting an out of memory error. reduce ``n_steps`` byt a factor of 2 once the error is hit. always set ``batch_size`` accordingly as ``batch_size=num_envs*n_steps``.

the only other param that meaningfully affects memory performance is ``lidar/density`` in ``mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml`` which controls the number of lidar scans.
## Evaluate
go to ``mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml`` and set record: True

to display simulation during evaluation set:

    env = mrsearch_IG_RL.envs.base_env(False,CFG_DIR+"/base.yaml"
on line 13 and uncomment the last line of the for loop:

    time.sleep(0.01)
videos are saved to ``mrsearch_IG_RL/mrsearch_IG_RL/log/videos/{date}.mp4``

    (search) mrsearch_IG_RL/mrsearch_IG_RL % python scripts/evaluate.py /log/logs/{date}/model.zip
## Tuning
see config params at ``mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml``

and reward function in ``base_env._get_rew()`` in ``mrsearch_IG_RL/mrsearch_IG_RL/envs/base.py``
## Model Architecture
CNN architecture can be found in ``mrsearch_IG_RL/mrsearch_IG_RL/models/entropycnn.py``

inputs are of size ``[1,201,201]`` and bounded between ``[-1,1]``

CNN output size and MLP policy architecture can be set in ``policy_kwargs`` on line 42 of ``mrsearch_IG_RL/mrsearch_IG_RL/scripts/train.py``
## TODO
- Does the current reward function make sense?
- How can the terms be adjusted to give the desired behavior?
- How can we generalize this policy to multiple agents?