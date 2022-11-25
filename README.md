repo for multi-robot RL search with information gains

## Setup
    / % conda create -n search python=3.9.7
    / % conda activate search
    / % git clone (...)
    (search) / % cd mrsearch_IG_RL
    (search) mrsearch_IG_RL % pip install -e .
## Train
go to mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml and set record: False
go to mrsearch_IG_RL/mrsearch_IG_RL/scripts/train.py and set the appopriate device on line 31 (options are ``'cpu'``,``'cuda'``,``'mps'``)
    (search) mrsearch_IG_RL % cd mrsearch_IG_RL
    (search) mrsearch_IG_RL/mrsearch_IG_RL % python scripts/train.py
to view tensorboard logs during traing, launch tensorboard in a separate terminal with:
    (search) mrsearch_IG_RL/mrsearch_IG_RL % tensorboard --logdir log/logs/{date}
training 1 million timesteps on my m1 macbookpro takes just over an hour
## Evaluate
go to mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml and set record: True
to display simulation during evaluation set:
    env = mrsearch_IG_RL.envs.base_env(False,CFG_DIR+"/base.yaml"
on line 13 and uncomment the last line of the for loop:
    time.sleep(0.01)
    (search) mrsearch_IG_RL/mrsearch_IG_RL % python scripts/evaluate.py /log/logs/{date}/model.zip
videos are saved to mrsearch_IG_RL/mrsearch_IG_RL/log/videos/{date}.mp4
## Tuning
see config params at mrsearch_IG_RL/mrsearch_IG_RL/cfg/base.yaml
and reward function in ``base_env._get_rew()`` in mrsearch_IG_RL/mrsearch_IG_RL/envs/base.py
## Model Architecture
CNN architecture can be found in mrsearch_IG_RL/mrsearch_IG_RL/models/entropycnn.py
inputs are of size ``[1,201,201]`` and bounded between ``[-1,1]``
CNN output size and MLP policy architecture can be set in ``policy_kwargs`` on line 42 of mrsearch_IG_RL/mrsearch_IG_RL/scripts/train.py
## TODO
- Does the current reward function make sense?
- How can the terms be adjusted to give the desired behavior?
- How can we generalize this policy to multiple agents?
