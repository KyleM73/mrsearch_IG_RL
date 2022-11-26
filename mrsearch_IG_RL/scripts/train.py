from mrsearch_IG_RL.external import *
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
    env = VecMonitor(SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)],start_method='forkserver')) #forkserver

    ## training params
    train_steps = 1_048_576 # train_steps % batch_size == 0
    device = torch.device('mps')
    n_steps = 64
    buffer_size = n_steps * num_envs
    batch_size = 2048
    assert buffer_size % batch_size == 0

    ## logging params
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    tb_log = LOG_PATH+"/{}".format(dt)

    ## train model
    policy_kwargs = dict(features_extractor_class=EntropyCnn,features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128,64,32],normalize_images=False)
    model = PPO(
        "CnnPolicy",env,policy_kwargs=policy_kwargs,
        n_steps=n_steps,batch_size=batch_size,
        verbose=1,tensorboard_log=tb_log,use_sde=True,
        device=device)
    start_time = time.time()
    model.learn(total_timesteps=train_steps,tb_log_name=dt,progress_bar=True)
    end_time = time.time()
    print("Model train time: "+str(datetime.timedelta(seconds=end_time-start_time)))

    ## save  model
    save_path = '{}/{}'.format(tb_log,"model")
    model.save(save_path)