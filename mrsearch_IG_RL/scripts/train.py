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
    env_id_boosted = "rl_search-v0"
    env_id = "rl_search-v1"
    num_envs = 32

    ## training params
    total_train_steps = 1_048_576 # train_steps % batch_size == 0
    boosting_percent = 1
    device = torch.device('mps')
    n_steps = 64
    buffer_size = n_steps * num_envs
    batch_size = 2048
    assert buffer_size % batch_size == 0
    policy_kwargs = dict(activation_fn=nn.Tanh,features_extractor_class=EntropyCnn,features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128,64,32],normalize_images=False)

    ## logging params
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    tb_log = LOG_PATH+"/{}".format(dt)
    save_path_boosted = '{}/{}'.format(tb_log,"model_boosted")
    save_path = '{}/{}'.format(tb_log,"model")

    ## ========== train model ==========
    ## phase 1 : boosting
    env = VecMonitor(SubprocVecEnv([makeEnvs(env_id_boosted) for i in range(num_envs)],start_method='forkserver')) #forkserver
    model = PPO(
        "CnnPolicy",env,policy_kwargs=policy_kwargs,
        n_steps=n_steps,batch_size=batch_size,
        verbose=1,tensorboard_log=tb_log,use_sde=True,
        device=device)
    start_time1 = time.time()
    model.learn(total_timesteps=int(boosting_percent*total_train_steps),tb_log_name=dt+"_phase_1",progress_bar=True)
    model.save(save_path_boosted)
    del model
    end_time1 = time.time()
    print("Phase 1 train time: "+str(datetime.timedelta(seconds=end_time1-start_time1)))
    """
    ## phase 2 : exploitation
    env = VecMonitor(SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)],start_method='forkserver')) #forkserver
    model = PPO.load(save_path_boosted,env,policy_kwargs=policy_kwargs,device=device)
    start_time2 = time.time()
    model.learn(total_timesteps=int((1-boosting_percent)*total_train_steps),tb_log_name=dt+"_phase_2",progress_bar=True)
    model.save(save_path)
    del model
    end_time2 = time.time()
    print("Phase 2 train time: "+str(datetime.timedelta(seconds=end_time2-start_time2)))
    print()
    print("Total model train time: "+str(datetime.timedelta(seconds=(end_time2-start_time2)+(end_time1-start_time1))))
    """