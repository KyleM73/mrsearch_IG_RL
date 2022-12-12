from mrsearch_IG_RL.external import *
from mrsearch_IG_RL import LOG_PATH
from mrsearch_IG_RL.models import ActorCriticICM,IdentityExtractor,ICM_PPO
from mrsearch_IG_RL.envs import ICM_SubprocVecEnv, ICM_Monitor

def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

if __name__ == "__main__":

    ## env params
    env_id = "icm_search_fstack-v0"
    num_envs = 2

    ## training params
    total_train_steps = 1_000_000 # train_steps % batch_size == 0
    device = torch.device('mps')
    n_steps = 64
    buffer_size = n_steps * num_envs
    batch_size = 128
    assert buffer_size % batch_size == 0
    policy_kwargs = dict(features_extractor_class=IdentityExtractor,normalize_images=False)

    ## logging params
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    tb_log = LOG_PATH+"/{}".format(dt)
    save_path = '{}/{}'.format(tb_log,"model")

    ## ========== train model ==========
    env = ICM_Monitor(ICM_SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)],start_method='forkserver')) #forkserver
    model = ICM_PPO(
        ActorCriticICM,env,policy_kwargs=policy_kwargs,
        n_steps=n_steps,batch_size=batch_size,verbose=1,
        tensorboard_log=tb_log,use_sde=True,device=device)
    start_time1 = time.time()
    model.learn(total_timesteps=total_train_steps,tb_log_name=dt,progress_bar=True)
    model.save(save_path)
    del model
    end_time1 = time.time()
    print("Train time: "+str(datetime.timedelta(seconds=end_time1-start_time1)))