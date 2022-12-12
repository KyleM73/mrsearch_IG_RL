from mrsearch_IG_RL.external import *
import mrsearch_IG_RL
from mrsearch_IG_RL.models import ActorCriticICM,IdentityExtractor,ICM_PPO
from mrsearch_IG_RL.envs import ICM_DummyVecEnv, ICM_Monitor

def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

if __name__ == "__main__":

    ## env params
    env_id = "icm_search-v1" ##testing env
    num_envs = 1

    ## training params
    total_train_steps = 1000 # train_steps % batch_size == 0
    device = torch.device('cpu')
    n_steps = 100
    buffer_size = n_steps * num_envs
    batch_size = 100
    assert buffer_size % batch_size == 0
    policy_kwargs = dict(features_extractor_class=IdentityExtractor,normalize_images=False)

    ## ========== train model ==========
    env = ICM_DummyVecEnv([makeEnvs(env_id) for i in range(1)])
    ob0 = env.reset()
    x0 = torch.from_numpy(ob0)

    model = ICM_PPO(
        ActorCriticICM,env,policy_kwargs=policy_kwargs,
        n_steps=n_steps,batch_size=batch_size,verbose=1,
        use_sde=True,device=device)
    
    print("init outputs")
    action0, states0, probs0, intrinsic_rewards0 = model.policy(x0)
    print(action0)
    print(intrinsic_rewards0)

    print("Training start:")
    start_time1 = time.time()
    model.learn(total_timesteps=total_train_steps)
    end_time1 = time.time()
    print("Train time: "+str(datetime.timedelta(seconds=end_time1-start_time1)))
    print()

    print("after training outputs")
    print("eval 1")
    action1, states1, probs1, intrinsic_rewards1 = model.policy(x0)
    print(action1)
    print(intrinsic_rewards1)

    #check no random output
    print("eval 2")
    action2, states2, probs2, intrinsic_rewards2 = model.policy(x0)
    print(action2)
    print(intrinsic_rewards2)







