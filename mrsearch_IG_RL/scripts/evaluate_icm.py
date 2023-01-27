from mrsearch_IG_RL.external import *
import mrsearch_IG_RL
from mrsearch_IG_RL import CFG_DIR
from mrsearch_IG_RL.models import ActorCriticICM,ICM_PPO

parser = argparse.ArgumentParser(prog="mrsearch_IG_RL",description="Evaluate trained model")
parser.add_argument("filename",type=str)
parser.add_argument("-n","--no_record",help="do not record simulation results",action="store_true")
parser.add_argument("-s","--simulate",help="show simulation",action="store_true")
parser.add_argument("-l","--no_load",help="do not load trained policy",action="store_true")
parser.add_argument("-p","--plot",help="plot action outputs",action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    env = mrsearch_IG_RL.envs.icm_env_fstack(not args.simulate,not args.no_record,CFG_DIR+"/icm.yaml",args.plot)
    if not args.no_load:
        model_path = args.filename
        model = ICM_PPO.load(model_path,env,custom_objects={"CustomModel": ActorCriticICM})
    else:
        model = ActorCriticICM(env.observation_space,env.action_space)

    ob = env.reset()

    while True:
        ob = torch.from_numpy(ob)
        if not args.no_load:
            action, states, probs, intrinsic_rewards = model.policy(ob)
        else:
            action, states, probs, intrinsic_rewards = model(ob)
        #    action = np.array([0.])
        action = action.detach().numpy()
        intrinsic_rewards = torch.tensor([intrinsic_rewards[0],intrinsic_rewards[1]]).detach()
        ob, reward, done, info = env.step([action, intrinsic_rewards])
        if done:
            print("Done condition hit.")
            break
        if args.simulate:
            time.sleep(env.dt)