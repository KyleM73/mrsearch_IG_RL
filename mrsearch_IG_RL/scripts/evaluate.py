from mrsearch_IG_RL.external import *
import mrsearch_IG_RL
from mrsearch_IG_RL import CFG_DIR
from mrsearch_IG_RL.models import EntropyCnn

parser = argparse.ArgumentParser(prog="mrsearch_IG_RL",description="Evaluate trained model")
parser.add_argument("filename",type=str)
parser.add_argument("-n","--no_record",help="do not record simulation results",action="store_true")
parser.add_argument("-s","--simulate",help="show simulation",action="store_true")
parser.add_argument("-b","--boosted",help="run simulation with boosting",action="store_true")
parser.add_argument("-z","--zero_action",help="run with no policy",action="store_true")
parser.add_argument("-p","--plot",help="plot action outputs",action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    env = mrsearch_IG_RL.envs.base_env(not args.simulate,not args.no_record,args.boosted,CFG_DIR+"/base.yaml",args.plot)
    if not args.zero_action:
        model_path = args.filename
        model = PPO.load(model_path,env,custom_objects={"CustomModel": EntropyCnn})
    ob = env.reset()

    for _ in range(env.max_steps+1):
        if not args.zero_action:
            action, states = model.predict(ob)
        else:
            action = np.array([0,0,0])
        ob, reward, done, info = env.step(action)
        if done:
            break
        if args.simulate:
            time.sleep(env.dt)