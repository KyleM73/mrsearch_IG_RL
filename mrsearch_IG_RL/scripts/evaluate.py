from mrsearch_IG_RL.external import *
import mrsearch_IG_RL
from mrsearch_IG_RL import CFG_DIR

parser = argparse.ArgumentParser(prog="mrsearch_IG_RL",description="Evaluate trained model")
parser.add_argument("filename",type=str)
parser.add_argument("-n","--no_record",help="do not record simulation results",action="store_true")
parser.add_argument("-p","--play",help="show simulation",action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    env = mrsearch_IG_RL.envs.base_env(not args.play,not args.no_record,CFG_DIR+"/base.yaml")
    model_path = args.filename
    model = PPO.load(model_path)
    ob = env.reset()

    for _ in range(env.max_steps+1):
        action, states = model.predict(ob)
        ob, reward, done, info = env.step(action)
        if done:
            break
        if args.play:
            time.sleep(env.dt)