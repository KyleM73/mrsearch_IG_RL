import time
import argparse
parser = argparse.ArgumentParser(prog="mrsearch_IG_RL",description="Evaluate trained model")
parser.add_argument("filename",type=str)
args = parser.parse_args()

from stable_baselines3 import PPO

import mrsearch_IG_RL
from mrsearch_IG_RL import CFG_DIR

if __name__ == "__main__":
    env = mrsearch_IG_RL.envs.base_env(True,CFG_DIR+"/base.yaml")
    model_path = args.filename
    model = PPO.load(model_path)
    ob = env.reset()

    for _ in range(env.max_steps+1):
        action, states = model.predict(ob)
        ob, reward, done, info = env.step(action)
        if done:
            break
        #time.sleep(0.01)