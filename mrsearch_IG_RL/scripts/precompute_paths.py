from mrsearch_IG_RL.external import *
import mrsearch_IG_RL
from mrsearch_IG_RL import CFG_DIR

if __name__ == "__main__":
    env = mrsearch_IG_RL.envs.base_env(True,False,False,CFG_DIR+"/base.yaml")
    env.reset()

    paths = {}
    for r in range(env.entropy.size()[0]):
        for c in range(env.entropy.size()[1]):
            if env._is_valid(r,c):
                for tr in range(env.entropy.size()[0]):
                    for tc in range(env.entropy.size()[1]):
                        if env._is_valid(tr,tc):
                            paths[(r,c)] = env._get_path_len((r,c),(tr,tc))
    with open("assets/paths.json", "w") as f:
        json.dump(paths , f) 