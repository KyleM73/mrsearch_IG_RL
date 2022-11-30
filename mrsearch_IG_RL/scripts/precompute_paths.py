from mrsearch_IG_RL.external import *
import mrsearch_IG_RL.cfg as cfg
import mrsearch_IG_RL
from mrsearch_IG_RL import CFG_DIR

if __name__ == "__main__":
    env = mrsearch_IG_RL.envs.base_env(True,False,True,CFG_DIR+cfg.base_cfg)
    dists = env.dists
    np.save("assets/distance_data.npy",dists)

