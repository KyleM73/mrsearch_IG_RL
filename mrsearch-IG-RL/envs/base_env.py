import os
import yaml
import numpy as np

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

class base_env(env):
    def __init__(self,training=True,cfg=None):
        if ifinstance(cfg,str):
            assert os.path.exists(cfg), "Configuration file specified does not exist"
            cfg = yaml.load(cfg)
        else:
            raise AssertionError("No configuration file specified")

        self.cfg = cfg

        if training:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.training = training
        
        ## simulation params
        self.horizon = self.cfg["horizon"]
        self.dt = self.cfg["dt"]

        ## environment params
        self.resolution = self.cfg["environment"]["resolution"]
        self.h = int(self.cfg["environment"]["height"] * self.resolution)
        self.w = int(self.cfg["environment"]["width"] * self.resolution)

        self.obs_space = spaces.Box(low=-1,high=1,shape=(self.h,self.w),dtype=np.float32)
        self.action_space = spaces.Box(low=0,high=1,shape=(2,),dtype=np.float32)

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass


