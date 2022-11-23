import os
import yaml
import numpy as np

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

PATH_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class base_env(env):
    def __init__(self,training=True,cfg=None):
        if ifinstance(cfg,str):
            assert os.path.exists(cfg), "configuration file specified does not exist"
            cfg = yaml.load(cfg)
        else:
            raise AssertionError("no configuration file specified")

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
        self.fname = self.cfg["environment"]["filename"]
        print("generating urdf...")
        self.env_c = EnvCreator.envCreator(PATH_DIR+self.fname)
        self.env_urdf = env_c.get_urdf_fast()
        print("done.")

        self.obs_space = spaces.Box(low=-1,high=1,shape=(self.h,self.w),dtype=np.float32)
        self.action_space = spaces.Box(low=0,high=1,shape=(2,),dtype=np.float32)

    def reset(self):
        
        ## initiate simulation
        self.client.resetSimulation()
        self.client.setTimeStep(self.dt)
        self.client.setGravity(0,0,-9.8)

        ## setup ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = p.loadURDF("plane.urdf")
        p.changeDynamics(self.ground, -1, lateralFriction=0.0) 

        ## setup walls
        self.walls = p.loadURDF(self.env_urdf,useFixedBase=True)
        p.changeDynamics(self.walls, -1, lateralFriction=1.0)

        self._get_obs()
        self._get_rew()

    def step(self):
        pass

    def render(self):
        pass

    def _get_obs(self):
        pass

    def _get_rew(self):
        pass

        



        





