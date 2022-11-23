import os
import yaml
import time
import numpy as np

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import EnvCreator

PATH_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class base_env(Env):
    def __init__(self,training=True,cfg=None):
        if isinstance(cfg,str):
            assert os.path.exists(cfg), "configuration file specified does not exist"
            with open(cfg, 'r') as stream:
                self.cfg = yaml.safe_load(stream)#,Loader=yaml.FullLoader)
        else:
            raise AssertionError("no configuration file specified")

        if training:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.training = training
        
        ## simulation params
        self.horizon = self.cfg["simulation"]["horizon"]
        self.dt = self.cfg["simulation"]["dt"]

        ## environment params
        self.resolution = self.cfg["environment"]["resolution"]
        self.h = int(self.cfg["environment"]["height"] * self.resolution)
        self.w = int(self.cfg["environment"]["width"] * self.resolution)
        self.fname = self.cfg["environment"]["filename"]
        self.env_c = EnvCreator.envCreator(PATH_DIR+self.fname,resolution=self.resolution)
        self.map = self.env_c.image2occupancy()
        self.map_urdf = self.cfg["environment"]["urdf"]

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
        self.walls = p.loadURDF(self.map_urdf,useFixedBase=True)
        p.changeDynamics(self.walls, -1, lateralFriction=1.0)

        self._get_obs()
        self._get_rew()

    def step(self):
        self.client.stepSimulation()

    def render(self):
        pass

    def _get_obs(self):
        pass

    def _get_rew(self):
        pass

if __name__ == "__main__":

    env = base_env(False,PATH_DIR+"/cfg/base_env.yaml")
    env.reset()

    for _ in range(100000):
        env.step()
        time.sleep(0.1)


        



        





