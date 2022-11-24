import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

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
        self.pad_l = self.cfg["simulation"]["pad_l"]

        ## environment params
        self.resolution = self.cfg["environment"]["resolution"]
        self.h = int(self.cfg["environment"]["height"] / self.resolution)
        self.w = int(self.cfg["environment"]["width"] / self.resolution)
        self.fname = self.cfg["environment"]["filename"]
        self.map = torch.from_numpy(plt.imread(PATH_DIR+self.fname)[:,:,-1])
        self.map_urdf = self.cfg["environment"]["urdf"]

        ## model I/O
        self.obs_space = spaces.Box(low=-1,high=1,shape=(self.h,self.w),dtype=np.float32)
        self.action_space = spaces.Box(low=0,high=1,shape=(2,),dtype=np.float32)

        ## kernel
        self.k = torch.ones((10,10))

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

        ## setup robots

        ## reset entropy
        self.entropy = torch.where(self.map==1,1,-1)

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

    def _get_random_pose(self):
        while True:
            pose = torch.round(torch.rand((2,))*torch.tensor([self.h,self.w])).to(int)
            if self._is_valid(*pose):
                return pose

    def _is_valid(self,r,c,convert=False):
        if convert:
            r,c = self._xy2rc(r,c) #xy -> rc
        padded_map = torch.nn.functional.pad(self.map,(self.pad_l,self.pad_l,self.pad_l,self.pad_l),value=1)
        footprint = padded_map[r+self.pad_l-5:r+self.pad_l+5,c+self.pad_l-5:c+self.pad_l+5]
        collisions = torch.linalg.norm(footprint@self.k)
        if collisions:
            return False
        return True

    def _xy2rc(self,x,y):
        r = max(min(-int(y/self.resolution) + int(self.h/2) + 1,self.h),0)
        c = max(min(int(x/self.resolution) + int(self.w/2) + 1,self.w),0)
        return r,c 

    def _rc2xy(self,r,c):
        x = self.resolution*(c - int(self.w/self.resolution)/2+1)
        y = self.resolution*(-r + int(self.h/self.resolution)/2+1)
        return x,y

if __name__ == "__main__":

    env = base_env(False,PATH_DIR+"/cfg/base_env.yaml")
    env.reset()

    print(env._get_random_pose())
    assert False

    for _ in range(100000):
        env.step()
        time.sleep(0.1)


        



        





