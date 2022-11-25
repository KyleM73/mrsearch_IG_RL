import os
import yaml
import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env

import pybullet as p
from pybullet import getQuaternionFromEuler as E2Q
from pybullet import getEulerFromQuaternion as Q2E
import pybullet_utils.bullet_client as bc
import pybullet_data

import mrsearch_IG_RL
from mrsearch_IG_RL import PATH_DIR,CFG_DIR

class base_env(Env):
    def __init__(self,training=True,cfg=None):
        self.dt = datetime.datetime.now().strftime('%m%d_%H%M')
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
        self._load_config()

        ## model I/O
        # observe : cropped entropy map
        # act     : XY waypts, desired heading
        self.obs_w = min(self.h,self.w) + 1
        self.observation_space = spaces.Box(low=-1,high=1,shape=(1,self.obs_w,self.obs_w),dtype=np.float32)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,),dtype=np.float32)

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

        ## setup robot
        self.pose,self.ori = self._get_random_pose()
        self.robot = p.loadURDF(self.robot_urdf,basePosition=self.pose,baseOrientation=E2Q([0,0,self.ori]))
        
        ## setup target
        self.target_pose,_ = self._get_random_pose()
        self.target_pose[2] = 0.5
        self.target = p.loadURDF(self.target_urdf,basePosition=self.target_pose,useFixedBase=True)
        
        ## initialize objects
        for _ in range(10):
            self.client.stepSimulation()

        ## reset entropy
        self.entropy = torch.where(self.map==1,1,0)
        self.information = torch.sum(torch.abs(self.entropy))
        self.information0 = self.information.clone()
        self.detection = False
        self.done = False
        self.t = 0

        ## setup recording
        if self.record:
            self._setup_recording()

        self._get_obs()
        self._get_rew()

        return torch.unsqueeze(self.crop,0).numpy()

    def step(self,action):
        self._act(action)
        self._get_obs()
        self._get_rew()
        self.t += 1

        if self.done and self.record:
            self._save_videos()

        return torch.unsqueeze(self.crop,0).numpy(), self.reward, self.done, self.dictLog

    def _act(self,action):
        self.force = [self.max_accel*action[0],self.max_accel*action[1],0]
        self.torque = [0,0,self.max_aaccel*action[2]]
        for i in range(len(action)):
            if self.vel[i] + self.dt*self.force[i] > self.max_vel:
                self.force[i] = 0
            if self.avel[i] + self.dt*self.torque[i] > self.max_avel:
                self.torque[i] = 0
        p.applyExternalForce(self.robot,-1,self.force,[0,0,0],p.LINK_FRAME)
        p.applyExternalTorque(self.robot,-1,self.torque,p.LINK_FRAME)
        
        for _ in range(self.repeat_action):
            self.client.stepSimulation()

    def _get_obs(self):
        self._get_pose()
        self._decay_entropy()
        self._get_scans()
        self._get_crop()
        self._get_IG()

        if self.record:
            self._save_entropy()

    def _get_rew(self):
        dictState = {}
        dictState["entropy"] = self.entropy
        dictState["pose"] = self.pose

        dictRew = {}
        dictRew["IG"] = self.IG.item()
        dictRew["vel"] = -torch.linalg.norm(torch.tensor(self.vel)).item()
        dictRew["avel"] = -torch.linalg.norm(torch.tensor(self.avel)).item()
        if self.detection:
            self.done = True
            dictRew["Detection"] = self.detection_reward
        elif self.t >= self.max_steps:
            self.done = True

        self.reward = 0
        for rew in dictRew.values():
            self.reward += rew
        dictRew["Sum"] = self.reward

        self.dictLog = {}
        if self.done:
            self.dictLog["Done"] = 1
        self.dictLog["Reward"] = dictRew
        self.dictLog["State"] = dictState

    def _get_pose(self):
        self.pose, oris = p.getBasePositionAndOrientation(self.robot)
        self.pose_rc = self._xy2rc(*self.pose[:2])
        self.ori = Q2E(oris)[2]
        self.vel,self.avel = p.getBaseVelocity(self.robot)

        self.target_pose,_ = p.getBasePositionAndOrientation(self.target)
        self.target_pose_rc = self._xy2rc(*self.target_pose[:2])

    def _setup_recording(self):
        try:
            plt.close("all")
        except:
            pass
        self.fig,self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=None,hspace=None)
        self.ax.set_axis_off()
        self.frames = []
        self.fig_obs,self.ax_obs = plt.subplots()
        self.fig_obs.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=None,hspace=None)
        self.ax_obs.set_axis_off()
        self.frames_obs = []
        self.fps = int(self.policy_dt**-1)
        self.writer = animation.FFMpegWriter(fps=self.fps) 

    def _save_entropy(self):
        r,c = self.pose_rc
        tr,tc = self.target_pose_rc
        entropy_marked = self.entropy.clone()
        entropy_marked[r-1:r+2,c-1:c+2] = 1
        entropy_marked[tr-2:tr+4,tc-2:tc+4] = 1
        self.frames.append([self.ax.imshow(entropy_marked.numpy(),animated=True,vmin=-1,vmax=1)])
        self.frames_obs.append([self.ax_obs.imshow(self.crop.numpy(),animated=True,vmin=-1,vmax=1)])

    def _save_videos(self):
        ani = animation.ArtistAnimation(self.fig,self.frames,interval=int(1000/self.fps),blit=True,repeat=False)
        ani.save(PATH_DIR+self.log_dir+self.log_name,writer=self.writer)

        ani_obs = animation.ArtistAnimation(self.fig_obs,self.frames_obs,interval=int(1000/self.fps),blit=True,repeat=False)
        ani_obs.save(PATH_DIR+self.log_dir+self.log_name_obs,writer=self.writer)

    def _get_IG(self):
        self.IG = torch.sum(torch.abs(self.entropy)) - self.information
        self.information = torch.sum(torch.abs(self.entropy))

    def _get_crop(self):
        # left side
        if self.pose_rc[1] < self.obs_w:
            self.crop = self.entropy[:,:self.obs_w]
        #right side
        elif self.h - self.pose_rc[1] < self.obs_w:
            self.crop = self.entropy[:,-self.obs_w:]
        #center crop on robot
        else:
            self.crop = self.entropy[:,self.pose_rc[1]-int(self.obs_w/2):self.pose_rc+int(self.obs_w/2)]
        
        assert self.crop.size() == (self.obs_w,self.obs_w)

    def _decay_entropy(self):
        self.entropy = self.entropy_decay * self.entropy

    def _get_scans(self):
        origins = [[self.pose[0],self.pose[1],0.5] for i in range(self.num_scans)]
        endpts = [
            [
                self.scan_range * math.cos(i*self.scan_angle + self.ori - self.FOV/2) + self.pose[0],
                self.scan_range * math.sin(i*self.scan_angle + self.ori - self.FOV/2) + self.pose[1],
                0.5
            ]
                for i in range(self.num_scans)]
        scan_data = p.rayTestBatch(origins,endpts,numThreads=self.lidar_threads)
        scans = []
        # scans: [scanX,scanY,endpointCollision,targetHitDist]
        for i in range(self.num_scans):
            if scan_data[i][0] == self.target:
                dist = torch.linalg.norm(torch.tensor(scan_data[i][3][:2])).item()
                scans.append([scan_data[i][3][:2],True,dist])
            elif scan_data[i][0] != -1: #check if hit detected
                scans.append([scan_data[i][3][:2],True,10])
            else:
                scans.append([endpts[i][:2],False,10])
        
        for hit in scans:
            hr,hc = self._xy2rc(*hit[0])
            sr,sc = self.pose_rc
            if hit[1]:
                self.entropy[hr,hc] = 1
                if hit[2] < 3:
                    self.detection = True
            free = self._bresenham((sr,sc),(hr,hc))
            for f in free:
                self.entropy[f[0],f[1]] = -1
        self.entropy += torch.where(self.map==1,1,0)
        self.entropy = torch.clamp(self.entropy,-1,1)

    def _get_random_pose(self):
        while True:
            pose = torch.round(torch.rand((2,))*torch.tensor([self.h,self.w])).to(int).tolist()
            if self._is_valid(*pose):
                pose = self._rc2xy(*pose)
                pose.append(self.robot_height/2+1e-4)
                ori = 2*math.pi*torch.rand((1)).item()-math.pi
                return [pose,ori]

    def _is_done(self):
        return self.done

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
        r = int(-y / self.resolution + (self.h + 1) / 2)
        r = max(min(r,self.h),0)
        c = int(x / self.resolution + (self.w + 1) / 2)
        c = max(min(c,self.w),0)
        return [r,c] 

    def _rc2xy(self,r,c):
        x = self.resolution * (c - (self.w + 1) / 2 )
        y = self.resolution * (-r + (self.h + 1) / 2)
        return [x,y]

    def _bresenham(self,start,end):
        """
        Adapted from PythonRobotics:
        https://github.com/AtsushiSakai/PythonRobotics/blob/master/Mapping/lidar_to_grid_map/lidar_to_grid_map.py

        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a list from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        [[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]]
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        return points[1:-1] #do not include endpoints

    def _load_config(self):
        ## record params
        self.record = self.cfg["record"]["record"]
        self.log_dir = self.cfg["record"]["log_dir"]
        self.log_name = "/"+self.dt+".mp4" #self.cfg["record"]["log_name"]
        self.log_name_obs = "/"+self.dt+"_crop.mp4" #self.cfg["record"]["log_name_obs"]

        ## simulation params
        self.horizon = self.cfg["simulation"]["horizon"]
        self.dt = self.cfg["simulation"]["dt"]
        self.policy_dt = self.cfg["simulation"]["policy_dt"]
        self.max_steps = int(self.horizon / self.policy_dt)
        self.repeat_action = int(self.policy_dt / self.dt)
        self.pad_l = self.cfg["simulation"]["pad_l"]
        self.entropy_decay = self.cfg["simulation"]["entropy_decay"]
        
        ## environment params
        self.resolution = self.cfg["environment"]["resolution"]
        self.h = int(self.cfg["environment"]["height"] / self.resolution)
        self.w = int(self.cfg["environment"]["width"] / self.resolution)
        self.map_fname = self.cfg["environment"]["filename"]
        self.map = torch.from_numpy(plt.imread(PATH_DIR+self.map_fname)[:,:,-1])
        self.map_urdf = self.cfg["environment"]["urdf"]
        self.target_urdf = self.cfg["environment"]["target_urdf"]
        
        ## robot params
        self.robot_urdf = self.cfg["robot"]["urdf"]
        self.robot_width = self.cfg["robot"]["width"]
        self.robot_depth = self.cfg["robot"]["depth"]
        self.robot_height = self.cfg["robot"]["height"]
        self.max_accel = self.cfg["robot"]["max_linear_accel"]
        self.max_vel = self.cfg["robot"]["max_linear_vel"]
        self.max_aaccel = self.cfg["robot"]["max_angular_accel"]
        self.max_avel = self.cfg["robot"]["max_angular_vel"]

        ## LiDAR params
        self.scan_density_coef = self.cfg["lidar"]["density"]
        self.scan_range = self.cfg["lidar"]["range"]
        self.FOV = 2 * math.pi * self.cfg["lidar"]["FOV"] / 360 #deg2rad
        self.scan_density = self.scan_density_coef * 2 * math.pi / math.atan(self.resolution/self.scan_range) # resolution / max(height, width)
        self.num_scans = int(self.FOV / (2 * math.pi) * self.scan_density)
        self.scan_angle = self.FOV / self.num_scans #rad
        self.lidar_threads = self.cfg["lidar"]["threads"]

        ## reward params
        self.detection_reward = self.cfg["rewards"]["detection"]

if __name__ == "__main__":
    env = base_env(False,CFG_DIR+"/base.yaml")
    check_env(env)