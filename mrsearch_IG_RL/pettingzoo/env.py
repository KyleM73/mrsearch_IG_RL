import datetime
import functools
import gymnasium as gym
import numpy as np
import os
import pettingzoo as pz
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import torch
import yaml

import util #bresenham

def wrap(env_fn, headless, record, cfg):
    env = env_fn(headless=headless, record=record, cfg=cfg)
    env = pz.utils.parallel_to_aec(env)
    env = wrappers.OrderEnforcingWrapper(env) #helpful user errors
    return env

class icm_env(pz.ParallelEnv):
    metadata = {"name": "icm_v1"}

    def __init__(self, headless=False, record=False, cfg=None):
        if isinstance(cfg,str):
            assert os.path.exists(cfg), "configuration file specified does not exist"
            with open(cfg, 'r') as stream:
                self.cfg = yaml.safe_load(stream)
            self._load_config()
        else:
            raise AssertionError("no configuration file specified")

        self.possible_agents = self.cfg["agents"]["agents"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # vec obs : vel [1] heading [1] action [1] => 3
        # actions : desired heading [1] intrinsic rewards [2] => 3
        self._observation_spaces = {
            agent: gym.spaces.Dict({
                "img" : gym.spaces.Box(low=-1,high=1,shape=(self.n_frames,self.obs_w,self.obs_w),dtype=np.float32),
                "vec" : gym.spaces.Box(low=-1,high=1,shape=(3,),dtype=np.float32)
                }
            ) for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: gym.spaces.Box(
                low=-1,high=1,shape=(3,),dtype=np.float32
            ) for agent in self.possible_agents
        }
        
        self.headless = headless
        if self.headless:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.record = record

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return gym.spaces.Dict({
                "img" : gym.spaces.Box(low=-1,high=1,shape=(self.n_frames,self.obs_w,self.obs_w),dtype=np.float32),
                "vec" : gym.spaces.Box(low=-1,high=1,shape=(3,),dtype=np.float32)
                }
            )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Box(
                low=-1,high=1,shape=(1,),dtype=np.float32
            )
    def observe(self, agent):
        return self.observations[agent]

    def close(self):
        self.client.disconnect()

    def reset(self, return_info=False):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {"pose" : None, "ori" : None, "vel" : None} for agent in self.agents}
        self.observations = {agent: {"img" : None, "vec" : None} for agent in self.agents}
        self.collisions = {agent : False for agent in self.agents}
        self.detections = {agent : False for agent in self.agents}
        self.t = 0

        self._agent_selector = pz.utils.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._load_config()

        ## initiate simulation
        self.client.resetSimulation()
        self.client.setTimeStep(self.dt)
        self.client.setGravity(0,0,-9.8)

        ## setup ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = p.loadURDF("plane.urdf")
        p.changeDynamics(self.ground, -1, lateralFriction=0.0,spinningFriction=0.0)

        ## setup walls
        self.walls = p.loadURDF(self.map_urdf,useFixedBase=True)
        p.changeDynamics(self.walls, -1, lateralFriction=0.0,spinningFriction=0.0)

        ## setup robots
        for agent in self.agents:
            pose,ori = self._get_random_pose()
            robot = p.loadURDF(self.robot_urdf, basePosition=pose, baseOrientation=E2Q([0,0,ori]))
            p.changeDynamics(robot, -1, lateralFriction=0.0, spinningFriction=0.0)
            self.observations[agent]["vec"] = np.array([0,ori/np.pi,0])
            self.state[agent]["pose"] = pose
            self.state[agent]["ori"] = ori
            self.state[agent]["vel"] = 0

        ## setup target
        self.target_pose,_ = self._get_random_pose()
        self.target_pose[2] = 0.5
        self.target = p.loadURDF(self.target_urdf, basePosition=self.target_pose, useFixedBase=True)

        ## initialize objects
        for _ in range(10):
            self.client.stepSimulation()

        ## setup recording
        if self.record:
            self._setup_recording()

        if not return_info:
            return self.observations
        else:
            infos = {agent: {} for agent in self.agents}
            return self.observations, self.infos

    def step(self, actions):
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        for agent in self.agents:
            self._act(agent, actions[agent])

        self.observations = {agent : self._get_obs(agent) for agent in self.agents}

        self.rewards = {agent : self._get_rew(agent, actions[agent]) for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}

        self.t += 1
        env_truncation = self.t >= self.max_steps
        self.truncations = {agent: env_truncation for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _act(self, agent, action):
        pass

    def _get_random_pose(self):
        while True:
            pose = torch.round(torch.rand((2,))*torch.tensor([self.h,self.w])).to(int).tolist()
            if self._is_valid(*pose):
                pose = self._rc2xy(*pose)
                pose.append(self.robot_height/2+1e-4)
                ori = 2*np.pi*torch.rand((1)).item()-np.pi
                return [pose,ori]

    def _is_valid(self, r, c, convert=False):
        if convert:
            r,c = self._xy2rc(r,c) #xy -> rc
        padded_map = torch.nn.functional.pad(self.map,(self.pad_l,self.pad_l,self.pad_l,self.pad_l),value=1)
        footprint = padded_map[r+self.pad_l-5:r+self.pad_l+5,c+self.pad_l-5:c+self.pad_l+5]
        collisions = torch.linalg.norm(footprint@self.k)
        if collisions:
            return False
        return True

    def _xy2rc(self, x, y):
        r = int(-y / self.resolution + (self.h + 1) / 2)
        r = max(min(r,self.h),0)
        c = int(x / self.resolution + (self.w + 1) / 2)
        c = max(min(c,self.w),0)
        return [r,c] 

    def _rc2xy(self, r, c):
        x = self.resolution * (c - (self.w + 1) / 2 )
        y = self.resolution * (-r + (self.h + 1) / 2)
        return [x,y]

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

    def _load_config(self):
        ## time
        self.date = datetime.datetime.now().strftime('%m%d_%H%M')

        ## record params
        self.log_dir = self.cfg["record"]["log_dir"]
        self.log_name = "/"+self.date+".mp4"
        self.log_name_obs = "/"+self.date+"_crop.mp4"

        ## simulation params
        self.horizon = self.cfg["simulation"]["horizon"]
        self.dt = self.cfg["simulation"]["dt"]
        self.policy_dt = self.cfg["simulation"]["policy_dt"]
        self.max_steps = int(self.horizon / self.policy_dt)
        self.repeat_action = int(self.policy_dt / self.dt)
        self.pad_l = self.cfg["simulation"]["pad_l"]
        self.entropy_decay = self.cfg["simulation"]["entropy_decay"]
        self.waypt_dist = self.cfg["simulation"]["waypoint_dist"]
        self.Kp = self.cfg["simulation"]["Kp"]
        self.Kd = self.cfg["simulation"]["Kd"]
        self.Ki = self.cfg["simulation"]["Ki"]
        self.n_frames = self.cfg["simulation"]["n_frames"]
        self.obs_w = self.cfg["simulation"]["obs_dim"]

        ## environment params
        self.resolution = self.cfg["environment"]["resolution"]
        self.h = int(self.cfg["environment"]["height"] / self.resolution)
        self.w = int(self.cfg["environment"]["width"] / self.resolution)
        self.scale_factor = self.cfg["environment"]["scale_factor"]
        self.map_fname = self.cfg["environment"]["filename"]
        self.map = torch.from_numpy(plt.imread(PATH_DIR+self.map_fname)[:,:,-1])
        self.map_urdf = self.cfg["environment"]["urdf"]
        self.target_urdf = self.cfg["environment"]["target_urdf"]

        ## robot params
        self.robot_urdf = self.cfg["robot"]["urdf"]
        self.robot_width = self.cfg["robot"]["width"]
        self.robot_depth = self.cfg["robot"]["depth"]
        self.robot_height = self.cfg["robot"]["height"]
        self.mass = self.cfg["robot"]["mass"]
        self.mass_matrix = 0.5*self.mass*torch.linalg.norm(torch.tensor([self.robot_width,self.robot_depth,self.robot_height])).item()**2 #I=0.5*m*r**2
        self.max_accel = self.cfg["robot"]["max_linear_accel"]
        self.max_vel = self.cfg["robot"]["max_linear_vel"]
        self.max_aaccel = self.cfg["robot"]["max_angular_accel"]
        self.max_avel = self.cfg["robot"]["max_angular_vel"]

        ## LiDAR params
        self.scan_density_coef = self.cfg["lidar"]["density"]
        self.scan_range = self.cfg["lidar"]["range"]
        self.scan_height = self.cfg["lidar"]["scan_height"]
        self.target_threshold = self.cfg["lidar"]["target_threshold"]
        self.FOV = 2 * np.pi * self.cfg["lidar"]["FOV"] / 360 #deg2rad
        self.scan_density = self.scan_density_coef * 2 * np.pi / np.arctan2(self.resolution,self.scan_range) # resolution / max(height, width)
        self.num_scans = int(self.FOV / (2 * np.pi) * self.scan_density)
        self.scan_angle = self.FOV / self.num_scans #rad
        self.lidar_threads = self.cfg["lidar"]["threads"] if self.vecenv else 1

        ## reward params
        self.detection_reward = self.cfg["rewards"]["detection"]
        self.collision_reward = self.cfg["rewards"]["collision"]
        self.beta = self.cfg["rewards"]["beta"]
        self.lmbda = self.cfg["rewards"]["lambda"]
        self.eta = self.cfg["rewards"]["eta"]

        ## kernel
        self.kernel_size = self.cfg["kernel"]
        self.k = torch.ones((self.kernel_size, self.kernel_size))
        









