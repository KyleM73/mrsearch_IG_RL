import datetime
import functools
import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pettingzoo as pz
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import torch
import yaml

from mrsearch_IG_RL import PATH_DIR
import util

def wrap(env_fn, headless, record, cfg):
    env = env_fn(headless=headless, record=record, cfg=cfg)
    env = pz.utils.parallel_to_aec(env)
    env = wrappers.OrderEnforcingWrapper(env) #helpful user errors
    return env

class icm_env(pz.ParallelEnv):
    metadata = {"name": "icm_v1"}

    def __init__(self, headless=False, record=False, cfg=None, output_dir=None):
        self.record = record
        if self.record: self.output_dir = output_dir

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
                low=-1,high=1,shape=(1,),dtype=np.float32
            ) for agent in self.possible_agents
        }
        
        self.headless = headless
        if self.headless:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

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
        self.detections = {agent: False for agent in self.agents}
        self.forces = {agent : np.array([0,0,0]) for agent in self.agents}
        self.torques = {agent : np.array([0,0,0]) for agent in self.agents}
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
        self.pb_agents = {}
        for agent in self.agents:
            pose,ori = self._get_random_pose()
            robot = p.loadURDF(self.robot_urdf, basePosition=pose, baseOrientation=p.getQuaternionFromEuler([0,0,ori]))
            p.changeDynamics(robot, -1, lateralFriction=0.0, spinningFriction=0.0)
            self.observations[agent]["vec"] = np.array([0,ori/np.pi,0])
            self.state[agent]["pose"] = pose
            self.state[agent]["pose_rc"] = self._xy2rc(*pose[:2])
            self.state[agent]["ori"] = ori
            self.state[agent]["vel"] = 0
            self.pb_agents[agent] = robot

        ## setup target
        self.target_pose,_ = self._get_random_pose()
        self.target_pose[2] = 0.5
        self.target_pose_rc = self._xy2rc(*self.target_pose[:2])
        self.target = p.loadURDF(self.target_urdf, basePosition=self.target_pose, useFixedBase=True)

        ## initialize objects
        for _ in range(10):
            self.client.stepSimulation()

        ## setup recording
        if self.record:
            self.figs = {agent : {} for agent in self.agents}
            self._setup_recording()

        ## initialize entropy
        self.entropy = torch.where(self.map==1,1,0)

        self._get_obs({agent : np.array([0]) for agent in self.agents})

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

        self.collisions = {agent : False for agent in self.agents} #reset every step

        self._act(actions)
        self._get_obs(actions)
        self._get_rew()

        self.terminations = {agent: self.detections[agent] for agent in self.agents}

        self.t += 1
        env_truncation = self.t >= self.max_steps
        self.truncations = {agent: env_truncation for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}

        if self.record:
            if any([self.terminations[agent] for agent in self.agents]) or any([self.truncations[agent] for agent in self.agents]):
                self._save_videos()

        if env_truncation:
            self.agents = []

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _act(self, actions):
        for agent in self.agents:
            d_heading = np.pi * actions[agent]
            if isinstance(d_heading,torch.Tensor):
                d_heading = d_heading.flatten().numpy()
            self.forces[agent] = self.mass * self.max_accel * np.array([np.cos(d_heading)[0],np.sin(d_heading)[0],0]).reshape((-1,))
            err = d_heading - (self.state[agent]["ori"] - np.pi/2)
            self.torques[agent] = self.mass_matrix * self.max_aaccel * np.array([0,0,self.Kp*err[0]/np.pi]).reshape((-1,))

        for _ in range(self.repeat_action):
            for agent in self.agents:
                p.applyExternalForce(self.pb_agents[agent],-1,self.forces[agent],[0,0,0],p.LINK_FRAME)
                p.applyExternalTorque(self.pb_agents[agent],-1,self.torques[agent],p.LINK_FRAME)
                self.client.stepSimulation()

                ## collision detection
                ctx = p.getContactPoints(self.pb_agents[agent],self.walls)
                if len(ctx) > 0:
                    self.collisions[agent] = True
                else:
                    for agent_ in self.agents:
                        if agent != agent_:
                            ctx = p.getContactPoints(self.pb_agents[agent],self.pb_agents[agent_])
                            if len(ctx) > 0:
                                self.collisions[agent] = True
        
    def _get_obs(self, actions):

        self._get_states()
        self._decay_entropy()
        self._get_scans()
        self._get_crops()

        for agent in self.agents:
            self.observations[agent]["vec"] = np.array([
                self.state[agent]["vel"],
                self.state[agent]["ori"]/np.pi,
                actions[agent][0]
                ],dtype=np.float32)

        if self.record:
            self._save_entropy()

    def _get_states(self):
        for agent in self.agents:
            pose, oris = p.getBasePositionAndOrientation(self.pb_agents[agent])
            pose_rc = self._xy2rc(*pose[:2])
            ori = p.getEulerFromQuaternion(oris)[2]
            vel,_ = p.getBaseVelocity(self.pb_agents[agent])

            self.state[agent]["pose"] = pose 
            self.state[agent]["pose_rc"] = pose_rc
            self.state[agent]["ori"] = ori 
            self.state[agent]["vel"] = float(np.linalg.norm(vel[:2]))

        self.target_pose,_ = p.getBasePositionAndOrientation(self.target)
        self.target_pose_rc = self._xy2rc(*self.target_pose[:2])

    def _decay_entropy(self):
        self.entropy = self.entropy_decay * self.entropy

    def _get_scans(self):
        for agent in self.agents:
            origins = [[self.state[agent]["pose"][0],self.state[agent]["pose"][1],self.scan_height] for i in range(self.num_scans)]
            endpts = [
                [
                    self.scan_range * np.cos(i*self.scan_angle + self.state[agent]["ori"] - self.FOV/2) + self.state[agent]["pose"][0],
                    self.scan_range * np.sin(i*self.scan_angle + self.state[agent]["ori"] - self.FOV/2) + self.state[agent]["pose"][1],
                    self.scan_height
                ]
                    for i in range(self.num_scans)]
            scan_data = p.rayTestBatch(origins,endpts,numThreads=self.lidar_threads)
            scans = []
            # scans: [scanX,scanY,endpointCollision,targetHitDist]
            for i in range(self.num_scans):
                if scan_data[i][0] == self.target:
                    dist = torch.linalg.norm(torch.tensor(scan_data[i][3][:2]) - torch.tensor(self.state[agent]["pose"][:2])).item()
                    scans.append([scan_data[i][3][:2],True,dist])
                elif scan_data[i][0] != -1: #check if hit detected
                    scans.append([scan_data[i][3][:2],True,10])
                else:
                    scans.append([endpts[i][:2],False,10])
            
            flag = False
            for hit in scans:
                hr,hc = self._xy2rc(*hit[0])
                sr,sc = self.state[agent]["pose_rc"]
                if hit[1]:
                    self.entropy[hr,hc] = 1
                    if hit[2] < self.target_threshold:
                        self.detections[agent] = True
                        flag = True
                free = util.bresenham((sr,sc),(hr,hc))
                for f in free:
                    self.entropy[f[0],f[1]] = -1
            if flag:
                print("Target found.")
            self.entropy += torch.where(self.map==1,1,0)
            self.entropy = torch.clamp(self.entropy,-1,1)

    def _get_crops(self):
        entropy_marked = self.entropy.clone()
        for agent in self.agents:
            r,c = self.state[agent]["pose_rc"]
            entropy_marked[r-4:r+5,c-4:c+5] = 0.5 #robot

        for agent in self.agents:
            r,c = self.state[agent]["pose_rc"]
            #entropy_marked = self.entropy.clone()
            #entropy_marked[r-2:r+3,c-2:c+3] = 0.5 #robot
            #entropy_marked[r-5:r+6,c-5:c+6] = 1
            #entropy_marked = torch.where(self.map==1,1.,0.)
            # top side
            if r < self.obs_w/2:
                # left side
                if c < self.obs_w/2:
                    crop = entropy_marked[:self.obs_w,:self.obs_w]
                # right side
                elif c > self.w - self.obs_w/2:
                    crop = entropy_marked[:self.obs_w,-self.obs_w:]
                else:
                    if self.obs_w % 2:
                        crop = entropy_marked[:self.obs_w,c-self.obs_w//2:c+self.obs_w//2+1]
                    else:
                        crop = entropy_marked[:self.obs_w,c-self.obs_w//2:c+self.obs_w//2]
            # bottom side
            elif r > self.h - self.obs_w/2:
                # left side
                if c < self.obs_w/2:
                    crop = entropy_marked[-self.obs_w:,:self.obs_w]
                # right side
                elif c > self.w - self.obs_w/2:
                    crop = entropy_marked[-self.obs_w:,-self.obs_w:]
                else:
                    if self.obs_w % 2:
                        crop = entropy_marked[-self.obs_w:,c-self.obs_w//2:c+self.obs_w//2+1]
                    else:
                        crop = entropy_marked[-self.obs_w:,c-self.obs_w//2:c+self.obs_w//2]
            else:
                if self.obs_w % 2:
                    # left side
                    if c < self.obs_w/2:
                        crop = entropy_marked[r-self.obs_w//2:r+self.obs_w//2+1,:self.obs_w]
                    # right side
                    elif c > self.w - self.obs_w/2:
                        crop = entropy_marked[r-self.obs_w//2:r+self.obs_w//2+1,-self.obs_w:]
                    #center crop on robot
                    else:
                        crop = entropy_marked[r-self.obs_w//2:r+self.obs_w//2+1,c-self.obs_w//2:c+self.obs_w//2+1]
                else:
                    # left side
                    if c < self.obs_w/2:
                        crop = entropy_marked[r-self.obs_w//2:r+self.obs_w//2,:self.obs_w]
                    # right side
                    elif c > self.w - self.obs_w/2:
                        crop = entropy_marked[r-self.obs_w//2:r+self.obs_w//2,-self.obs_w:]
                    #center crop on robot
                    else:
                        crop = entropy_marked[r-self.obs_w//2:r+self.obs_w//2,c-self.obs_w//2:c+self.obs_w//2]
            
            #assert crop.size() == (self.obs_w,self.obs_w)

            if self.t == 0:
                self.observations[agent]["img"] = torch.unsqueeze(crop,0).repeat(self.n_frames,1,1)
            else:
                self.observations[agent]["img"] = torch.cat((torch.unsqueeze(crop,0),self.observations[agent]["img"]),dim=0)[:self.n_frames,:,:]

    def _get_rew(self):
        for agent in self.agents:
            self.rewards[agent] = self.lmbda * self.detection_reward * self.detections[agent]
            self.rewards[agent] = self.collision_reward * self.collisions[agent]

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
        for agent in self.agents:
            fig_obs,ax_obs = plt.subplots()
            fig_obs.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=None,hspace=None)
            ax_obs.set_axis_off()
            self.figs[agent]["fig"] = fig_obs
            self.figs[agent]["ax"] = ax_obs
            self.figs[agent]["frames"] = []
        self.fps = int(self.policy_dt**-1)
        self.writer = animation.FFMpegWriter(fps=self.fps) 

    def _save_entropy(self):
        entropy_marked = self.entropy.clone()
        tr,tc = self.target_pose_rc
        entropy_marked[tr-3:tr+4,tc-3:tc+4] = 1
        for agent in self.agents:
            r,c = self.state[agent]["pose_rc"]
            entropy_marked[r-2:r+3,c-2:c+3] = 1
            self.figs[agent]["frames"].append([
                self.figs[agent]["ax"].imshow(
                    self.observations[agent]["img"][0,...].numpy(),
                    animated=True,vmin=-1,vmax=1)
                ])
            #self.frames_obs.append([self.ax_obs.imshow(self.crop.numpy(),animated=True,vmin=-1,vmax=1)])
        self.frames.append([self.ax.imshow(entropy_marked.numpy(),animated=True,vmin=-1,vmax=1)])
        
    def _save_videos(self):
        ani = animation.ArtistAnimation(self.fig,self.frames,interval=int(1000/self.fps),blit=True,repeat=False)
        ani.save("."+self.log_dir+self.log_name+".mp4",writer=self.writer)
        print("Video saved to "+"."+self.log_dir+self.log_name+".mp4")

        for agent in self.figs.keys():
            ani_obs = animation.ArtistAnimation(self.figs[agent]["fig"],self.figs[agent]["frames"],interval=int(1000/self.fps),blit=True,repeat=False)
            ani_obs.save("."+self.log_dir+self.log_name_obs+"_crop_{}.mp4".format(agent),writer=self.writer)
            print("Video saved to "+"."+self.log_dir+self.log_name_obs+"_crop_{}.mp4".format(agent))

        print("Saving videos complete.")

    def _load_config(self):
        ## time
        self.date = datetime.datetime.now().strftime('%m%d_%H%M')

        ## record params
        self.log_dir = self.cfg["record"]["log_dir"]
        self.log_name = "/"+self.date
        self.log_name_obs = "/"+self.date
        if self.record and self.output_dir is not None:
            self.log_dir += "/{}".format(self.output_dir)
            if not os.path.exists("./"+self.log_dir):
                os.makedirs("./"+self.log_dir)

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
        self.map = torch.from_numpy(plt.imread("."+self.map_fname)[:,:,-1])
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
        self.lidar_threads = self.cfg["lidar"]["threads"]

        ## reward params
        self.detection_reward = self.cfg["rewards"]["detection"]
        self.collision_reward = self.cfg["rewards"]["collision"]
        self.beta = self.cfg["rewards"]["beta"]
        self.lmbda = self.cfg["rewards"]["lambda"]
        self.eta = self.cfg["rewards"]["eta"]

        ## kernel
        self.kernel_size = self.cfg["environment"]["kernel"]
        self.k = torch.ones((self.kernel_size, self.kernel_size))
        
if __name__ == "__main__":
    import time

    cfg_file = "cfg.yaml"

    env = icm_env(headless=False,record=False,cfg=cfg_file)
    env.reset()

    act = {agent : np.array([0]) for agent in env.agents}

    for i in range(1000):
        obs,rew,term,trunc,info = env.step(act)
        if all(term[agent] for agent in env.agents):
            print("Terminated.")
            break
        if all(trunc[agent] for agent in env.agents):
            print("Time out.")
            break
        if not env.headless:
            time.sleep(0.1)





