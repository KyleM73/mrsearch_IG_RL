from mrsearch_IG_RL.external import *

class ICM_DummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super(ICM_DummyVecEnv, self).__init__(env_fns)
        self.intrinsic_rewards = None

    def step_async(self, actions: np.ndarray, intrinsic_rewards: np.ndarray) -> None:
        self.actions = actions
        self.intrinsic_rewards = intrinsic_rewards

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                [self.actions[env_idx,:], self.intrinsic_rewards[env_idx,:]]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def step(self, actions: np.ndarray, intrinsic_rewards: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions, intrinsic_rewards)
        return self.step_wait()

class ICM_SubprocVecEnv(SubprocVecEnv):
    def step_async(self, actions: np.ndarray, intrinsic_rewards: np.ndarray) -> None:
        for remote, action, reward in zip(self.remotes, actions, intrinsic_rewards):
            remote.send(("step", [action, reward]))
        self.waiting = True

    def step(self, actions: np.ndarray, intrinsic_rewards: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions, intrinsic_rewards)
        return self.step_wait()

class ICM_Monitor(VecMonitor):
    def step_async(self, actions: np.ndarray, intrinsic_rewards: np.ndarray) -> None:
        self.venv.step_async(actions, intrinsic_rewards)

    def step(self, actions: np.ndarray, intrinsic_rewards: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions, intrinsic_rewards)
        return self.step_wait()







