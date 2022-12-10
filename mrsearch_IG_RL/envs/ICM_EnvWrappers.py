import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

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







