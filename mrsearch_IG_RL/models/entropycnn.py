import gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class EntropyCnn(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(EntropyCnn, self).__init__(observation_space, features_dim)
        n_input_dim = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_dim, 32, kernel_size=11, stride=5, padding=0),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=6, stride=3, padding=0),
            nn.Tanh(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Tanh())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.linear(self.cnn(observations))
        return x