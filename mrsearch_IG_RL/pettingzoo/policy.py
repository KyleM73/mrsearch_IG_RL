import gymnasium as gym
import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(self, input_dim=3, channels=5, vec_features_dim=8, features_dim=256, cnn_output_dim=512, img_dim=96):
        super(StateEncoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.img_dim = img_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=2, padding=0), #96 -> 45
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=0), #45 -> 21
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=0), #21-> 9
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0), #9 -> 4
            nn.ELU(),
            nn.Flatten(1)) #[-1,512]
        self.projection = nn.Sequential(nn.Linear(input_dim, vec_features_dim), nn.ELU()) #[-1,8]
        self.linear = nn.Sequential(nn.Linear(cnn_output_dim+vec_features_dim, features_dim), nn.ELU())

    def forward(self, img, vec):
        x = self.cnn(img.view(-1,self.channels,self.img_dim,self.img_dim)) #[1,512]
        x = torch.cat((x,self.projection(vec.view(-1,self.input_dim))),dim=1) #[-1,520]
        x = self.linear(x) #[-1,256]
        return x

class ActionDecoder(nn.Module):
    def __init__(self, features_dim=256, latent_dim=256, action_dim=1, sigma_init=0.01):
        super(ActionDecoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(2*features_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, action_dim),
            nn.Tanh()) #[1,1]
        self.sigma = nn.Parameter(sigma_init*torch.ones(action_dim, requires_grad=True, dtype=torch.float32), requires_grad=True)

    def forward(self, last_encoded_state, encoded_state):
        x = torch.cat((last_encoded_state, encoded_state),dim=1)
        mu = self.linear(x)
        covar = torch.diag(self.sigma)#.unsqueeze(dim=0)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu,covar)
        action_est = dist.sample()
        return action_est

class StatePredictor(nn.Module):
    def __init__(self, latent_dim=256, features_dim=256, action_dim=1):
        super(StatePredictor, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(features_dim+action_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, features_dim),
            nn.ELU())

    def forward(self, last_encoded_state, action):
        action = action.detach().view(-1,1)
        x = torch.cat((last_encoded_state,action),dim=1)
        x = self.linear(x)
        return x


class Agent(nn.Module):
    def __init__(self, vec_features_dim=8, features_dim=256, latent_dim=256, action_dim=1, sigma_init=0.01):
        super(Agent, self).__init__()
        
        self.encoder = StateEncoder(vec_features_dim=vec_features_dim, features_dim=features_dim)
        self.latent = nn.Sequential(nn.Linear(features_dim,latent_dim), nn.ELU())
        
        self.mu = nn.Sequential(nn.Linear(latent_dim,action_dim), nn.Tanh())
        self.sigma = nn.Parameter(sigma_init*torch.ones(action_dim, requires_grad=True, dtype=torch.float32), requires_grad=True)
        
        self.value = nn.Linear(latent_dim,1)

    def get_value(self, obs):
        x = self.encoder(*obs)
        x = self.latent(x)
        return self.value(x)

    def get_action_and_value(self, obs, action=None):
        x = self.encoder(*obs)
        x = self.latent(x)
        mu = self.mu(x)
        covar = torch.diag(self.sigma)#.unsqueeze(dim=0)
        dist = torch.distributions.normal.Normal(mu,covar)
        if action is None:
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(x)
        return action, action_logprob, entropy, value


class ICM(nn.Module):
    def __init__(self, vec_features_dim=8, features_dim=256, latent_dim=256, action_dim=1, sigma_init=0.01):
        super(ICM, self).__init__()
        
        self.encoder = StateEncoder(vec_features_dim=vec_features_dim, features_dim=features_dim)
        self.decoder = ActionDecoder(features_dim=features_dim, latent_dim=latent_dim, action_dim=action_dim, sigma_init=sigma_init)
        self.predictor = StatePredictor(latent_dim=latent_dim, features_dim=features_dim, action_dim=action_dim)

    def forward(self, last_state, state, action):
        last_phi = self.encoder(*last_state)
        phi = self.encoder(*state)
        phi_est = self.predictor(last_phi, action)
        action_est = self.decoder(last_phi, phi)
        return phi, phi_est, action_est

class Network(nn.Module):
    def __init__(self, features_dim=256):
        super(Network, self).__init__()

        self.agent = Agent(features_dim=features_dim)
        self.icm = ICM(features_dim=features_dim)

    def get_value(self, obs):
        return self.agent.get_value(obs)

    def get_action_and_value(self, obs, action=None):
        return self.agent.get_action_and_value(obs, action)

    def get_intrinsic_reward(self, last_state, state, action):
        phi, phi_est, _ = self.icm(last_state, state, action)
        intrinsic_reward = 0.5 * torch.linalg.norm(phi_est - phi, dim=-1).pow(2) #* phi.size()[1]
        return intrinsic_reward.detach().cpu().numpy()

    def get_icm_loss(self, last_state, state, action, beta=0.2):
        phi, phi_est, action_est = self.icm(last_state, state, action)
        forward_loss = 0.5 * torch.linalg.norm(phi_est - phi.detach(), dim=-1).pow(2)
        inverse_loss = 0.5 * torch.linalg.norm(action_est - action.detach(), dim=-1).pow(2)
        return beta * forward_loss + (1 - beta) * inverse_loss


if __name__ == "__main__":
    from env import icm_env
    import util
    import numpy as np
    cfg_file = "cfg.yaml"
    env = icm_env(headless=True,record=True,cfg=cfg_file)
    obs = env.reset()

    device = "cpu"

    obs = util.dict2torch(obs, device)

    net = Network().to(device)

    act_dict = {}
    for i in range(len(obs)):
        actions, logprobs, _, values = net.get_action_and_value(obs[i])
        act_dict[env.possible_agents[i]] = actions.flatten().numpy()

    print(act_dict)
    next_obs, rewards, terms, truncs, infos = env.step(act_dict)






