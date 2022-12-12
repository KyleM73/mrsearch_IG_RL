from mrsearch_IG_RL.external import *

class IdentityExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        features_dim = int(np.product([observation_space.shape[i] for i in range(len(observation_space.shape))]))
        super().__init__(observation_space, features_dim)
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations

class Encoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Encoder, self).__init__()
        n_input_dim = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_dim, 32, kernel_size=11, stride=5, padding=0), #201 -> 39
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=6, stride=3, padding=0), #39 -> 12
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0), #12 -> 5
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), #5 -> 3
            nn.ELU(),
            nn.Flatten(), #[-1,288]
            #nn.Unflatten(0,(1,-1)), #[1,288]
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            self.n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        #self.linear = nn.Sequential(nn.Linear(self.n_flatten, features_dim), nn.ELU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        if x.size()[0] != observations.size()[0] or x.size()[1] != self.n_flatten:
            x = nn.Flatten(0)(x)
            x = nn.Unflatten(0,(observations.size()[0],-1))(x) 
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_state_dim: int = 288, features_dim: int = 256, action_dim: int = 1):
        super(Decoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(encoded_state_dim+encoded_state_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, action_dim),
            nn.Tanh() #[-1,1]
            )

    def forward(self, encoded_state: torch.Tensor, last_encoded_state: torch.Tensor):
        x = torch.cat((last_encoded_state,encoded_state),dim=1)
        x = self.linear(x)
        return x

class StatePredictor(nn.Module):
    def __init__(self, encoded_state_dim: int = 288, features_dim: int = 256, action_dim: int = 1):
        super(StatePredictor, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(encoded_state_dim+action_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, encoded_state_dim)
            )

    def forward(self, encoded_state: torch.Tensor, action: torch.Tensor):
        action = action.detach()
        x = torch.cat((encoded_state,action),dim=1)
        x = self.linear(x)
        return x

class ActionPolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space, last_layer_dim: int = 256, action_dim: int = 1):
        super(ActionPolicy, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim
        self.latent_dim_vf = last_layer_dim

        self.encoder = Encoder(observation_space)
        features_dim = self.encoder.n_flatten

        # in ICM paper this was an LSTM, could try frame stacking
        self.linear = nn.Sequential(nn.Linear(features_dim, last_layer_dim), nn.ELU())

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(features)
        x = self.linear(x)
        return x, x

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        x = self.encoder(features)
        x = self.linear(x)
        return x

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        x = self.encoder(features)
        x = self.linear(x)
        return x

class ICMPolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space):
        super(ICMPolicy, self).__init__()

        self.encoder = Encoder(observation_space)
        self.decoder = Decoder()
        self.statepredictor = StatePredictor()

    def forward(self, state: torch.Tensor, last_state: torch.Tensor, action: torch.Tensor):
        encoded_state = self.encoder(state)
        last_encoded_state = self.encoder(last_state)

        predicted_action = self.decoder(encoded_state,last_encoded_state)
        predicted_encoded_state = self.statepredictor(last_encoded_state,action)

        pred_err_action = predicted_action-action
        Li = torch.sum(torch.square(pred_err_action),dim=-1)
        pred_err_encoding = predicted_encoded_state-encoded_state
        Lf = 0.5*torch.sum(torch.square(pred_err_encoding),dim=-1)

        return torch.cat((Li,Lf),dim=0)

class ActorCriticICM(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float] = lambda t: 1e-3,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ELU,
        normalize_images: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = IdentityExtractor,
        use_sde: bool = True,
        *args,
        **kwargs,
        ):

        super(ActorCriticICM, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            normalize_images,
            use_sde,
            # Pass remaining arguments to base class
            *args,
            features_extractor_class=features_extractor_class,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.last_obs = None

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActionPolicy(self.observation_space)
        self.icm_extractor = ICMPolicy(self.observation_space)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        #assert torch.equal(obs,features)
        latent_pi, latent_vf = self.mlp_extractor(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi) 
        actions = distribution.get_actions(deterministic=deterministic)
        actions = nn.Tanh()(actions) #[-1,1]
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        if self.last_obs is None:
            self.last_obs = obs
        intrinsic_rewards = self.icm_extractor(obs,self.last_obs,actions) #(Li,Lf)
        intrinsic_rewards = intrinsic_rewards.reshape((obs.size()[0],)+(-1,))
        self.last_obs = obs
        return actions, values, log_prob, intrinsic_rewards

if __name__ == "__main__":
    import numpy as np
    from gym import spaces
    obs_space = spaces.Box(low=-1,high=1,shape=(1,201,201),dtype=np.float32)
    act_space = spaces.Box(low=-1,high=1,shape=(1,),dtype=np.float32)

    pi = ActorCriticICM(obs_space,act_space)
    obs = torch.as_tensor(obs_space.sample()[None]).float()
    obs4d = torch.cat((obs,obs,obs,obs),dim=0)
    out = pi(obs4d)
    print(out)



    """
        s1 = torch.as_tensor(obs_space.sample()[None]).float()
        s2 = torch.as_tensor(obs_space.sample()[None]).float()

        print(s1.size())
        print(s2.size())

        encoder = Encoder(obs_space)
        phis1 = encoder(s1)
        phis2 = encoder(s2)

        print(phis1.size())
        print(phis2.size())

        decoder = Decoder()
        a_hat = decoder(phis1,phis2)

        print(a_hat.size())

        statepredictor = StatePredictor()
        phis2_hat = statepredictor(phis1,a_hat)

        print(phis2_hat.size())
    """
    """
    self.action_dist is set

    self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
    mean_actions = self.action_net(latent_pi)
        int(np.prod(action_space.shape))

        if isinstance(action_space, spaces.Box):
            cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
            return cls(get_action_dim(action_space), **dist_kwargs)

        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        dist_kwargs = {
                    "full_std": full_std,
                    "squash_output": squash_output,
                    "use_expln": use_expln,
                    "learn_features": False,
                }

    """
    # action space:
    # [th, Ri, Li, Lf]
    # jk
    # have forward return 4 terms
    # see ICM_PPO
    # also reward terms input in env.step