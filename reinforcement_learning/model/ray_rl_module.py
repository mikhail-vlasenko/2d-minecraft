"""
Minimal Custom PPO RLModule for Ray RLlib.
Uses your existing ResidualNetwork and FeatureExtractor from SB3.
"""

from typing import Any, Dict, Optional, Type
import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

from reinforcement_learning.model.feature_extractor import FeatureExtractor
from reinforcement_learning.model.policy_network import ResidualNetwork


class CustomPPORLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.features_extractor = FeatureExtractor(self.observation_space)
        features_dim = self.features_extractor.features_dim

        self.mlp_extractor = ResidualNetwork(features_dim)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf

        # Action head
        if isinstance(self.action_space, spaces.Discrete):
            self.action_net = nn.Linear(latent_dim_pi, self.action_space.n)
        elif isinstance(self.action_space, spaces.Box):
            action_dim = int(np.prod(self.action_space.shape))
            self.action_net = nn.Linear(latent_dim_pi, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Value head
        self.value_net = nn.Linear(latent_dim_vf, 1)

    def _preprocess_obs(self, obs):
        if isinstance(obs, dict):
            return {k: v.float() for k, v in obs.items()}
        return obs.float()

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs = batch[Columns.OBS]
        obs = self._preprocess_obs(obs)
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)


        if isinstance(self.action_space, spaces.Discrete):
            action_dist_inputs = self.action_net(latent_pi)
        else:
            mean = self.action_net(latent_pi)
            action_dist_inputs = torch.cat([mean, self.log_std.expand_as(mean)], dim=-1)

        return {Columns.ACTION_DIST_INPUTS: action_dist_inputs}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> torch.Tensor:
        obs = batch[Columns.OBS]
        obs = self._preprocess_obs(obs)
        features = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf).squeeze(-1)
