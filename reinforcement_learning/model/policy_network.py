from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from reinforcement_learning.config import CONFIG


class ResidualNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = CONFIG.model.residual_main_dim,
        last_layer_dim_vf: int = CONFIG.model.residual_main_dim,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        residual_kwargs = dict(
            main_dim=CONFIG.model.residual_main_dim,
            hidden_dim=CONFIG.model.residual_hidden_dim,
            num_residual_blocks=CONFIG.model.residual_num_blocks,
            activation=nn.Tanh
        )

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, CONFIG.model.residual_main_dim),
            nn.Tanh(),
            ResidualMLP(**residual_kwargs),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, CONFIG.model.residual_main_dim),
            nn.Tanh(),
            ResidualMLP(**residual_kwargs),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResidualNetwork(self.features_dim)


class ResidualMLP(nn.Module):
    def __init__(
            self,
            main_dim: int,
            hidden_dim: int,
            num_residual_blocks: int,
            activation: Type[nn.Module] = nn.Tanh
    ):
        super().__init__()

        self.activation = activation()
        layers = []
        for i in range(num_residual_blocks):
            layers.append(nn.Sequential(
                nn.Linear(main_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, main_dim),
            ))
        self.hidden_layers = nn.ModuleList(
            layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(x + layer(x))
        return x
