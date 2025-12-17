from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from reinforcement_learning.model.policy_network import ResidualNetwork
from reinforcement_learning.config import ModelConfig


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, model_config: ModelConfig, **kwargs):
        self.model_config = model_config
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResidualNetwork(self.features_dim, self.model_config)
