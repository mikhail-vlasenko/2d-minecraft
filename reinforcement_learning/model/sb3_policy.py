from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from reinforcement_learning.model.policy_network import ResidualNetwork


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResidualNetwork(self.features_dim)
