import numpy as np

import gymnasium as gym
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict, Union, Optional

from reinforcement_learning.config import CONFIG


class MinecraftMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_end(
        self,
        *,
        episode: Union[SingleAgentEpisode, MultiAgentEpisode],
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional[EnvRunner] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ):
        episode_return = episode.get_return()
        episode_length = len(episode)

        last_info = episode.get_infos(-1)

        metrics_logger.log_value("episode return", episode_return)
        metrics_logger.log_value("episode length", episode_length)
        metrics_logger.log_value("game time", last_info["time"])
        metrics_logger.log_value("game score", last_info["game_score"])

        if CONFIG.env.discovered_actions_reward:
            metrics_logger.log_value(
                "num discovered actions",
                np.sum(last_info["discovered_actions"]),
            )
