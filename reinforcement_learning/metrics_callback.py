import numpy as np

import gymnasium as gym
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID, EpisodeType
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from typing import Dict, Union, Optional

from reinforcement_learning.config import CONFIG


class MinecraftMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
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
        episode.custom_metrics["episode return"] = episode.total_reward
        episode.custom_metrics["episode length"] = episode.length
        last_info = episode.last_info_for()
        episode.custom_metrics["game time"] = last_info["time"]
        episode.custom_metrics["game score"] = last_info["game_score"]
        if CONFIG.env.discovered_actions_reward:
            episode.custom_metrics["num discovered actions"] = np.sum(last_info['discovered_actions'])
