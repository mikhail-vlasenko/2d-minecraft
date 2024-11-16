import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional

import torch


@dataclass
class EnvConfig:
    num_envs: int = 16
    lib_path: str = 'C:/Users/Mikhail/RustProjects/2d-minecraft/target/release/ffi.dll'
    discovered_actions_reward: float = 25.
    include_actions_in_obs: bool = True
    observation_distance: int = 3
    max_observable_mobs: int = 8
    start_loadout: str = 'random'
    simplified_action_space: bool = True


@dataclass
class TrainConfig:
    env_steps: int = 1280000
    time_total_s: Optional[int] = 3600 * 3  # if None, then env_steps is used
    iter_env_steps: int = 256
    load_from: str = None
    # load_from: str = "reinforcement_learning/saved_models/sb3_ppo_interrupted.pt"
    # load_checkpoint: str = None
    load_checkpoint: str = "reinforcement_learning/ray_results/saved_models/IMPALA2"
    save_to: str = f'reinforcement_learning/ray_results/saved_models/IMPALA3'
    fall_back_save_to: str = f'reinforcement_learning/saved_models/sb3_ppo_interrupted.pt'
    checkpoints_per_training: int = 16
    checkpoint_frequency: int = 32  # in training iterations
    num_runners: int = 8
    cpus_per_runner: int = 1
    gpus_per_runner: float = 0


@dataclass
class EvaluationConfig:
    n_games: int = 3
    record_replays: bool = True


@dataclass
class ModelConfig:
    nonlinear: str = 'tanh'
    extractor_dim: int = 256
    dimensions: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    update_epochs: int = 5
    ent_coef: float = 0.01
    batch_size: int = 512
    rollout_fragment_length: Union[int, str] = 'auto'


@dataclass
class IMPALAConfig:
    gamma: float = 0.995
    rollout_fragment_length: int = 256
    ent_coef: float = 0.01


@dataclass
class Config:
    storage_path: str = f"{os.getcwd()}/reinforcement_learning/ray_results"
    wandb_resume_id: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    impala: IMPALAConfig = field(default_factory=IMPALAConfig)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()

ENV_KWARGS = {
    "observation_distance": CONFIG.env.observation_distance,
    "max_observable_mobs": CONFIG.env.max_observable_mobs,
    "discovered_actions_reward": CONFIG.env.discovered_actions_reward,
    "include_actions_in_obs": CONFIG.env.include_actions_in_obs,
    "start_loadout": CONFIG.env.start_loadout,
    "lib_path": CONFIG.env.lib_path,
    "num_total_envs": CONFIG.env.num_envs,
    "record_replays": False,
}

WANDB_KWARGS = {
    'project': 'minecraft-rl',
    'entity': 'mvlasenko',
    'config': CONFIG.as_dict()
}
if CONFIG.wandb_resume_id:
    WANDB_KWARGS['resume'] = "must"
    WANDB_KWARGS['id'] = CONFIG.wandb_resume_id
