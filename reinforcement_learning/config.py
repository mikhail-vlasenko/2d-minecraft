import os
from dataclasses import dataclass, field, asdict
from typing import List, Union

import torch


@dataclass
class EnvConfig:
    num_envs: int = 8
    lib_path: str = 'C:/Users/Mikhail/RustProjects/2d-minecraft/target/release/ffi.dll'
    discovered_actions_reward: float = 100.
    include_actions_in_obs: bool = True


@dataclass
class TrainConfig:
    env_steps: int = 16000000
    iter_env_steps: int = 1024
    load_from: str = None
    save_to: str = f'reinforcement_learning/saved_models/sb3_ppo.pt'
    checkpoints_per_training: int = 16
    num_runners: int = 8


@dataclass
class EvaluationConfig:
    n_games: int = 2
    record_replays: bool = True


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    update_epochs: int = 10
    ent_coef: float = 0.01
    batch_size: int = 512
    rollout_fragment_length: Union[int, str] = 'auto'
    nonlinear: str = 'tanh'
    dimensions: List[int] = field(default_factory=lambda: [512, 256, 128, 64])


@dataclass
class IMPALAConfig:
    gamma: float = 0.995
    rollout_fragment_length: int = 256


@dataclass
class Config:
    storage_path: str = f"{os.getcwd()}/reinforcement_learning/ray_results"
    wandb_resume_id: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    impala: IMPALAConfig = field(default_factory=IMPALAConfig)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()
