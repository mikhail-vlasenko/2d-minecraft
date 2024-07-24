from dataclasses import dataclass, field, asdict
from typing import List

import torch


@dataclass
class EnvConfig:
    num_envs: int = 32
    lib_path: str = './target/release/ffi.dll'


@dataclass
class PPOTrainConfig:
    env_steps: int = 2000000
    load_from: str = None
    # load_from: str = f'reinforcement_learning/saved_models/one_hot_top_materials.pt'
    save_to: str = f'reinforcement_learning/saved_models/one_hot_top_materials.pt'
    save_every: int = env_steps // 10


@dataclass
class EvaluationConfig:
    n_games: int = 2
    record_replays: bool = True


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    update_epochs: int = 10
    nonlinear: str = 'tanh'  # tanh, relu
    dimensions: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()
