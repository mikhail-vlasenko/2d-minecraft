from dataclasses import dataclass, field, asdict
from typing import List

import torch


@dataclass
class EnvConfig:
    batch_size: int = 32
    lib_path: str = './target/release/ffi.dll'

@dataclass
class PPOTrainConfig:
    env_steps: int = 10000000
    load_from: str = None
    save_to: str = f'reinforcement_learning/saved_models/just_ppo.pt'


@dataclass
class PPOConfig:
    batch_size: int = 4096
    n_workers: int = 64
    lr: float = 1e-3 / 2
    entropy_reg: float = 0.05
    gamma: float = 0.99
    epsilon: float = 0.1
    update_epochs: int = 5
    nonlinear: str = 'relu'  # tanh, relu
    dimensions: List[int] = field(default_factory=lambda: [265, 128])
    simple_architecture: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # if True, will train AIRL with curriculum in experiment_setup. Otherwise, will train ppo
    curriculum_for_airl: bool = True
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()
