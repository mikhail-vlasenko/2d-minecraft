from dataclasses import dataclass, field, asdict
from typing import List

import torch


@dataclass
class EnvConfig:
    num_envs: int = 16
    lib_path: str = 'C:/Users/Mikhail/RustProjects/2d-minecraft/target/release/ffi.dll'
    discovered_actions_reward: float = 50.
    include_actions_in_obs: bool = True


@dataclass
class PPOTrainConfig:
    env_steps: int = 64000000
    iter_env_steps: int = 1024
    load_from: str = None
    # load_from: str = f'reinforcement_learning/saved_models/rl_model_64000000_steps.zip'
    # load_from: str = f'reinforcement_learning/saved_models/sb3_ppo.pt'
    save_to: str = f'reinforcement_learning/saved_models/sb3_ppo.pt'
    fall_back_save_to: str = f'reinforcement_learning/saved_models/unfinished_run.pt'
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
    ent_coef: float = 0.01
    batch_size: int = 256
    nonlinear: str = 'tanh'
    dimensions: List[int] = field(default_factory=lambda: [512, 256, 128, 64])


@dataclass
class Config:
    wandb_resume_id: str = ""
    num_runners: int = 1
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()
