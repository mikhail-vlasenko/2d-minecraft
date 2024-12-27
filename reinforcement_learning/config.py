import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional

import torch

from python_wrapper.checkpoint_handler import CheckpointHandler


@dataclass
class EnvConfig:
    num_envs: int = 32
    lib_path: str = 'C:/Users/Mikhail/RustProjects/2d-minecraft/target/release/ffi.dll'
    flatten_observation: bool = False
    discovered_actions_reward: float = 25.
    observation_distance: int = 3
    max_observable_mobs: int = 8
    start_loadout: str = 'random'
    checkpoint_starts: float = 0.75
    simplified_action_space: bool = True
    seed: Optional[int] = None  # todo


@dataclass
class TrainConfig:
    env_steps: int = 32_000_000
    time_total_s: Optional[int] = None  # if None, then env_steps is used
    iter_env_steps: int = 256
    load_from: str = None
    # load_from: str = "reinforcement_learning/saved_models/sb3_ppo_interrupted.pt"
    load_checkpoint: str = None
    # load_checkpoint: str = "reinforcement_learning/ray_results/saved_models/IMPALA3"
    save_to: str = f'reinforcement_learning/saved_models/sb3_ppo.pt'
    fall_back_save_to: str = f'reinforcement_learning/saved_models/sb3_ppo_interrupted.pt'
    checkpoints_per_training: int = 16
    checkpoint_frequency: int = 32  # in training iterations
    num_runners: int = 8
    cpus_per_runner: int = 1
    gpus_per_runner: float = 0


@dataclass
class EvaluationConfig:
    n_games: int = 5
    record_replays: bool = True
    milestone_checkpoint: str = None
    # milestone_checkpoint: str = "autosave_ms_2_score_167_2024-11-17_18-37-38"


@dataclass
class ModelConfig:
    nonlinear: str = 'tanh'
    dimensions: List[int] = field(default_factory=lambda: [256, 128, 64])


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
    vf_loss_coeff: float = 0.5
    replay_proportion: float = 3.0
    replay_buffer_num_slots: int = 64


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

# declare the global checkpoint handler to share game saves between the env instances
checkpoint_handler = CheckpointHandler(
    max_checkpoints=8, initial_checkpoints=[]
)
ENV_KWARGS = {
    "flatten_observation": CONFIG.env.flatten_observation,
    "observation_distance": CONFIG.env.observation_distance,
    "max_observable_mobs": CONFIG.env.max_observable_mobs,
    "discovered_actions_reward": CONFIG.env.discovered_actions_reward,
    "start_loadout": CONFIG.env.start_loadout,
    "checkpoint_starts": CONFIG.env.checkpoint_starts,
    "checkpoint_handler": checkpoint_handler,
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
