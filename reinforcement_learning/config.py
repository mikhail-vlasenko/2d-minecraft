import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from python_wrapper.checkpoint_handler import CheckpointHandler
from python_wrapper.ffi_elements import MAX_MOBS


@dataclass
class EnvConfig:
    num_envs: int = 32
    lib_path: str = 'C:/Users/Mikhail/RustProjects/2d-minecraft/target/release/ffi.dll'
    discovered_actions_reward: float = 25.
    observation_distance: int = 8
    max_observable_mobs: int = MAX_MOBS
    start_loadout: str = 'random'
    checkpoint_starts: float = 0.75
    simplified_action_space: bool = True
    use_past_actions: bool = True
    normalize_reward: bool = False
    reward_norm_gamma: float = 0.999
    seed: Optional[int] = None  # todo


@dataclass
class TrainConfig:
    env_steps: int = 64_000_000
    time_total_s: Optional[int] = None  # if None, then env_steps is used
    iter_env_steps: int = 256
    load_from: Optional[str] = None
    load_checkpoint: Optional[str] = None
    save_to: str = f'reinforcement_learning/saved_models/sb3_ppo.pt'
    fall_back_save_to: str = f'reinforcement_learning/saved_models/sb3_ppo_interrupted.pt'
    checkpoint_dir: str = './reinforcement_learning/saved_models/'
    checkpoints_per_training: int = 16
    checkpoint_frequency: int = 64  # in training iterations
    num_runners: int = 2
    cpus_per_runner: int = 4
    gpus_per_runner: float = 0


@dataclass
class EvaluationConfig:
    n_games: int = 5
    record_replays: bool = True
    milestone_checkpoint: Optional[str] = None


@dataclass
class ModelConfig:
    nonlinear: str = 'tanh'
    residual_main_dim: int = 1024
    residual_hidden_dim: int = 1536
    residual_num_blocks: int = 4


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    update_epochs: int = 2
    ent_coef: float = 0.01
    batch_size: int = 512
    rollout_fragment_length: Union[int, str] = 'auto'


@dataclass
class WandbConfig:
    project: str = 'minecraft-rl'
    entity: str = 'mvlasenko'


@dataclass
class Config:
    storage_path: str = f"{os.getcwd()}/reinforcement_learning/ray_results"
    wandb_resume_id: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    @property
    def device(self) -> torch.device:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


def config_from_hydra(cfg: DictConfig) -> Config:
    """Convert a Hydra DictConfig to our Config dataclass."""
    # Convert OmegaConf to plain dict, then instantiate dataclasses
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    return Config(
        storage_path=cfg_dict.get('storage_path', Config.storage_path),
        wandb_resume_id=cfg_dict.get('wandb_resume_id', ''),
        env=EnvConfig(**cfg_dict.get('env', {})),
        train=TrainConfig(**cfg_dict.get('train', {})),
        evaluation=EvaluationConfig(**cfg_dict.get('evaluation', {})),
        model=ModelConfig(**cfg_dict.get('model', {})),
        ppo=PPOConfig(**cfg_dict.get('ppo', {})),
        wandb=WandbConfig(**cfg_dict.get('wandb', {})),
    )


def make_env_kwargs(config: Config, checkpoint_handler: CheckpointHandler) -> dict:
    """Create environment kwargs from config."""
    return {
        "observation_distance": config.env.observation_distance,
        "max_observable_mobs": config.env.max_observable_mobs,
        "discovered_actions_reward": config.env.discovered_actions_reward,
        "start_loadout": config.env.start_loadout,
        "checkpoint_starts": config.env.checkpoint_starts,
        "checkpoint_handler": checkpoint_handler,
        "lib_path": config.env.lib_path,
        "num_total_envs": config.env.num_envs,
        "record_replays": False,
    }


def make_wandb_kwargs(config: Config) -> dict:
    """Create wandb kwargs from config."""
    kwargs = {
        'project': config.wandb.project,
        'entity': config.wandb.entity,
        'config': config.as_dict()
    }
    if config.wandb_resume_id:
        kwargs['resume'] = "must"
        kwargs['id'] = config.wandb_resume_id
    return kwargs


# Default checkpoint handler
checkpoint_handler = CheckpointHandler(
    max_checkpoints=8, initial_checkpoints=[]
)
