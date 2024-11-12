import datetime
import os
import pathlib
from dataclasses import dataclass, field, asdict
from typing import List, Union

import torch


@dataclass
class EnvConfig:
    num_envs: int = 1
    lib_path: str = 'C:/Users/Mikhail/RustProjects/2d-minecraft/target/release/ffi.dll'
    discovered_actions_reward: float = 50.
    include_actions_in_obs: bool = True
    observation_distance: int = 3
    max_observable_mobs: int = 8
    start_loadout: str = 'random'
    simplified_action_space: bool = True


@dataclass
class TrainConfig:
    env_steps: int = 128000000
    iter_env_steps: int = 1024
    # load_from: str = None
    load_from: str = "reinforcement_learning/saved_models/sb3_ppo_interrupted.pt"
    load_checkpoint: str = None
    # load_checkpoint: str = "reinforcement_learning/saved_models/rl_model_88000000_steps.zip"
    save_to: str = f'reinforcement_learning/saved_models/sb3_ppo.pt'
    fall_back_save_to: str = f'reinforcement_learning/saved_models/sb3_ppo_interrupted.pt'
    checkpoints_per_training: int = 16
    num_runners: int = 8


@dataclass
class EvaluationConfig:
    n_games: int = 3
    record_replays: bool = True


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    update_epochs: int = 5
    ent_coef: float = 0.01
    batch_size: int = 512
    rollout_fragment_length: Union[int, str] = 'auto'
    nonlinear: str = 'tanh'
    extractor_dim: int = 256
    dimensions: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class IMPALAConfig:
    gamma: float = 0.995
    rollout_fragment_length: int = 256


@dataclass
class MuZeroConfig:
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

    seed = 0  # Seed for numpy, torch and the game
    max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game (set automatically)
    observation_shape = None  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    action_space = None  # Fixed list of all possible actions. You should only edit the length
    players = list(range(1))  # List of players. You should only edit the length
    stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    num_workers = 7  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    selfplay_on_gpu = True
    max_moves = 2000  # Maximum number of moves if game is not finished before
    num_simulations = 16  # Number of future moves self-simulated
    discount = 0.997  # Chronological discount of the reward
    temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    root_dirichlet_alpha = 0.25
    root_exploration_fraction = 0.25

    # UCB formula
    pb_c_base = 19652
    pb_c_init = 1.25

    ### Network
    network = "fullyconnected"  # "resnet" / "fullyconnected"
    support_size = 50  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

    # Residual Network
    downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    blocks = 1  # Number of blocks in the ResNet
    channels = 2  # Number of channels in the ResNet
    reduced_channels_reward = 2  # Number of channels in reward head
    reduced_channels_value = 2  # Number of channels in value head
    reduced_channels_policy = 2  # Number of channels in policy head
    resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
    resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
    resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    encoding_size = 16
    fc_representation_layers = [128]  # Define the hidden layers in the representation network
    fc_dynamics_layers = [64, 64]  # Define the hidden layers in the dynamics network
    fc_reward_layers = [64]  # Define the hidden layers in the reward network
    fc_value_layers = [64]  # Define the hidden layers in the value network
    fc_policy_layers = [64, 64]  # Define the hidden layers in the policy network

    ### Training
    results_path = None  # Path to store the model weights and TensorBoard logs
    save_model = True  # Save the checkpoint in results_path as model.checkpoint
    training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
    batch_size = 128  # Number of parts of games to train on at each training step
    checkpoint_interval = 10  # Number of training steps before using the model for self-playing
    value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

    optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
    weight_decay = 1e-4  # L2 weights regularization
    momentum = 0.9  # Used only if optimizer is SGD

    # Exponential learning rate schedule
    lr_init = 0.01  # Initial learning rate
    lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
    lr_decay_steps = 1000

    ### Replay Buffer
    replay_buffer_size = 1000  # Number of self-play games to keep in the replay buffer
    num_unroll_steps = 64  # Number of game moves to keep for every batch element
    td_steps = 50  # Number of steps in the future to take into account for calculating the target value
    PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    # Reanalyze (See paper appendix Reanalyse)
    use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
    reanalyse_on_gpu = False

    ### Adjust the self play / training ratio to avoid over/underfitting
    self_play_delay = 0  # Number of seconds to wait after each played game
    training_delay = 0  # Number of seconds to wait after each training step
    ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        # if trained_steps < 0.5 * self.training_steps:
        #     return 1.0
        # elif trained_steps < 0.75 * self.training_steps:
        #     return 0.5
        # else:
        #     return 0.25
        return 0.7


@dataclass
class Config:
    storage_path: str = f"{os.getcwd()}/reinforcement_learning/ray_results"
    wandb_resume_id: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    impala: IMPALAConfig = field(default_factory=IMPALAConfig)
    muzero: MuZeroConfig = field(default_factory=MuZeroConfig)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()
