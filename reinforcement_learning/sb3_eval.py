import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from python_wrapper.checkpoint_handler import CheckpointHandler
from python_wrapper.minecraft_2d_env import initialize_minecraft_connection, Minecraft2dEnv
from reinforcement_learning.config import config_from_hydra, make_env_kwargs
from reinforcement_learning.sb3_ppo import make_wrapper_fn


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = config_from_hydra(cfg)
    
    initialize_minecraft_connection(num_envs=1, lib_path=config.env.lib_path, record_replays=True)

    # Create checkpoint handler with optional milestone
    if config.evaluation.milestone_checkpoint:
        checkpoint_handler = CheckpointHandler(
            max_checkpoints=8, 
            initial_checkpoints=[(1, config.evaluation.milestone_checkpoint)]
        )
        checkpoint_starts = 1.0
    else:
        checkpoint_handler = CheckpointHandler(max_checkpoints=8, initial_checkpoints=[])
        checkpoint_starts = 0.0  # every evaluation starts from scratch
    
    env_kwargs = make_env_kwargs(config, checkpoint_handler)
    env_kwargs["num_total_envs"] = 1
    env_kwargs["record_replays"] = config.evaluation.record_replays
    env_kwargs["checkpoint_starts"] = checkpoint_starts

    env = Minecraft2dEnv(**env_kwargs)
    env = make_wrapper_fn(config)(env)

    if config.train.load_checkpoint:
        model = PPO.load(config.train.load_checkpoint, env=env)
    else:
        model = PPO.load(config.train.load_from, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=config.evaluation.n_games, deterministic=False)

    print(f'Mean Evaluation Reward: {mean_reward}')
    print(f'Std Evaluation Reward: {std_reward}')

    env.close()


if __name__ == '__main__':
    main()
