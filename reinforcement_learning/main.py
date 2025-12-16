import os

import hydra
from omegaconf import DictConfig
import ray
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.tune import Tuner
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from python_wrapper.checkpoint_handler import CheckpointHandler
from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from reinforcement_learning.config import Config, config_from_hydra, make_env_kwargs, make_wandb_kwargs
from reinforcement_learning.metrics_callback import MinecraftMetricsCallback
from reinforcement_learning.model.ray_rl_module import CustomPPORLModule


os.environ["RAY_DEDUP_LOGS"] = "0"


def make_env_creator(config: Config, env_kwargs: dict):
    """Create env_creator function with config closure."""
    def env_creator(env_config):
        env = Minecraft2dEnv(**env_config)
        if config.env.simplified_action_space:
            return ActionSimplificationWrapper(env)
        return env
    return env_creator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to our Config dataclass
    config = config_from_hydra(cfg)
    
    # Create checkpoint handler
    checkpoint_handler = CheckpointHandler(max_checkpoints=8, initial_checkpoints=[])
    
    # Create env kwargs and wandb kwargs
    env_kwargs = make_env_kwargs(config, checkpoint_handler)
    wandb_kwargs = make_wandb_kwargs(config)
    
    # Register the environment
    env_creator = make_env_creator(config, env_kwargs)
    register_env("Minecraft2D", env_creator)

    ray.init()

    train_batch_size = config.train.iter_env_steps * config.env.num_envs * max(1, config.train.num_runners) // 2
    print(f"Train batch size: {train_batch_size}")

    ppo_config = (
        PPOConfig()
        .environment("Minecraft2D", env_config=env_kwargs)
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CustomPPORLModule,
            ),
        )
        .training(
            lr=config.ppo.lr,
            gamma=config.ppo.gamma,
            lambda_=0.95,
            entropy_coeff=config.ppo.ent_coef,
            num_epochs=config.ppo.update_epochs,
            minibatch_size=config.ppo.batch_size,
            train_batch_size=train_batch_size,
        )
        .learners(
            num_gpus_per_learner=1,
            num_learners=0,
        )
        .env_runners(
            num_env_runners=config.train.num_runners,
            num_envs_per_env_runner=config.env.num_envs,
            num_cpus_per_env_runner=config.train.cpus_per_runner,
            num_gpus_per_env_runner=config.train.gpus_per_runner,
        )
        .callbacks(MinecraftMetricsCallback)
    )

    stop_conditions = {}
    if config.train.time_total_s:
        stop_conditions["time_total_s"] = config.train.time_total_s
    else:
        stop_conditions["num_env_steps_sampled_lifetime"] = config.train.env_steps

    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_at_end=True,
        checkpoint_frequency=config.train.checkpoint_frequency,
    )

    run_config = RunConfig(
        "minecraft_ppo",
        storage_path=config.storage_path,
        stop=stop_conditions,
        checkpoint_config=checkpoint_config,
        callbacks=[
            WandbLoggerCallback(
                log_config=True,
                **wandb_kwargs
            )
        ],
    )

    tuner = Tuner(
        "PPO",
        param_space=ppo_config.get_state(),
        run_config=run_config,
    )

    if config.train.load_checkpoint:
        tuner = Tuner.restore(
            path=config.train.load_checkpoint,
            trainable="PPO",
            param_space=ppo_config.get_state(),
            resume_errored=True,
        )
        print(f"\nCheckpoint loaded from: {config.train.load_checkpoint}\n")

    results = tuner.fit()

    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max"
    )

    if best_result and best_result.checkpoint:
        if config.train.save_to:
            best_result.checkpoint.to_directory(config.train.save_to)
            print(f"Best checkpoint saved to: {config.train.save_to}")

    ray.shutdown()


if __name__ == "__main__":
    main()
