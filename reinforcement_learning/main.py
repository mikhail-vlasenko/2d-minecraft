import os

import ray
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.tune import Tuner
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from reinforcement_learning.config import CONFIG, ENV_KWARGS, WANDB_KWARGS
from reinforcement_learning.metrics_callback import MinecraftMetricsCallback
from reinforcement_learning.model.ray_rl_module import CustomPPORLModule


os.environ["RAY_DEDUP_LOGS"] = "0"


def env_creator(env_config):
    env = Minecraft2dEnv(**env_config)
    if CONFIG.env.simplified_action_space:
        return ActionSimplificationWrapper(env)
    return env


register_env("Minecraft2D", env_creator)


def main():
    ray.init()

    train_batch_size = CONFIG.train.iter_env_steps * CONFIG.env.num_envs * max(1, CONFIG.train.num_runners) // 2
    print(f"Train batch size: {train_batch_size}")

    ppo_config = (
        PPOConfig()
        .environment("Minecraft2D", env_config=ENV_KWARGS)
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CustomPPORLModule,
            ),
        )
        .training(
            lr=CONFIG.ppo.lr,
            gamma=CONFIG.ppo.gamma,
            lambda_=0.95,
            entropy_coeff=CONFIG.ppo.ent_coef,
            num_epochs=CONFIG.ppo.update_epochs,
            minibatch_size=CONFIG.ppo.batch_size,
            train_batch_size=train_batch_size,
        )
        .learners(
            num_gpus_per_learner=1,
            num_learners=0,
        )
        .env_runners(
            num_env_runners=CONFIG.train.num_runners,
            num_envs_per_env_runner=CONFIG.env.num_envs,
            num_cpus_per_env_runner=CONFIG.train.cpus_per_runner,
            num_gpus_per_env_runner=CONFIG.train.gpus_per_runner,
        )
        .callbacks(MinecraftMetricsCallback)
    )

    stop_conditions = {}
    if CONFIG.train.time_total_s:
        stop_conditions["time_total_s"] = CONFIG.train.time_total_s
    else:
        stop_conditions["num_env_steps_sampled_lifetime"] = CONFIG.train.env_steps

    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_at_end=True,
        checkpoint_frequency=CONFIG.train.checkpoint_frequency,
    )

    run_config = RunConfig(
        "minecraft_ppo",
        storage_path=CONFIG.storage_path,
        stop=stop_conditions,
        checkpoint_config=checkpoint_config,
        callbacks=[
            WandbLoggerCallback(
                log_config=True,
                **WANDB_KWARGS
            )
        ],
    )

    tuner = Tuner(
        "PPO",
        param_space=ppo_config.get_state(),
        run_config=run_config,
    )

    if CONFIG.train.load_checkpoint:
        tuner = Tuner.restore(
            path=CONFIG.train.load_checkpoint,
            trainable="PPO",
            param_space=ppo_config.get_state(),
            resume_errored=True,
        )
        print(f"\nCheckpoint loaded from: {CONFIG.train.load_checkpoint}\n")

    results = tuner.fit()

    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max"
    )

    if best_result and best_result.checkpoint:
        if CONFIG.train.save_to:
            best_result.checkpoint.to_directory(CONFIG.train.save_to)
            print(f"Best checkpoint saved to: {CONFIG.train.save_to}")

    ray.shutdown()


if __name__ == "__main__":
    main()
