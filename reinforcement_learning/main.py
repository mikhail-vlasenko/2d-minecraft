import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.rllib.algorithms import ImpalaConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train import Checkpoint
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from reinforcement_learning.config import CONFIG, ENV_KWARGS, WANDB_KWARGS
from reinforcement_learning.metrics_callback import MinecraftMetricsCallback


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

    impala_config = (
        ImpalaConfig()
        .environment("Minecraft2D", env_config=ENV_KWARGS)
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": [CONFIG.model.extractor_dim] + CONFIG.model.dimensions,
                "fcnet_activation": CONFIG.model.nonlinear,
            },
            gamma=CONFIG.impala.gamma,
            vf_loss_coeff=CONFIG.impala.vf_loss_coeff,
            entropy_coeff=CONFIG.impala.ent_coef,
            train_batch_size=train_batch_size,
            vtrace=True,
            vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
            replay_proportion=CONFIG.impala.replay_proportion,
            replay_buffer_num_slots=CONFIG.impala.replay_buffer_num_slots,
        )
        .resources(
            num_gpus=(1 - CONFIG.train.num_runners * CONFIG.train.gpus_per_runner),
            num_cpus_for_main_process=2
        )
        .env_runners(
            rollout_fragment_length=CONFIG.impala.rollout_fragment_length,
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
        stop_conditions["timesteps_total"] = CONFIG.train.env_steps

    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_at_end=True,
        checkpoint_frequency=CONFIG.train.checkpoint_frequency,
    )

    checkpoint = None
    if CONFIG.train.load_checkpoint:
        checkpoint = Checkpoint.from_directory(CONFIG.train.load_checkpoint).to_directory()
        print(f"\nCheckpoint loaded from: {CONFIG.train.load_checkpoint}\n")

    results = tune.run(
        "IMPALA",
        restore=checkpoint,
        storage_path=CONFIG.storage_path,
        config=impala_config.to_dict(),
        stop=stop_conditions,
        checkpoint_config=checkpoint_config,
        callbacks=[
            WandbLoggerCallback(
                log_config=True,
                **WANDB_KWARGS
            )
        ],
    )

    best_checkpoint = results.get_best_checkpoint(results.trials[0], "env_runners/episode_return_mean", "max")
    if CONFIG.train.save_to:
        best_checkpoint.to_directory(CONFIG.train.save_to)
        print(f"Best checkpoint saved to: {CONFIG.train.save_to}")

    ray.shutdown()


if __name__ == "__main__":
    main()
