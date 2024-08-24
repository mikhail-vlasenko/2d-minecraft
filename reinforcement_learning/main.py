import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.rllib.algorithms import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from reinforcement_learning.config import CONFIG
from reinforcement_learning.metrics_callback import MinecraftMetricsCallback


def env_creator(env_config):
    return Minecraft2dEnv(**env_config)


register_env("Minecraft2D", env_creator)


def main():
    ray.init()

    wandb_kwargs = {
        'entity': 'mvlasenko',
        'project': "minecraft-rl",
        'config': CONFIG.as_dict()
    }
    if CONFIG.wandb_resume_id:
        wandb_kwargs['resume'] = "must"
        wandb_kwargs['id'] = CONFIG.wandb_resume_id

    train_batch_size = CONFIG.train.iter_env_steps * CONFIG.env.num_envs * max(1, CONFIG.train.num_runners) // 2

    impala_config = (
        ImpalaConfig()
        .environment("Minecraft2D", env_config={
            "discovered_actions_reward": CONFIG.env.discovered_actions_reward,
            "include_actions_in_obs": CONFIG.env.include_actions_in_obs,
            "lib_path": CONFIG.env.lib_path,
            "num_total_envs": CONFIG.env.num_envs,
            "record_replays": False,
        })
        .framework("torch")
        .training(
            # model={
            #     "fcnet_hiddens": CONFIG.ppo.dimensions.insert(CONFIG.ppo.extractor_dim, 0),
            #     "fcnet_activation": CONFIG.ppo.nonlinear,
            # },
            gamma=CONFIG.ppo.gamma,
            vf_loss_coeff=0.5,
            # entropy_coeff=CONFIG.ppo.ent_coef,
            train_batch_size=train_batch_size,
            vtrace=True,
            vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
        )
        .resources(num_gpus=1, num_cpus_for_main_process=2)
        .env_runners(
            rollout_fragment_length=CONFIG.impala.rollout_fragment_length,
            num_env_runners=CONFIG.train.num_runners,
            num_envs_per_env_runner=CONFIG.env.num_envs,
            num_cpus_per_env_runner=1,
        )
        .callbacks(MinecraftMetricsCallback)
    )

    stop_conditions = {
        "timesteps_total": CONFIG.train.env_steps,
    }

    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_at_end=True,
        checkpoint_frequency=CONFIG.train.env_steps // train_batch_size // CONFIG.train.checkpoints_per_training
    )

    analysis = tune.run(
        "IMPALA",
        storage_path=CONFIG.storage_path,
        config=impala_config.to_dict(),
        stop=stop_conditions,
        checkpoint_config=checkpoint_config,
        callbacks=[
            WandbLoggerCallback(
                log_config=True,
                **wandb_kwargs
            )
        ],
    )

    best_trial = analysis.get_best_trial("env_runners/episode_return_mean", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final episode reward mean: {best_trial.last_result['env_runners/episode_return_mean']}")

    best_checkpoint = analysis.best_checkpoint
    if CONFIG.train.save_to:
        best_checkpoint.to_directory(CONFIG.train.save_to)
        print(f"Best checkpoint saved to: {CONFIG.train.save_to}")

    ray.shutdown()


if __name__ == "__main__":
    main()
