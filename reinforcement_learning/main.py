import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from reinforcement_learning.config import CONFIG
from reinforcement_learning.metrics_callback import MinecraftMetricsCallback


def env_creator(env_config):
    return Minecraft2dEnv(
        discovered_actions_reward=env_config["discovered_actions_reward"],
        include_actions_in_obs=env_config["include_actions_in_obs"],
        lib_path=CONFIG.env.lib_path,
        record_replays=False,
        num_total_envs=CONFIG.env.num_envs,
    )


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

    ppo_config = (
        PPOConfig()
        .environment("Minecraft2D", env_config={
            "discovered_actions_reward": CONFIG.env.discovered_actions_reward,
            "include_actions_in_obs": CONFIG.env.include_actions_in_obs,
        })
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": CONFIG.ppo.dimensions,
                "fcnet_activation": CONFIG.ppo.nonlinear,
            },
            lr=CONFIG.ppo.lr,
            gamma=CONFIG.ppo.gamma,
            lambda_=0.95,
            entropy_coeff=CONFIG.ppo.ent_coef,
            num_sgd_iter=CONFIG.ppo.update_epochs,
            sgd_minibatch_size=CONFIG.ppo.batch_size,
            train_batch_size=CONFIG.ppo_train.iter_env_steps * CONFIG.env.num_envs,
        )
        .rollouts(num_rollout_workers=CONFIG.num_runners)
        .resources(num_gpus=1)
        .env_runners(num_cpus_per_env_runner=1)
        .callbacks(MinecraftMetricsCallback)
    )

    if CONFIG.ppo_train.load_from:
        ppo_config = ppo_config.restore(CONFIG.ppo_train.load_from)

    stop_conditions = {
        "timesteps_total": CONFIG.ppo_train.env_steps,
    }

    analysis = tune.run(
        "PPO",
        config=ppo_config.to_dict(),
        stop=stop_conditions,
        checkpoint_freq=CONFIG.ppo_train.save_every // CONFIG.env.num_envs,
        checkpoint_at_end=True,
        callbacks=[
            WandbLoggerCallback(
                log_config=True,
                **wandb_kwargs
            )
        ],
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final episode reward mean: {best_trial.last_result['episode_reward_mean']}")

    best_checkpoint = analysis.best_checkpoint
    if CONFIG.ppo_train.save_to:
        best_checkpoint.to_directory(CONFIG.ppo_train.save_to)
        print(f"Best checkpoint saved to: {CONFIG.ppo_train.save_to}")

    ray.shutdown()


if __name__ == "__main__":
    main()
