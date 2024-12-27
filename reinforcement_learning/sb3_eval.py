from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from python_wrapper.checkpoint_handler import CheckpointHandler
from python_wrapper.minecraft_2d_env import initialize_minecraft_connection, Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from reinforcement_learning.config import CONFIG, ENV_KWARGS


def main():
    initialize_minecraft_connection(num_envs=1, lib_path=CONFIG.env.lib_path, record_replays=True)

    ENV_KWARGS["num_total_envs"] = 1
    ENV_KWARGS["record_replays"] = CONFIG.evaluation.record_replays
    ENV_KWARGS["checkpoint_starts"] = 0.  # every evaluation starts from scratch
    if CONFIG.evaluation.milestone_checkpoint:
        ENV_KWARGS["checkpoint_handler"] = CheckpointHandler(
            max_checkpoints=8, initial_checkpoints=[(1, CONFIG.evaluation.milestone_checkpoint)]
        )
        ENV_KWARGS["checkpoint_starts"] = 1.

    env = Minecraft2dEnv(**ENV_KWARGS)

    if CONFIG.env.simplified_action_space:
        env = ActionSimplificationWrapper(env)

    if CONFIG.train.load_checkpoint:
        model = PPO.load(CONFIG.train.load_checkpoint, env=env)
    else:
        model = PPO.load(CONFIG.train.load_from, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=CONFIG.evaluation.n_games, deterministic=False)

    print(f'Mean Evaluation Reward: {mean_reward}')
    print(f'Std Evaluation Reward: {std_reward}')

    env.close()


if __name__ == '__main__':
    main()
