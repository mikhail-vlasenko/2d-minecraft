import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from python_wrapper.env_manager import Minecraft2dEnv
from reinforcement_learning.config import CONFIG


def main():
    env = Minecraft2dEnv(num_envs=1, lib_path=CONFIG.env.lib_path, record_replays=CONFIG.evaluation.record_replays)

    model = PPO.load(CONFIG.ppo_train.save_to, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=CONFIG.evaluation.n_games, deterministic=False)

    print(f'Mean Evaluation Reward: {mean_reward}')
    print(f'Std Evaluation Reward: {std_reward}')

    env.close()


if __name__ == '__main__':
    main()
