import glob
import os
import ray
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
import numpy as np
from tqdm import tqdm

from python_wrapper.minecraft_2d_env import Minecraft2dEnv, initialize_minecraft_connection
from reinforcement_learning.config import CONFIG, ENV_KWARGS
from reinforcement_learning.main import env_creator


def evaluate_model(checkpoint_path, num_episodes):
    register_env("Minecraft2D", env_creator)

    ray.init()

    algo = Algorithm.from_checkpoint(checkpoint_path)

    print("Reinitializing lib connection "
          "because checkpoint loading makes its envs in the training configuration automatically.")
    initialize_minecraft_connection(num_envs=1, lib_path=CONFIG.env.lib_path, record_replays=True)

    ENV_KWARGS["num_total_envs"] = 1
    ENV_KWARGS["record_replays"] = CONFIG.evaluation.record_replays

    env = env_creator(ENV_KWARGS)

    episode_rewards = []
    episode_lengths = []
    final_times = []
    game_scores = []
    discovered_actions = []

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_times.append(info['time'])
        game_scores.append(info['game_score'])
        if CONFIG.env.discovered_actions_reward:
            discovered_actions.append(info['discovered_actions'].sum())

    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average final time: {np.mean(final_times):.2f} ± {np.std(final_times):.2f}")
    print(f"Average game score: {np.mean(game_scores):.2f} ± {np.std(game_scores):.2f}")
    if CONFIG.env.discovered_actions_reward:
        print(f"Average discovered actions: {np.mean(discovered_actions):.2f} ± {np.std(discovered_actions):.2f}")
        print(f"Max discovered actions: {np.max(discovered_actions)}")

    # Clean up
    env.close()
    ray.shutdown()


if __name__ == "__main__":
    # Find the latest experiment directory
    experiment_dirs = glob.glob(os.path.join(CONFIG.storage_path, "PPO_*"))
    latest_experiment = max(experiment_dirs, key=os.path.getmtime)

    # Find the latest trial directory within the latest experiment
    trial_dirs = glob.glob(os.path.join(latest_experiment, "PPO_Minecraft2D_*"))
    latest_trial = max(trial_dirs, key=os.path.getmtime)

    # Find the latest checkpoint file
    checkpoint_files = glob.glob(os.path.join(latest_trial, "checkpoint_*"))
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    if CONFIG.train.load_checkpoint:
        latest_checkpoint = CONFIG.train.load_checkpoint

    print(f"Evaluating checkpoint: {latest_checkpoint}")
    evaluate_model(latest_checkpoint, num_episodes=CONFIG.evaluation.n_games)
