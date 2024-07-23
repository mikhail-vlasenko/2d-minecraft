import random
import torch
import numpy as np

from tqdm import tqdm
from config import CONFIG
from python_wrapper.env_manager import EnvManager
from ppo import PPO, device
from reinforcement_learning.main import preprocess_observation, tensor_observations, load_model


def evaluate_policy(ppo, env_manager, n_games=2):
    total_rewards = []
    observations = env_manager.reset()
    state = tensor_observations(observations)

    for _ in range(n_games):
        done = False
        total_reward = 0
        while not done:
            actions, _ = ppo.act(state)
            observations, rewards, dones = env_manager.step(actions)
            total_reward += rewards[0]
            done = dones[0]
            if done:
                # get the new start state
                observations, _, _ = env_manager.step(actions)
                state = tensor_observations(observations)

        print(f'Game reward: {total_reward}')
        total_rewards.append(total_reward)

    return np.mean(total_rewards)


def main():
    env_manager = EnvManager(CONFIG.env.lib_path, batch_size=1, record_replays=CONFIG.evaluation.record_replays)

    observations = env_manager.reset()
    sample_obs = preprocess_observation(observations[0])
    ppo = PPO(state_shape=sample_obs.shape[0], n_actions=env_manager.num_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)

    if CONFIG.ppo_train.load_from is not None:
        load_model(ppo, optimizer, CONFIG.ppo_train.load_from)

    ppo.eval()
    eval_reward = evaluate_policy(ppo, env_manager, CONFIG.evaluation.n_games)
    print(f'Mean Evaluation Reward: {eval_reward}')

    env_manager.close()


if __name__ == '__main__':
    main()
