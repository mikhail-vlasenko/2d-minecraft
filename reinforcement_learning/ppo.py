import pickle
import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from config import CONFIG


device = CONFIG.device


class PPO(nn.Module):
    def __init__(self, state_shape, in_channels=6, n_actions=9):
        super(PPO, self).__init__()

        self.state_shape = state_shape
        self.simple_architecture = CONFIG.ppo.simple_architecture

        if self.simple_architecture:
            self.l1 = nn.Linear(state_shape, CONFIG.ppo.dimensions[0])
            self.l2 = nn.Linear(CONFIG.ppo.dimensions[0], CONFIG.ppo.dimensions[1])
            self.actor_out = nn.Linear(CONFIG.ppo.dimensions[1], n_actions)
            self.critic_out = nn.Linear(CONFIG.ppo.dimensions[1], 1)
        else:
            self.in_channels = in_channels

            # Network Layers
            self.l1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=2)
            self.l2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2)
            self.actor_l3 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=2)
            self.critic_l3 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=2)
            self.actor_out = nn.Linear(32*(state_shape[0]-3)*(state_shape[1]-3), n_actions)
            self.critic_out = nn.Linear(32*(state_shape[0]-3)*(state_shape[1]-3), 1)

        if CONFIG.ppo.nonlinear == 'tanh':
            self.non_linear = nn.Tanh()
        elif CONFIG.ppo.nonlinear == 'relu':
            self.non_linear = nn.ReLU()
        else:
            raise ValueError(f'Non-linear function {CONFIG.ppo.nonlinear} is not supported')
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        if self.simple_architecture:
            x = self.non_linear(self.l1(x))
            x = self.non_linear(self.l2(x))
            x_actor = self.softmax(self.actor_out(x))
            x_critic = self.critic_out(x)

            return x_actor, x_critic

        x = x.view(-1, self.in_channels, self.state_shape[0], self.state_shape[1])
        x = self.non_linear(self.l1(x))
        x = self.non_linear(self.l2(x))
        x_actor = self.non_linear(self.actor_l3(x))
        x_actor = x_actor.view(x_actor.shape[0], -1)
        x_critic = self.non_linear(self.critic_l3(x))
        x_critic = x_critic.view(x_critic.shape[0], -1)
        x_actor = self.softmax(self.actor_out(x_actor))
        x_critic = self.critic_out(x_critic)

        return x_actor, x_critic

    def act(self, state) -> (np.ndarray, np.ndarray):
        action_probabilities, _ = self.forward(state)
        m = Categorical(action_probabilities)
        action = m.sample()
        return action.detach().cpu().numpy(), m.log_prob(action).detach().cpu().numpy()

    def evaluate_trajectory(self, tau):
        trajectory_states = torch.stack(tau['states']).to(device)
        trajectory_actions = torch.tensor(tau['actions']).to(device)
        action_probabilities, critic_values = self.forward(trajectory_states)
        dist = Categorical(action_probabilities)
        action_entropy = dist.entropy().mean()
        action_log_probabilities = dist.log_prob(trajectory_actions)

        return action_log_probabilities, torch.squeeze(critic_values), action_entropy


class TrajectoryDataset:
    def __init__(self, batch_size, n_workers):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.trajectories: List[Dict] = []
        self.buffer = [{'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'latents': None, 'logs': []}
                       for _ in range(n_workers)]
        self.step_count = 0

    def reset_buffer(self, i):
        self.buffer[i] = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'latents': None, 'logs': []}

    def reset_trajectories(self):
        self.trajectories = []
        self.step_count = 0

    def write_tuple(self, states, actions, rewards, done, log_probs, logs=None) -> bool:
        # Takes states of shape (n_workers, state_shape[0], state_shape[1])
        for i in range(self.n_workers):
            self.buffer[i]['states'].append(states[i])
            self.buffer[i]['actions'].append(actions[i])
            self.buffer[i]['rewards'].append(rewards[i])
            self.buffer[i]['log_probs'].append(log_probs[i])

            if logs is not None:
                self.buffer[i]['logs'].append(logs[i])

            if done[i]:
                self.trajectories.append(self.buffer[i].copy())
                self.step_count += len(self.buffer[i]['actions'])
                self.reset_buffer(i)

        return self.step_count >= self.batch_size

    def log_returns(self):
        # Calculates (undiscounted) returns in self.trajectories
        returns = [0 for i in range(len(self.trajectories))]
        for i, tau in enumerate(self.trajectories):
            returns[i] = sum(tau['rewards'])
        return np.array(returns)

    def log_objectives(self):
        # Calculates achieved objectives in self.trajectories
        objective_logs = []
        for i, tau in enumerate(self.trajectories):
            objective_logs.append(sum(tau['logs']))

        return np.array(objective_logs)

    def log_lengths(self):
        lengths = []
        for tau in self.trajectories:
            lengths.append(len(tau['rewards']))

        return np.array(lengths)


def g_clip(epsilon, A):
    return torch.tensor([1 + epsilon if i else 1 - epsilon for i in A >= 0]).to(device) * A


def update_policy(ppo: PPO, dataset: TrajectoryDataset, optimizer, gamma, epsilon, n_epochs, entropy_reg) -> None:
    for epoch in range(n_epochs):
        batch_loss = 0
        value_loss = 0
        for i, tau in enumerate(dataset.trajectories):
            reward_togo = 0
            returns = []
            normalized_reward = np.array(tau['rewards'])
            normalized_reward = (normalized_reward - normalized_reward.mean())/(normalized_reward.std()+1e-5)
            for r in normalized_reward[::-1]:
                # Compute rewards-to-go and advantage estimates
                reward_togo = r + gamma * reward_togo
                returns.insert(0, reward_togo)
            action_log_probabilities, critic_values, action_entropy = ppo.evaluate_trajectory(tau)
            advantages = torch.tensor(returns).to(device) - critic_values.detach().to(device)
            likelihood_ratios = torch.exp(action_log_probabilities - torch.tensor(tau['log_probs']).detach().to(device))
            clipped_losses = -torch.min(likelihood_ratios * advantages, g_clip(epsilon, advantages))
            batch_loss += torch.mean(clipped_losses) - entropy_reg * action_entropy
            value_loss += torch.mean((torch.tensor(returns).to(device) - critic_values) ** 2)
        overall_loss = (batch_loss + value_loss) / dataset.batch_size
        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

