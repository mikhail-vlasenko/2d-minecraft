import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
import wandb
from torch import nn

from python_wrapper.env_manager import Minecraft2dEnv
from python_wrapper.observation import OBSERVATION_GRID_SIZE
from reinforcement_learning.config import CONFIG


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = np.zeros(CONFIG.env.num_envs)
        self.episode_lengths = np.zeros(CONFIG.env.num_envs, dtype=int)
        self.total_rewards = []
        self.total_lengths = []
        self.final_times = []
        self.num_discovered_actions = []
        self.game_scores = []

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        # Update the rewards and lengths
        self.episode_rewards += rewards
        self.episode_lengths += 1

        # Check for completed episodes
        for i in range(len(dones)):
            if dones[i]:
                # Log completed episodes
                self.total_rewards.append(self.episode_rewards[i])
                self.total_lengths.append(self.episode_lengths[i])
                self.final_times.append(self.extract_time_from_observation(self.locals["obs_tensor"][i]))
                self.game_scores.append(self.locals["infos"][i]['game_score'])
                if CONFIG.env.discovered_actions_reward:
                    self.num_discovered_actions.append(self.locals["infos"][i]['discovered_actions'].sum())

                # Reset the counters for completed episodes
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

        return True

    def _on_rollout_end(self):
        # Calculate average reward and length of completed episodes
        if self.total_rewards:
            average_reward = np.mean(self.total_rewards)
            average_length = np.mean(self.total_lengths)
            average_max_time = np.mean(self.final_times)
            average_game_score = np.mean(self.game_scores)
            print(f"Over {len(self.total_rewards)} episodes:")
            print(f"Average episode reward: {average_reward:.2f}")
            print(f"Average episode length: {average_length:.2f}")
            print(f"Average episode time: {average_max_time:.2f}")
            print(f"Average game score: {average_game_score:.2f}")
            metrics = {
                'Average Reward': average_reward,
                'Average Length': average_length,
                'Average End Time': average_max_time,
                'Average Game Score': average_game_score,
                'Episodes': len(self.total_rewards)
            }
            if CONFIG.env.discovered_actions_reward:
                metrics['Discovered Actions'] = np.mean(self.num_discovered_actions)
                metrics['Max Discovered Actions'] = np.max(self.num_discovered_actions)
                print(f"Average number of discovered actions: {metrics['Discovered Actions']:.2f}, max: {metrics['Max Discovered Actions']:.2f}")
            wandb.log(metrics)

            # Reset the lists for the next set of episodes
            self.total_rewards = []
            self.total_lengths = []
            self.final_times = []
            self.num_discovered_actions = []
            self.game_scores = []

    @staticmethod
    def extract_time_from_observation(obs):
        # sum the sizes of previous arrays to get the index of the time
        idx = OBSERVATION_GRID_SIZE ** 2 * 13 + OBSERVATION_GRID_SIZE ** 2 + 3 + 1 + 1
        return obs[idx].item()


class CustomMLPFeatureExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int):
        in_feats = observation_space.shape[0]
        self.features_dim = features_dim
        super(CustomMLPFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_feats, self.features_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.feature_extractor(observations)


def main():
    wandb_kwargs = {
        'project': 'minecraft-rl',
        'entity': 'mvlasenko',
        'config': CONFIG.as_dict()
    }
    if CONFIG.wandb_resume_id:
        wandb_kwargs['resume'] = "must"
        wandb_kwargs['id'] = CONFIG.wandb_resume_id

    run = wandb.init(**wandb_kwargs)

    env = Minecraft2dEnv(
        num_envs=CONFIG.env.num_envs,
        lib_path=CONFIG.env.lib_path,
        discovered_actions_reward=CONFIG.env.discovered_actions_reward,
        include_actions_in_obs=CONFIG.env.include_actions_in_obs
    )

    policy_kwargs = dict(
        net_arch=dict(pi=CONFIG.ppo.dimensions, vf=CONFIG.ppo.dimensions),
        features_extractor_class=CustomMLPFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=CONFIG.ppo.extractor_dim),
    )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                verbose=0, tensorboard_log=f"runs/{run.id}",
                learning_rate=CONFIG.ppo.lr,
                n_steps=CONFIG.ppo_train.iter_env_steps,
                batch_size=CONFIG.ppo.batch_size,
                ent_coef=CONFIG.ppo.ent_coef,
                n_epochs=CONFIG.ppo.update_epochs, gamma=CONFIG.ppo.gamma, gae_lambda=0.95)

    if CONFIG.ppo_train.load_from is not None:
        print(f"Loading model from {CONFIG.ppo_train.load_from}")
        model = model.load(CONFIG.ppo_train.load_from, env)

    print(model.policy)

    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG.ppo_train.save_every // CONFIG.env.num_envs,
        save_path="./reinforcement_learning/saved_models/"
    )

    callback_list = CallbackList([checkpoint_callback, LoggingCallback()])

    try:
        model.learn(
            total_timesteps=CONFIG.ppo_train.env_steps,
            callback=callback_list,
        )
        model.save(CONFIG.ppo_train.save_to)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model anyway.")
        model.save(CONFIG.ppo_train.fall_back_save_to)

    model_art = wandb.Artifact('sb3_ppo_checkpoint', type='model')
    # saves the last checkpoint by the callback
    model_art.add_file(f"./reinforcement_learning/saved_models/rl_model_{CONFIG.ppo_train.env_steps}_steps.zip")
    run.log_artifact(model_art)

    env.close()
    run.finish()


if __name__ == '__main__':
    main()
