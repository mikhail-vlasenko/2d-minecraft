import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
import wandb

from python_wrapper.env_manager import Minecraft2dEnv
from reinforcement_learning.config import CONFIG


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = np.zeros(CONFIG.env.num_envs)
        self.episode_lengths = np.zeros(CONFIG.env.num_envs, dtype=int)
        self.total_rewards = []
        self.total_lengths = []

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

                # Reset the counters for completed episodes
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

        return True

    def _on_rollout_end(self):
        # Calculate average reward and length of completed episodes
        if self.total_rewards:
            average_reward = np.mean(self.total_rewards)
            average_length = np.mean(self.total_lengths)
            print(f"Average episode reward: {average_reward:.2f} over {len(self.total_rewards)} episodes")
            print(f"Average episode length: {average_length:.2f}")
            wandb.log({
                'Average Reward': average_reward,
                'Average Length': average_length,
                'Episodes': len(self.total_rewards)
            })

            # Reset the lists for the next set of episodes
            self.total_rewards = []
            self.total_lengths = []


def main():
    run = wandb.init(entity="mvlasenko", project='minecraft-rl', config=CONFIG.as_dict())

    env = Minecraft2dEnv(num_envs=CONFIG.env.num_envs, lib_path=CONFIG.env.lib_path)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}",
                learning_rate=CONFIG.ppo.lr, n_steps=1024,
                n_epochs=CONFIG.ppo.update_epochs, gamma=CONFIG.ppo.gamma, gae_lambda=0.95)

    if CONFIG.ppo_train.load_from is not None:
        model.load(CONFIG.ppo_train.load_from)

    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG.ppo_train.save_every,
        save_path="./reinforcement_learning/saved_models/"
    )

    callback_list = CallbackList([checkpoint_callback, LoggingCallback()])

    model.learn(
        total_timesteps=CONFIG.ppo_train.env_steps,
        callback=callback_list,
        progress_bar=True,
    )

    model.save(CONFIG.ppo_train.save_to)
    # model_art = wandb.Artifact('sb3_ppo_model', type='model')
    # model_art.add_file(CONFIG.ppo_train.save_to)
    # run.log_artifact(model_art)

    env.close()
    run.finish()


if __name__ == '__main__':
    main()
