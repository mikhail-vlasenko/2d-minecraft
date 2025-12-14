import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import wandb

from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from python_wrapper.stale_score_wrapper import StaleScoreWrapper
from python_wrapper.past_actions_wrapper import PastActionsWrapper
from reinforcement_learning.config import CONFIG, ENV_KWARGS, WANDB_KWARGS
from reinforcement_learning.model.feature_extractor import FeatureExtractor
from reinforcement_learning.model.sb3_policy import CustomActorCriticPolicy


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
        self.num_env_steps_sampled = 0

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
                self.final_times.append(self.locals["infos"][i]['time'])
                self.game_scores.append(self.locals["infos"][i]['game_score'])
                if CONFIG.env.discovered_actions_reward:
                    self.num_discovered_actions.append(self.locals["infos"][i]['discovered_actions'].sum())

                self.num_env_steps_sampled += self.episode_lengths[i]
                # Reset the counters for completed episodes
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

        return True

    def _on_rollout_end(self):
        if self.total_rewards:
            metrics = {
                'env_runners/custom_metrics/episode return_mean': np.mean(self.total_rewards),
                'env_runners/custom_metrics/episode length_mean': np.mean(self.total_lengths),
                'env_runners/custom_metrics/game time_mean': np.mean(self.final_times),
                'env_runners/custom_metrics/game score_mean': np.mean(self.game_scores),
                'env_runners/num_episodes': len(self.total_rewards),
                'num_env_steps_sampled': self.num_env_steps_sampled,
            }
            if CONFIG.env.discovered_actions_reward:
                metrics['env_runners/custom_metrics/num discovered actions_mean'] = np.mean(self.num_discovered_actions)
                metrics['env_runners/custom_metrics/num discovered actions_max'] = np.max(self.num_discovered_actions)

            print(metrics)
            wandb.log(metrics)

            # Reset the lists for the next set of episodes
            self.total_rewards = []
            self.total_lengths = []
            self.final_times = []
            self.num_discovered_actions = []
            self.game_scores = []


def apply_wrappers(env):
    """Apply wrappers in order: past actions (innermost), simplified actions, stale score (outermost)."""
    if isinstance(env, Monitor):
        # strip out the Monitor wrapper cause i dont think i need it
        env = env.env
    if CONFIG.env.use_past_actions:
        env = PastActionsWrapper(env)
    if CONFIG.env.simplified_action_space:
        env = ActionSimplificationWrapper(env)
    env = StaleScoreWrapper(env)
    return env


def main():
    run = wandb.init(**WANDB_KWARGS)

    env = make_vec_env(Minecraft2dEnv, n_envs=CONFIG.env.num_envs, env_kwargs=ENV_KWARGS, wrapper_class=apply_wrappers)

    if CONFIG.model.residual:
        policy = CustomActorCriticPolicy
        policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
        )
    else:
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            net_arch=dict(pi=CONFIG.model.dimensions, vf=CONFIG.model.dimensions),
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs=dict(),
        )

    # Always create model with CONFIG hyperparameters
    model = PPO(policy, env, policy_kwargs=policy_kwargs,
                verbose=0, tensorboard_log=f"runs/{run.id}",
                learning_rate=CONFIG.ppo.lr,
                n_steps=CONFIG.train.iter_env_steps,
                batch_size=CONFIG.ppo.batch_size,
                ent_coef=CONFIG.ppo.ent_coef,
                n_epochs=CONFIG.ppo.update_epochs, gamma=CONFIG.ppo.gamma, gae_lambda=0.95)

    # Load weights and optimizer state from checkpoint, but keep CONFIG hyperparameters
    weights_path = CONFIG.train.load_from
    if weights_path is not None:
        print(f"Loading network weights from {weights_path}")
        model.set_parameters(weights_path, exact_match=True)

    print(model.policy)

    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG.train.env_steps // CONFIG.train.checkpoints_per_training // CONFIG.env.num_envs,
        save_path="./reinforcement_learning/saved_models/"
    )

    callback_list = CallbackList([checkpoint_callback, LoggingCallback()])

    try:
        model.learn(
            total_timesteps=CONFIG.train.env_steps,
            callback=callback_list,
        )
        model.save(CONFIG.train.save_to)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model anyway.")
        model.save(CONFIG.train.fall_back_save_to)

    model_art = wandb.Artifact('sb3_ppo_checkpoint', type='model')
    # saves the last checkpoint by the callback
    model_art.add_file(f"./reinforcement_learning/saved_models/rl_model_{CONFIG.train.env_steps}_steps.zip")
    run.log_artifact(model_art)

    env.close()
    run.finish()


if __name__ == '__main__':
    main()
