import os
import numpy as np
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import wandb

from python_wrapper.checkpoint_handler import CheckpointHandler
from python_wrapper.minecraft_2d_env import Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from python_wrapper.stale_score_wrapper import StaleScoreWrapper
from python_wrapper.past_actions_wrapper import PastActionsWrapper
from reinforcement_learning.config import Config, config_from_hydra, make_env_kwargs, make_wandb_kwargs
from reinforcement_learning.model.feature_extractor import FeatureExtractor
from reinforcement_learning.model.sb3_policy import CustomActorCriticPolicy


class LoggingCallback(BaseCallback):
    def __init__(self, config: Config, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.config = config
        self.episode_rewards = np.zeros(config.env.num_envs)
        self.episode_lengths = np.zeros(config.env.num_envs, dtype=int)
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
                if self.config.env.discovered_actions_reward:
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
            if self.config.env.discovered_actions_reward:
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


def make_wrapper_fn(config: Config):
    """Create wrapper function with config closure."""
    def apply_wrappers(env):
        """Apply wrappers in order: past actions (innermost), simplified actions, stale score (outermost)."""
        if isinstance(env, Monitor):
            # strip out the Monitor wrapper cause i dont think i need it
            env = env.env
        if config.env.use_past_actions:
            env = PastActionsWrapper(env)
        if config.env.simplified_action_space:
            env = ActionSimplificationWrapper(env)
        env = StaleScoreWrapper(env)
        return env
    return apply_wrappers


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to our Config dataclass
    config = config_from_hydra(cfg)
    
    # Create checkpoint handler
    checkpoint_handler = CheckpointHandler(max_checkpoints=8, initial_checkpoints=[])
    
    # Create env kwargs and wandb kwargs
    env_kwargs = make_env_kwargs(config, checkpoint_handler)
    wandb_kwargs = make_wandb_kwargs(config)

    run = wandb.init(**wandb_kwargs)

    env = make_vec_env(
        Minecraft2dEnv, 
        n_envs=config.env.num_envs, 
        env_kwargs=env_kwargs, 
        wrapper_class=make_wrapper_fn(config)
    )

    policy = CustomActorCriticPolicy
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs={'config': config},
        model_config=config.model,
    )

    # Always create model with config hyperparameters
    model = PPO(policy, env, policy_kwargs=policy_kwargs,
                verbose=0, tensorboard_log=f"runs/{run.id}",
                learning_rate=config.ppo.lr,
                n_steps=config.train.iter_env_steps,
                batch_size=config.ppo.batch_size,
                ent_coef=config.ppo.ent_coef,
                n_epochs=config.ppo.update_epochs, gamma=config.ppo.gamma, gae_lambda=0.95)

    # Load weights and optimizer state from checkpoint, but keep config hyperparameters
    weights_path = config.train.load_from
    if weights_path is not None:
        print(f"Loading network weights from {weights_path}")
        model.set_parameters(weights_path, exact_match=True)

    print(model.policy)

    checkpoint_callback = CheckpointCallback(
        save_freq=config.train.env_steps // config.train.checkpoints_per_training // config.env.num_envs,
        save_path=config.train.checkpoint_dir,
    )

    callback_list = CallbackList([checkpoint_callback, LoggingCallback(config)])

    try:
        model.learn(
            total_timesteps=config.train.env_steps,
            callback=callback_list,
        )
        model.save(config.train.save_to)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model anyway.")
        model.save(config.train.fall_back_save_to)

    final_model_path = os.path.join(config.train.checkpoint_dir, f"final_model_{run.id}.zip")
    model.save(final_model_path)

    model_art = wandb.Artifact('sb3_ppo_checkpoint', type='model')
    model_art.add_file(final_model_path)
    run.log_artifact(model_art)

    env.close()
    run.finish()


if __name__ == '__main__':
    main()
