from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from python_wrapper.minecraft_2d_env import initialize_minecraft_connection, Minecraft2dEnv
from python_wrapper.simplified_actions import ActionSimplificationWrapper
from reinforcement_learning.config import CONFIG


def main():
    initialize_minecraft_connection(num_envs=1, lib_path=CONFIG.env.lib_path, record_replays=True)

    env_kwargs = {
        "observation_distance": CONFIG.env.observation_distance,
        "max_observable_mobs": CONFIG.env.max_observable_mobs,
        "discovered_actions_reward": CONFIG.env.discovered_actions_reward,
        "include_actions_in_obs": CONFIG.env.include_actions_in_obs,
        "start_loadout": CONFIG.env.start_loadout,
        "lib_path": CONFIG.env.lib_path,
        "num_total_envs": 1,
        "record_replays": True,
    }

    env = Minecraft2dEnv(**env_kwargs)

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
