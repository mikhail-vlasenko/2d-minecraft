import random
import torch
from tqdm import tqdm
import wandb
import numpy as np

from config import CONFIG
from python_wrapper.env_manager import EnvManager
from ppo import PPO, TrajectoryDataset, update_policy, device


def preprocess_observation(obs):
    top_materials_flat = obs.top_materials.flatten()
    tile_heights_flat = obs.tile_heights.flatten()
    player_pos_flat = np.array(obs.player_pos)
    player_rot = np.array([obs.player_rot])
    hp = np.array([obs.hp])
    time = np.array([obs.time])
    inventory_state = np.array(obs.inventory_state)
    mobs_flat = np.array(obs.mobs).flatten()

    return np.concatenate(
        [top_materials_flat, tile_heights_flat, player_pos_flat, player_rot, hp, time, inventory_state, mobs_flat]
    )


def main(logging_start_step=0):
    env_manager = EnvManager(CONFIG.env.lib_path, CONFIG.env.batch_size)

    sample_obs = preprocess_observation(env_manager.reset()[0])
    obs_shape = sample_obs.shape
    print('Observation shape:', obs_shape)
    n_actions = env_manager.num_actions

    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)

    if CONFIG.ppo_train.load_from is not None:
        print('Loading PPO from:', CONFIG.ppo_train.load_from)
        ppo.load_state_dict(torch.load(CONFIG.ppo_train.load_from))

    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)
    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=env_manager.batch_size)

    observations = env_manager.reset()
    states = torch.tensor(np.array([preprocess_observation(obs) for obs in observations])).float().to(device)

    step = 0
    for t in tqdm(range(int(CONFIG.ppo_train.env_steps / env_manager.batch_size))):
        actions, log_probs = ppo.act(states)
        observations, rewards, dones = env_manager.step(actions)

        train_ready = dataset.write_tuple(states, actions, rewards, dones, log_probs)

        states = torch.tensor(np.array([preprocess_observation(obs) for obs in observations])).float().to(device)

        if train_ready:
            step = t * env_manager.batch_size + logging_start_step
            wandb.log({'Returns': dataset.log_returns().mean(), 'Lengths': dataset.log_lengths().mean()}, step=step)

            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)

            dataset.reset_trajectories()

    torch.save(ppo.state_dict(), CONFIG.ppo_train.save_to)
    print(f'Saved PPO to: {CONFIG.ppo_train.save_to}')
    model_art = wandb.Artifact('ppo_model', type='model')
    model_art.add_file(CONFIG.ppo_train.save_to)
    wandb.log_artifact(model_art)

    env_manager.close()
    return step


if __name__ == '__main__':
    wandb.init(entity="mvlasenko", project='minecraft-rl', config=CONFIG.as_dict())
    main()
