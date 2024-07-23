import random
import torch
from tqdm import tqdm
import wandb
import numpy as np

from config import CONFIG
from python_wrapper.env_manager import EnvManager
from ppo import PPO, TrajectoryDataset, update_policy, device


def preprocess_observation(obs):
    # One-hot encode top_materials
    top_materials_flat = obs.top_materials.flatten()
    top_materials_one_hot = np.eye(13)[top_materials_flat].flatten()

    tile_heights_flat = obs.tile_heights.flatten()
    player_pos_flat = np.array(obs.player_pos)
    player_rot = np.array([obs.player_rot])
    hp = np.array([obs.hp])
    time = np.array([obs.time])
    inventory_state = np.array(obs.inventory_state)
    mobs_flat = np.array(obs.mobs).flatten()

    return np.concatenate(
        [top_materials_one_hot, tile_heights_flat, player_pos_flat, player_rot, hp, time, inventory_state, mobs_flat]
    )


def save_model(ppo, optimizer, path):
    torch.save({
        'model_state_dict': ppo.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f'Saved PPO to: {CONFIG.ppo_train.save_to}')


def load_model(ppo, optimizer, path):
    checkpoint = torch.load(path)
    ppo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded PPO from: {path}')


def tensor_observations(observations):
    return torch.tensor(np.array([preprocess_observation(obs) for obs in observations])).float().to(device)


def main(logging_start_step=0):
    env_manager = EnvManager(CONFIG.env.lib_path, CONFIG.env.batch_size)

    observations = env_manager.reset()
    sample_obs = preprocess_observation(observations[0])
    obs_shape = sample_obs.shape
    print('Observation shape:', obs_shape)
    n_actions = env_manager.num_actions

    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)

    if CONFIG.ppo_train.load_from is not None:
        load_model(ppo, optimizer, CONFIG.ppo_train.load_from)

    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=env_manager.batch_size)

    states = tensor_observations(observations)

    step = 0
    num_saved = 1
    for t in tqdm(range(int(CONFIG.ppo_train.env_steps / env_manager.batch_size))):
        actions, log_probs = ppo.act(states)
        observations, rewards, dones = env_manager.step(actions)

        train_ready = dataset.write_tuple(states, actions, rewards, dones, log_probs)

        states = tensor_observations(observations)

        if train_ready:
            step = t * env_manager.batch_size + logging_start_step
            wandb.log({'Returns': dataset.log_returns().mean(), 'Lengths': dataset.log_lengths().mean()}, step=step)

            if step > num_saved * CONFIG.ppo_train.save_every:
                save_model(ppo, optimizer, CONFIG.ppo_train.save_to)
                num_saved += 1

            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)

            dataset.reset_trajectories()

    save_model(ppo, optimizer, CONFIG.ppo_train.save_to)
    model_art = wandb.Artifact('ppo_model_and_optim', type='model')
    model_art.add_file(CONFIG.ppo_train.save_to)
    wandb.log_artifact(model_art)

    env_manager.close()
    return step


if __name__ == '__main__':
    wandb.init(entity="mvlasenko", project='minecraft-rl', config=CONFIG.as_dict())
    main()
