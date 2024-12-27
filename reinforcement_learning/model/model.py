from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from python_wrapper.observation import NUM_MATERIALS, MOB_INFO_SIZE, LOOT_INFO_SIZE
from reinforcement_learning.model.attentive_pooler import AttentivePooler


class FeatureExtractor(nn.Module):
    def __init__(self, observation_space):
        super(FeatureExtractor, self).__init__()

        # todo: center crop the height and material maps
        self.material_channels = 8
        self.height_channels = 8
        self.mob_dim = 16
        self.loot_dim = 8

        self.block_encoder = nn.Sequential(
            nn.Linear(NUM_MATERIALS, self.material_channels),
            nn.ReLU(),
        )
        self.height_conv = nn.Sequential(
            nn.Conv2d(1, self.height_channels, kernel_size=3, stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
        )
        self.mob_encoder = nn.Sequential(
            nn.Linear(MOB_INFO_SIZE, self.mob_dim),
            nn.ReLU(),
        )
        self.loot_encoder = nn.Sequential(
            nn.Linear(LOOT_INFO_SIZE, self.loot_dim),
            nn.ReLU(),
        )
        self.mob_pooler = nn.Sequential(
            AttentivePooler(embed_dim=self.mob_dim, num_heads=1, complete_block=False),  # todo: heads
            nn.ReLU(),
        )
        self.loot_pooler = nn.Sequential(
            AttentivePooler(embed_dim=self.loot_dim, num_heads=1, complete_block=False),
            nn.ReLU(),
        )

        sample_observation = gym.vector.utils.batch_space(observation_space, 2).sample()
        for key, value in sample_observation.items():
            sample_observation[key] = torch.tensor(value).float()

        self.features_dim = self.forward(sample_observation).size(1)
        print(f"Extractor features_dim: {self.features_dim}")

    def forward(self, observation: dict) -> torch.Tensor:
        materials = observation["top_materials"]
        materials = materials.view(materials.size(0), -1)  # (batch_size, 17x17)
        materials = torch.eye(NUM_MATERIALS).to(materials.device)[materials.int()]  # (batch_size, 17x17, num_materials)
        materials = self.block_encoder(materials)  # (batch_size, 17x17, material_channels)
        materials = materials.view(materials.size(0), -1)

        height_map = observation["tile_heights"].unsqueeze(1)
        height_features = self.height_conv(height_map)
        height_features = height_features.view(height_features.size(0), -1)

        mobs = self.mob_encoder(observation["mobs"])
        mob_pool = self.mob_pooler(mobs)
        mob_pool = mob_pool.view(mob_pool.size(0), -1)

        loots = self.loot_encoder(observation["loot"])
        loot_pool = self.loot_pooler(loots)
        loot_pool = loot_pool.view(loot_pool.size(0), -1)

        # todo: time modulo 2

        return torch.cat([materials, height_features, mob_pool, loot_pool,
                          self.extract_flat_features(observation)], dim=1)

    def extract_flat_features(self, observation: dict) -> torch.Tensor:
        player_pos = observation["player_pos"]
        # sb3 one-hot encodes discrete spaces
        player_rot = observation["player_rot"]
        hp = observation["hp"]
        time = observation["time"]
        inventory = observation["inventory_state"]
        action_mask = observation["action_mask"]
        return torch.cat([player_pos, player_rot, hp, time, inventory, action_mask], dim=1)
