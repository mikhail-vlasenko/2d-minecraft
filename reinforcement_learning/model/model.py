from typing import Tuple, List

import torch
import torch.nn as nn
from gym.vector.utils import spaces

from python_wrapper.observation import NUM_MATERIALS, MOB_INFO_SIZE, LOOT_INFO_SIZE, ProcessedObservation
from reinforcement_learning.model.attentive_pooler import AttentivePooler


class FeatureExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Box):
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

        self.features_dim = self.forward([ProcessedObservation.default()])[0].shape[0]

    def forward(self, observation: List[ProcessedObservation]) -> torch.Tensor:
        if not observation:
            print(observation)
            return torch.zeros(0, self.features_dim).to(next(self.parameters()).device)
        print(len(observation))
        materials = torch.stack([torch.Tensor(obs.top_materials) for obs in observation])
        materials = materials.view(materials.size(0), -1)  # (batch_size, 17x17)
        materials = torch.eye(NUM_MATERIALS)[materials.long()]  # (batch_size, 17x17, num_materials)
        materials = self.block_encoder(materials)  # (batch_size, 17x17, material_channels)
        materials = materials.view(materials.size(0), -1)

        height_map = torch.stack([torch.Tensor(obs.tile_heights) for obs in observation])
        height_map = height_map.unsqueeze(1)
        height_features = self.height_conv(height_map)
        height_features = height_features.view(height_features.size(0), -1)

        mobs = torch.stack([torch.Tensor(obs.mobs) for obs in observation])
        mobs = self.mob_encoder(mobs)
        mob_pool = self.mob_pooler(mobs)
        mob_pool = mob_pool.view(mob_pool.size(0), -1)

        loots = torch.stack([torch.Tensor(obs.loot) for obs in observation])
        loots = self.loot_encoder(loots)
        loot_pool = self.loot_pooler(loots)
        loot_pool = loot_pool.view(loot_pool.size(0), -1)

        return torch.cat([materials, height_features, mob_pool, loot_pool,
                          self.extract_flat_features(observation)], dim=1)

    @staticmethod
    def extract_flat_features(observation: List[ProcessedObservation]) -> torch.Tensor:
        player_pos = torch.stack([torch.Tensor(obs.player_pos) for obs in observation])
        player_rot = torch.stack([torch.Tensor([obs.player_rot]) for obs in observation])
        hp = torch.stack([torch.Tensor([obs.hp]) for obs in observation])
        time = torch.stack([torch.Tensor([obs.time]) for obs in observation])
        inventory = torch.stack([torch.Tensor(obs.inventory_state) for obs in observation])
        action_mask = torch.stack([torch.Tensor(obs.action_mask) for obs in observation])
        return torch.cat([player_pos, player_rot, hp, time, inventory, action_mask], dim=1)


class MainModel(nn.Module):
    """
    Applied after the feature extractor to predict the value function or policy.
    """
    def __init__(self, feature_extractor: FeatureExtractor, dimensions, is_policy: bool = True):
        super(MainModel, self).__init__()

        self.fc1 = nn.Linear(8 + 8 + 16 + 8, 64)
        self.fc2 = nn.Linear(64, 1)
