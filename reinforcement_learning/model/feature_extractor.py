import torch
import torch.nn as nn
import gymnasium as gym
from torch.nn import MaxPool2d

from python_wrapper.ffi_elements import INVENTORY_SIZE, NUM_ACTIONS
from python_wrapper.observation import NUM_MATERIALS, MOB_INFO_SIZE, LOOT_INFO_SIZE
from python_wrapper.past_actions_wrapper import PastActionsWrapper
from reinforcement_learning.config import Config
from reinforcement_learning.model.attentive_pooler import AttentivePooler


class FeatureExtractor(nn.Module):
    def __init__(self, observation_space, config: Config):
        super(FeatureExtractor, self).__init__()

        self.material_channels = 8
        self.height_channels = 8
        self.mob_heads = 16
        self.mob_dim = 16 * self.mob_heads
        self.loot_heads = 8
        self.loot_dim = 8 * self.loot_heads
        self.inventory_dim = 2 * INVENTORY_SIZE
        self.past_actions_dim = 16

        self.block_encoder = nn.Sequential(
            nn.Linear(NUM_MATERIALS, self.material_channels),
            nn.GELU(),
        )
        self.height_conv_hor = nn.Sequential(
            nn.Conv2d(1, self.height_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
        )
        self.height_conv_vert = nn.Sequential(
            nn.Conv2d(1, self.height_channels, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
        )
        self.mob_encoder = nn.Sequential(
            nn.Linear(MOB_INFO_SIZE, self.mob_dim),
            nn.GELU(),
        )
        self.loot_encoder = nn.Sequential(
            nn.Linear(LOOT_INFO_SIZE, self.loot_dim),
            nn.GELU(),
        )
        self.mob_pooler = nn.Sequential(
            AttentivePooler(embed_dim=self.mob_dim, num_heads=self.mob_heads, complete_block=False),
            nn.GELU(),
        )
        self.loot_pooler = nn.Sequential(
            AttentivePooler(embed_dim=self.loot_dim, num_heads=self.loot_heads, complete_block=False),
            nn.GELU(),
        )
        self.inventory_encoder = nn.Sequential(
            nn.Linear(INVENTORY_SIZE, self.inventory_dim),
            nn.GELU(),
        )

        self.use_past_actions = config.env.use_past_actions
        if self.use_past_actions:
            self.past_actions_encoder = nn.Sequential(
                nn.Linear(PastActionsWrapper.NUM_PAST_ACTIONS * NUM_ACTIONS, self.past_actions_dim),
                nn.GELU(),
            )

        self.position_scaler = SymmetricLogScaling()

        # some spatial invariance for grid-based observations
        self.full_materials_distance = 2  # 5x5 around the player gets full material information
        self.height_distance = 4  # 9x9 around the player gets height information
        self.total_grid_size = 2 * config.env.observation_distance + 1
        middle = config.env.observation_distance
        self.materials_grid_start = middle - self.full_materials_distance
        self.materials_grid_end = middle + self.full_materials_distance + 1
        self.height_grid_start = middle - self.height_distance
        self.height_grid_end = middle + self.height_distance + 1

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
        materials = materials.view(materials.size(0), self.total_grid_size, self.total_grid_size, self.material_channels)
        # materials in immediate vicinity of the player have full information
        near_materials = materials[:, self.materials_grid_start:self.materials_grid_end,
                                      self.materials_grid_start:self.materials_grid_end]
        near_materials = near_materials.reshape(materials.size(0), -1)
        # materials further away are pooled
        pooled_materials = MaxPool2d(kernel_size=3, stride=3)(materials.permute(0, 3, 1, 2))
        pooled_materials = pooled_materials.reshape(materials.size(0), -1)

        tile_heights = observation["tile_heights"] / 4  # max height is 4
        tile_heights = tile_heights[:, self.height_grid_start:self.height_grid_end,
                                       self.height_grid_start:self.height_grid_end].unsqueeze(1)
        height_features_hor = self.height_conv_hor(tile_heights)
        height_features_hor = height_features_hor.view(height_features_hor.size(0), -1)
        height_features_vert = self.height_conv_vert(tile_heights)
        height_features_vert = height_features_vert.view(height_features_vert.size(0), -1)

        mobs = observation["mobs"]  # (batch_size, NUM_MOBS, MOB_INFO_SIZE)
        # first two mob features are x and y positions, which should be log-scaled
        mobs[:, :, :2] = self.position_scaler(mobs[:, :, :2])
        # third is mob health, 0 to 100
        mobs[:, :, 2] = mobs[:, :, 2] / 100
        # mob kind is already one-hot encoded in positions 3+
        mobs = self.mob_encoder(mobs)
        mob_pool = self.mob_pooler(mobs)
        mob_pool = mob_pool.view(mob_pool.size(0), -1)

        loots = observation["loot"]  # (batch_size, NUM_LOOTS, LOOT_INFO_SIZE)
        loots[:, :, :2] = self.position_scaler(loots[:, :, :2])
        loots = self.loot_encoder(loots)
        loot_pool = self.loot_pooler(loots)
        loot_pool = loot_pool.view(loot_pool.size(0), -1)

        inventory = torch.log(observation["inventory_state"] + 1)
        inventory = self.inventory_encoder(inventory)

        features = [
            near_materials, pooled_materials, height_features_hor, height_features_vert,
            mob_pool, loot_pool, inventory,
            self.extract_flat_features(observation)
        ]

        if self.use_past_actions:
            past_actions = observation["past_actions"]  # (batch_size, NUM_PAST_ACTIONS)
            past_actions = past_actions.long()
            # One-hot encode each action
            past_actions_onehot = torch.eye(NUM_ACTIONS, device=past_actions.device)[past_actions]
            # Flatten to (batch_size, NUM_PAST_ACTIONS * NUM_ACTIONS)
            past_actions_onehot = past_actions_onehot.view(past_actions.size(0), -1)
            past_actions_encoded = self.past_actions_encoder(past_actions_onehot)
            features.append(past_actions_encoded)

        return torch.cat(features, dim=1)

    def extract_flat_features(self, observation: dict) -> torch.Tensor:
        player_pos = observation["player_pos"]
        player_pos[:, :2] = self.position_scaler(player_pos[:, :2])
        player_pos[:, 2] = player_pos[:, 2] / 4  # z position is in range 0 to 4
        player_rot = observation["player_rot"].view(player_pos.size(0)).int()
        player_rot = torch.eye(4).to(player_rot.device)[player_rot]
        hp = observation["hp"] / 100  # max hp is 100
        time = torch.log(observation["time"] + 1) / 7  # becomes 1 at around 1000
        time1 = time % 1  # cyclic time component
        time2 = (time % 2) / 2  # cyclic time component
        action_mask = observation["action_mask"]
        status_effects = torch.log(observation["status_effects"] + 1)
        discovered_actions = observation["discovered_actions"]
        return torch.cat([
            player_pos, player_rot, hp, time, time1, time2, action_mask, status_effects, discovered_actions
        ], dim=1)


class SymmetricLogScaling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        signs = torch.sign(x)
        abs_x = torch.abs(x)
        log_scaled = torch.log(1 + abs_x)
        return signs * log_scaled
