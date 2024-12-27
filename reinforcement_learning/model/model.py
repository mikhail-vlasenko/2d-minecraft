import torch
import torch.nn as nn
import gymnasium as gym
from torch.nn import MaxPool2d

from python_wrapper.ffi_elements import INVENTORY_SIZE
from python_wrapper.observation import NUM_MATERIALS, MOB_INFO_SIZE, LOOT_INFO_SIZE
from reinforcement_learning.config import CONFIG
from reinforcement_learning.model.attentive_pooler import AttentivePooler


class FeatureExtractor(nn.Module):
    def __init__(self, observation_space):
        super(FeatureExtractor, self).__init__()

        self.material_channels = 8
        self.height_channels = 8
        self.mob_heads = 12
        self.mob_dim = 8 * self.mob_heads
        self.loot_heads = 12
        self.loot_dim = 4 * self.loot_heads

        self.block_encoder = nn.Sequential(
            nn.Linear(NUM_MATERIALS, self.material_channels),
            nn.Tanh(),  # relu will likely lead to dead neurons, as input is one-hot
        )
        self.height_conv = nn.Sequential(
            nn.Conv2d(1, self.height_channels, kernel_size=3, stride=(1, 1), padding=(0, 0)),
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
            nn.Linear(INVENTORY_SIZE, INVENTORY_SIZE),
            nn.GELU(),
        )

        self.position_scaler = SymmetricLogScaling()

        # some spatial invariance for grid-based observations
        self.full_materials_distance = 2  # 5x5 around the player gets full material information
        self.height_distance = 3  # 7x7 around the player gets height information
        self.total_grid_size = 2 * CONFIG.env.observation_distance + 1
        middle = CONFIG.env.observation_distance
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
        height_features = self.height_conv(tile_heights)
        height_features = height_features.view(height_features.size(0), -1)

        mobs = observation["mobs"]  # (batch_size, NUM_MOBS, MOB_INFO_SIZE)
        # first two mob features are x and y positions, which should be log-scaled
        mobs[:, :, :2] = self.position_scaler(mobs[:, :, :2])
        # third is mob health, 0 to 100
        mobs[:, :, 2] = mobs[:, :, 2] / 100
        mobs = self.mob_encoder(mobs)
        mob_pool = self.mob_pooler(mobs)
        mob_pool = mob_pool.view(mob_pool.size(0), -1)

        loots = observation["loot"]  # (batch_size, NUM_LOOTS, LOOT_INFO_SIZE)
        loots[:, :, :2] = self.position_scaler(loots[:, :, :2])
        loots = self.loot_encoder(loots)
        loot_pool = self.loot_pooler(loots)
        loot_pool = loot_pool.view(loot_pool.size(0), -1)

        inventory = observation["inventory_state"] / 4  # arbitrary downscaling
        inventory = self.inventory_encoder(inventory)

        return torch.cat([near_materials, pooled_materials, height_features, mob_pool, loot_pool, inventory,
                          self.extract_flat_features(observation)], dim=1)

    def extract_flat_features(self, observation: dict) -> torch.Tensor:
        player_pos = observation["player_pos"]
        player_pos[:, :2] = self.position_scaler(player_pos[:, :2])
        player_pos[:, 2] = player_pos[:, 2] / 4  # z position is in range 0 to 4
        player_rot = observation["player_rot"].view(player_pos.size(0)).int()
        player_rot = torch.eye(4).to(player_rot.device)[player_rot]
        hp = observation["hp"] / 100  # max hp is 100
        time = torch.log(observation["time"] + 1) / 7  # becomes 1 at around 1000
        time1 = time % 1  # cyclic time component
        action_mask = observation["action_mask"]
        discovered_actions = observation["discovered_actions"]
        return torch.cat([player_pos, player_rot, hp, time, time1, action_mask, discovered_actions], dim=1)


class SymmetricLogScaling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        signs = torch.sign(x)
        abs_x = torch.abs(x)
        log_scaled = torch.log(1 + abs_x)
        return signs * log_scaled
