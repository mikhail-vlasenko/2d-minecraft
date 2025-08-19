use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;
use crate::auxiliary::actions::Action;
use crate::character::player::Player;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::map_generation::block::Block;
use crate::map_generation::field::Field;
use crate::SETTINGS;

#[derive(PartialEq, Copy, Clone, EnumIter, Serialize, Deserialize, Debug, Hash, Eq)]
pub enum Ability {
    /// Deals damage to all enemies in the 8 tiles around the player.
    WhirlAttack,
    /// 2 steps forward or until first enemy/obstacle. 1x, 1.5x, 2x melee attack of damage to the first enemy depending on distance travelled.
    Charge,
    /// Places 2 blocks each on 3 forward tiles in a line perpendicular to the player.
    Barricade,
    /// Heals 100% of the player's max HP.
    SecondWind,
}

impl Ability {
    pub fn get_cooldown(&self) -> i32 {
        match self {
            Ability::WhirlAttack => 20,
            Ability::Charge => 20,
            Ability::Barricade => 30,
            Ability::SecondWind => 700,
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            Ability::WhirlAttack => "Whirl Attack",
            Ability::Charge => "Charge",
            Ability::Barricade => "Barricade",
            Ability::SecondWind => "Second Wind",
        }
    }
}

impl Default for Ability {
    fn default() -> Self {
        Ability::WhirlAttack
    }
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct AbilityState {
    cooldowns: HashMap<Ability, i32>,
}

impl AbilityState {
    pub fn new() -> Self {
        AbilityState {
            cooldowns: HashMap::new(),
        }
    }

    pub fn step_cooldowns(&mut self) {
        // Reduce all cooldowns by 1, removing entries that reach 0
        self.cooldowns.retain(|_, cooldown| {
            *cooldown -= 1;
            *cooldown > 0
        });
    }

    pub fn is_on_cooldown(&self, ability: &Ability) -> bool {
        self.cooldowns.get(ability).unwrap_or(&0) > &0
    }

    pub fn get_remaining_cooldown(&self, ability: &Ability) -> i32 {
        *self.cooldowns.get(ability).unwrap_or(&0)
    }

    pub fn set_cooldown(&mut self, ability: &Ability) {
        self.cooldowns.insert(*ability, ability.get_cooldown());
    }
}

impl Player {
    pub fn use_ability(&mut self, ability: &Ability, field: &mut Field) -> f32 {
        self.stop_interacting();
        if self.ability_state.is_on_cooldown(ability) {
            let remaining = self.ability_state.get_remaining_cooldown(ability);
            self.add_message(&format!("{} is on cooldown ({} turns remaining)",
                                      ability.get_name(), remaining));
            return 0.0;
        }

        let action_spent = match ability {
            Ability::WhirlAttack => self.use_whirl_attack(field),
            Ability::Charge => self.use_charge(field),
            Ability::Barricade => self.use_barricade(field),
            Ability::SecondWind => self.use_second_wind(),
        };

        // only set cooldown if the ability was successfully used
        if action_spent > 0.0 {
            self.ability_state.set_cooldown(ability);
        }

        action_spent
    }

    fn use_whirl_attack(&mut self, field: &mut Field) -> f32 {
        let deltas = vec![
            (-1, -1), (-1, 0), (-1, 1),
            (0,  -1),          (0,  1),
            (1,  -1), (1,  0), (1,  1),
        ];
        let (player_x, player_y) = self.xy();
        let mut enemies_hit = 0;

        for (dx, dy) in deltas {
            let target_pos = (player_x + dx, player_y + dy);
            if field.is_occupied(target_pos) & (field.len_at(target_pos) <= self.get_position().z + 1)  {
                self.damage_mob(field, target_pos, self.get_melee_damage());
                enemies_hit += 1;
            }
        }

        self.add_message(&format!("Whirl attack hit {} enemies!", enemies_hit));
        self.get_speed_multiplier()
    }

    fn use_charge(&mut self, field: &mut Field) -> f32 {
        let direction = self.coords_from_rotation();
        let walk_action = match direction {
            (-1, 0) => Action::WalkNorth,
            (0, -1) => Action::WalkWest,
            (1, 0) => Action::WalkSouth,
            (0, 1) => Action::WalkEast,
            _ => panic!()
        };
        let max_distance = 3;
        let (start_x, start_y) = self.xy();

        // Move forward up to 2 tiles or until hitting obstacle/enemy
        for step in 1..=max_distance {
            if !self.can_walk(&walk_action, field) {
                break;
            }

            let target_pos = (start_x + direction.0 * step, start_y + direction.1 * step);

            if field.is_occupied(target_pos) {
                // deal damage based on distance
                let damage_multiplier = match step {
                    1 => 1.0,
                    2 => 1.5,
                    _ => 2.0,
                };
                let damage = (self.get_melee_damage() as f32 * damage_multiplier) as i32;
                self.damage_mob(field, target_pos, damage);
                self.add_message(&format!("Charge attack deals {} damage!", damage));
                break;
            }

            self.walk(&walk_action, field);
        }
        self.get_speed_multiplier()
    }

    fn use_barricade(&mut self, field: &mut Field) -> f32 {
        let forward = self.coords_from_rotation();
        let perpendicular = (-forward.1, forward.0); // Rotate 90 degrees

        let (player_x, player_y) = self.xy();
        let blocks_per_tile = 2;
        let material = Material::Plank;

        // place blocks on 3 tiles perpendicular to facing direction
        for i in vec![0, -1, 1] {  // start with the middle one
            let base_pos = (
                player_x + forward.0 + perpendicular.0 * i,
                player_y + forward.1 + perpendicular.1 * i
            );

            for _ in 0..blocks_per_tile {
                if !field.full_at(base_pos) {
                    if self.drop(&Storable::M(material), 1) {
                        field.push_at(Block { material }, base_pos);
                    }
                }
            }
        }
        self.get_speed_multiplier()
    }

    fn use_second_wind(&mut self) -> f32 {
        self.heal(SETTINGS.read().unwrap().player.max_hp);
        self.get_speed_multiplier()
    }

    pub fn get_ability_cooldown(&self, ability: &Ability) -> i32 {
        self.ability_state.get_remaining_cooldown(ability)
    }

    pub fn is_ability_ready(&self, ability: &Ability) -> bool {
        self.get_ability_cooldown(ability) == 0
    }

    // Method to step cooldowns (should be called each turn)
    pub fn step_ability_cooldowns(&mut self) {
        self.ability_state.step_cooldowns();
    }
}
