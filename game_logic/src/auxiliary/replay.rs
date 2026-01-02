use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use derivative::Derivative;
use serde::{Deserialize, Serialize};
use crate::character::player::Player;
use crate::crafting::consumable::Consumable;
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::auxiliary::animations::{AnimationManager, ProjectileAnimation, TileAnimation};
use crate::map_generation::field::{AbsolutePos, Field, relative_to_absolute, RelativePos};
use crate::map_generation::field_observation::get_tile_observation;
use crate::map_generation::mobs::mob::Mob;

/// Stores an incomplete snapshot of the game state.
/// Incomplete for saving memory and time.
/// The complete one is also not necessary for the replay as most information in not displayed.
#[derive(Serialize, Deserialize, Debug, Derivative)]
pub struct ObservableState {
    pub top_materials: Vec<Vec<Material>>,
    pub tile_heights: Vec<Vec<i32>>,
    // todo: interactables
    pub loot_positions: Vec<AbsolutePos>,
    pub arrow_positions: Vec<AbsolutePos>,
    pub mobs: Vec<Mob>,
    pub player: Player,
    pub time: f32,
    /// Animations started in this step (from `AnimationsBuffer`). todo: channeling animations
    pub new_tile_animations: Vec<TileAnimation>,
    pub new_projectile_animations: Vec<ProjectileAnimation>,
}

#[derive(Serialize, Deserialize, Debug, Derivative)]
pub struct Replay {
    states: Vec<ObservableState>,
    #[serde(skip)]
    current_step: usize,
}

impl Replay {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            current_step: 0,
        }
    }
    
    pub fn record_state(
        &mut self,
        field: &Field,
        player: &Player,
        new_tile_animations: Vec<TileAnimation>,
        new_projectile_animations: Vec<ProjectileAnimation>,
    ) {
        let (top_materials, tile_heights) = get_tile_observation(field, player);
        let loot_positions = self.vec_to_absolute(field.loot_indices(player), player);
        let arrow_positions = self.vec_to_absolute(field.arrow_indices(player), player);
        let mobs = field.close_mob_info(|mob| mob.clone(), player);
        let time = field.get_time();
        self.states.push(ObservableState {
            top_materials,
            tile_heights,
            loot_positions,
            arrow_positions,
            mobs,
            player: player.clone(),
            time,
            new_tile_animations,
            new_projectile_animations,
        });
    }

    pub fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    pub fn from_binary_string(data: &Vec<u8>) -> Self {
        let mut deserialized: Replay = postcard::from_bytes(data).unwrap();
        deserialized.current_step = 0;
        deserialized
    }
    
    pub fn save(&self, path: &Path) {
        let serialized = self.to_binary_string();
        let mut file = File::create(path).unwrap();
        file.write_all(&serialized).unwrap();
    }
    
    pub fn load(path: &Path) -> Self {
        let mut file = File::open(path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        Self::from_binary_string(&data)
    }
    
    /// Returns the name of the save file based on the current time and player score.
    /// Returns None if there are no states and the save is not possible.
    pub fn make_save_name(&self) -> Option<String> {
        let mut name = String::from("replay_");
        name.push_str(&chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
        let state_option = self.states.last();
        if let Some(state) = state_option {
            name.push_str(&format!("_score_{}", state.player.get_score()));
            name.push_str(".postcard");
            Some(name)
        } else {
            None
        }

    }
    
    pub fn apply_state(
        &mut self,
        field: &mut Field,
        player: &mut Player,
        animation_manager: Option<&mut AnimationManager>,
    ) {
        let state = &self.states[self.current_step];
        player.clone_from(&state.player);
        field.set_time(state.time);
        field.set_visible_tiles(&state.top_materials, &state.tile_heights, (player.x, player.y));
        field.set_mobs(state.mobs.clone());
        self.clear_field_loot(field, player);
        self.place_field_loot(field);
        field.animations_buffer.clear();
        if let Some(animation_manager) = animation_manager {
            // These animations are expected to be short-lived, so we just start them at their step.
            animation_manager.clear();
            for animation in state.new_tile_animations.iter().copied() {
                animation_manager.add_tile_animation(animation);
            }
            for animation in state.new_projectile_animations.iter().copied() {
                animation_manager.add_projectile_animation(animation);
            }
        }
        self.current_step += 1;
    }
    
    pub fn step_back(
        &mut self,
        field: &mut Field,
        player: &mut Player,
        animation_manager: Option<&mut AnimationManager>,
    ) {
        if self.current_step > 0 {
            self.current_step -= 1;
            if self.current_step > 0 {
                self.current_step -= 1;
            }
            self.apply_state(field, player, animation_manager);
        }
    }
    
    pub fn finished(&self) -> bool {
        self.current_step >= self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
    
    fn clear_field_loot(&self, field: &mut Field, player: &Player) {
        let loot_positions = self.vec_to_absolute(field.loot_indices(player), player);
        let arrow_positions = self.vec_to_absolute(field.arrow_indices(player), player);
        for pos in loot_positions {
            field.gather_loot_at(pos);
        }
        for pos in arrow_positions {
            field.gather_loot_at(pos);
        }
    }
    
    fn place_field_loot(&self, field: &mut Field) {
        let state = &self.states[self.current_step];
        for pos in &state.loot_positions {
            // place a dummy loot, as it will not actually be picked up, just drawn
            field.add_loot_at(vec![Storable::C(Consumable::RawMeat)], pos.clone());
        }
        for pos in &state.arrow_positions {
            field.add_loot_at(vec![Storable::I(Item::Arrow)], pos.clone());
        }
    }
    
    fn vec_to_absolute(&self, vec: Vec<RelativePos>, player: &Player) -> Vec<AbsolutePos> {
        vec.iter().map(|pos| relative_to_absolute(*pos, player)).collect()
    }
}
