use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;
use derivative::Derivative;
use serde::{Deserialize, Serialize};
use crate::character::player::Player;
use crate::crafting::material::Material;
use crate::map_generation::field::{AbsolutePos, Field};
use crate::map_generation::field_observation::get_tile_observation;
use crate::map_generation::mobs::mob::Mob;
use crate::SETTINGS;

/// Stores an incomplete snapshot of the game state.
/// Incomplete for saving memory and time.
/// The complete one is also not necessary for the replay as most information in not displayed.
#[derive(Serialize, Deserialize, Debug, Derivative)]
pub struct ObservableState {
    pub top_materials: Vec<Vec<Material>>,
    pub tile_heights: Vec<Vec<i32>>,
    // todo: interactables
    // todo: loot and arrows
    pub mobs: Vec<Mob>,
    pub player: Player,
    pub time: f32,
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
    
    pub fn record_state(&mut self, field: &Field, player: &Player, ) {
        let (top_materials, tile_heights) = get_tile_observation(field, player);
        let mobs = field.close_mob_info(|mob| mob.clone(), player);
        let time = field.get_time();
        self.states.push(ObservableState {
            top_materials,
            tile_heights,
            mobs,
            player: player.clone(),
            time,
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
    
    pub fn make_save_name(&self) -> String {
        let mut name = String::from("replay_");
        name.push_str(&chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
        let score = self.states.last().unwrap().player.get_score();
        name.push_str(&format!("_score_{}", score));
        name.push_str(".postcard");
        name
    }
    
    pub fn apply_state(&mut self, field: &mut Field, player: &mut Player) {
        let state = &self.states[self.current_step];
        player.clone_from(&state.player);
        field.set_time(state.time);
        field.set_visible_tiles(&state.top_materials, &state.tile_heights, (player.x, player.y));
        field.set_mobs(state.mobs.clone());
        field.animations_buffer.clear();
        self.current_step += 1;
    }
    
    pub fn step_back(&mut self, field: &mut Field, player: &mut Player) {
        if self.current_step > 0 {
            self.current_step -= 1;
            if self.current_step > 0 {
                self.current_step -= 1;
            }
            self.apply_state(field, player);
        }
    }
    
    pub fn finished(&self) -> bool {
        self.current_step >= self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}
