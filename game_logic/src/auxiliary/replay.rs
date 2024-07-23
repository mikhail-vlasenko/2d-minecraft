use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use derivative::Derivative;
use serde::{Deserialize, Serialize};
use crate::character::player::Player;
use crate::crafting::material::Material;
use crate::map_generation::field::{AbsolutePos, Field};
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
    
    pub fn save(&self, path: &str) {
        let serialized = self.to_binary_string();
        let mut file = File::create(path).unwrap();
        file.write_all(&serialized).unwrap();
    }
    
    pub fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        Self::from_binary_string(&data)
    }
    
    // pub fn apply_state(&self, field: &mut Field, player: &mut Player) {
    //     let state = &self.states[self.current_step];
    //     field.set_top_materials(state.top_materials.clone());
    //     field.set_tile_heights(state.tile_heights.clone());
    //     field.set_mobs(state.mobs.clone());
    //     player = &state.player;
    // }
}

pub fn get_tile_observation(field: &Field, player: &Player) -> (Vec<Vec<Material>>, Vec<Vec<i32>>) {
    let render_distance = SETTINGS.read().unwrap().window.render_distance;
    let observation_grid_size = render_distance as usize * 2 + 1;
    let mut top_materials = vec![vec![Material::default(); observation_grid_size]; observation_grid_size];
    let mut tile_heights = vec![vec![0; observation_grid_size]; observation_grid_size];
    for i in (player.x - render_distance)..=(player.x + render_distance) {
        for j in (player.y - render_distance)..=(player.y + render_distance) {
            let pos: AbsolutePos = (i, j);
            let idx = ((i - player.x + render_distance) as usize,
                       (j - player.y + render_distance) as usize);
            top_materials[idx.0][idx.1] = field.top_material_at(pos);
            tile_heights[idx.0][idx.1] = field.len_at(pos) as i32;
        }
    }
    (top_materials, tile_heights)
}
