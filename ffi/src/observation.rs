use std::cmp::min;
use std::ffi::CString;
use std::os::raw::c_char;
use interoptopus::ffi_type;
use game_logic::character::player::Player;
use game_logic::crafting::storable::{ALL_STORABLES, Storable};
use game_logic::is_game_over;
use game_logic::map_generation::field::Field;
use game_logic::settings::DEFAULT_SETTINGS;
use crate::game_state::GameState;


// use constants for array sizes to avoid dynamically sized arrays that may leak memory during FFI
pub const OBSERVATION_GRID_SIZE: usize = ((DEFAULT_SETTINGS.window.render_distance * 2) + 1) as usize;
pub const INVENTORY_SIZE: usize = 26;
pub const NUM_ACTIONS: usize = 39;
pub const MOB_INFO_SIZE: usize = 4;
pub const MAX_MOBS: usize = 16;  // also max number of loot items
pub const LOOT_INFO_SIZE: usize = 3;


#[ffi_type]
#[repr(C)]
pub struct Observation {
    pub top_materials: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub tile_heights: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub player_pos: [i32; 3],  // x, y, z
    pub player_rot: i32,  // one of 4 values: 0, 1, 2, 3
    pub hp: i32,
    pub time: f32,
    pub inventory_state: [i32; INVENTORY_SIZE],  // amount of storables of i-th type
    // player-relative x, player-relative y, type, health. for the 16 closest mobs that are visible. mob type -1 is no mob
    pub mobs: [[i32; MOB_INFO_SIZE]; MAX_MOBS],
    // player-relative x, player-relative y, content (1: arrow, 2: other loot, 3: arrow and other loot). content -1 for no loot
    pub loot: [[i32; LOOT_INFO_SIZE]; MAX_MOBS],
    pub score: i32,
    pub message: *mut c_char,
    pub done: bool,
}

impl Observation {
    pub fn new(
        vec_top_materials: Vec<Vec<i32>>, 
        vec_tile_heights: Vec<Vec<i32>>, 
        field: &Field, player: &Player, 
        close_mobs: Vec<[i32; MOB_INFO_SIZE]>, 
        close_loot: [[i32; LOOT_INFO_SIZE]; MAX_MOBS]
    ) -> Self {
        if vec_top_materials.len() != OBSERVATION_GRID_SIZE || vec_tile_heights.len() != OBSERVATION_GRID_SIZE {
            panic!("Invalid observation size");
        }
        let mut top_materials = [[0; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE];
        let mut tile_heights = [[0; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE];
        for i in 0..OBSERVATION_GRID_SIZE {
            for j in 0..OBSERVATION_GRID_SIZE {
                top_materials[i][j] = vec_top_materials[i][j];
                tile_heights[i][j] = vec_tile_heights[i][j];
            }
        }
        let player_pos = [player.x, player.y, player.z as i32];
        let player_rot = player.get_rotation() as i32;
        let hp = player.get_hp();
        let time = field.get_time();
        
        let mut inventory_state = [0; INVENTORY_SIZE];
        for (storable, n) in player.get_inventory() {
            let idx = storable_to_inv_index(storable);
            inventory_state[idx] = *n as i32;
        }
        
        let mut mobs = [[0, 0, -1, 0]; MAX_MOBS];
        for i in 0..min(close_mobs.len(), MAX_MOBS) {
            for j in 0..4 {
                mobs[i][j] = close_mobs[i][j];
            }
        }
        let loot = close_loot;
        let score = player.get_score();
        let message = CString::new(player.message.clone()).unwrap().into_raw();
        let done = is_game_over(player);
        Self {
            top_materials,
            tile_heights,
            player_pos,
            player_rot,
            hp,
            time,
            inventory_state,
            mobs,
            loot,
            score,
            message,
            done,
        }
    }
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            top_materials: [[0; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
            tile_heights: [[0; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
            player_pos: [0; 3],
            player_rot: 0,
            hp: 0,
            time: 0.0,
            inventory_state: [0; INVENTORY_SIZE],
            mobs: [[0, 0, -1, 0]; MAX_MOBS],
            loot: [[0, 0, -1]; MAX_MOBS],
            score: 0,
            message: CString::new(String::from("")).unwrap().into_raw(),
            done: false,
        }
    }
}


fn storable_to_inv_index(storable: &Storable) -> usize {
    ALL_STORABLES.iter().position(|s| s == storable).unwrap()
}

#[ffi_type]
#[repr(C)]
pub struct ActionMask {
    pub mask: [i32; NUM_ACTIONS],
}

impl ActionMask {
    pub fn new(game_state: &GameState) -> Self {
        let mut mask = [0; NUM_ACTIONS];
        if !game_state.is_done() {
            for i in 0..NUM_ACTIONS {
                if game_state.can_take_action(i as i32) {
                    mask[i] = 1;
                }
            }
        }
        Self { mask }
    }
}
