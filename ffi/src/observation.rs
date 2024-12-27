use std::cmp::min;
use std::ffi::CString;
use std::os::raw::c_char;
use interoptopus::{ffi_constant, ffi_type};
use game_logic::character::player::Player;
use game_logic::crafting::storable::{ALL_STORABLES, Storable};
use game_logic::is_game_over;
use game_logic::map_generation::field::Field;
use game_logic::settings::DEFAULT_SETTINGS;
use crate::game_state::GameState;


// use constants for array sizes to avoid dynamically sized arrays that may leak memory during FFI
#[ffi_constant]
pub const OBSERVATION_GRID_SIZE: u32 = ((DEFAULT_SETTINGS.window.render_distance * 2) + 1) as u32;
#[ffi_constant]
pub const INVENTORY_SIZE: u32 = 26;
#[ffi_constant]
pub const NUM_ACTIONS: u32 = 39;
#[ffi_constant]
pub const NUM_MOB_KINDS: u32 = 5;
#[ffi_constant]
pub const MOB_INFO_SIZE: u32 = 3 + NUM_MOB_KINDS;  // x, y, health share (0 to 100), [one-hot-encoded type]
#[ffi_constant]
pub const MAX_MOBS: u32 = 16;  // also max number of loot items
#[ffi_constant]
pub const LOOT_INFO_SIZE: u32 = 3;
#[ffi_constant]
pub const NUM_MATERIALS: u32 = 13;


#[ffi_type]
#[repr(C)]
pub struct Observation {
    pub top_materials: [[i32; OBSERVATION_GRID_SIZE as usize]; OBSERVATION_GRID_SIZE as usize],
    pub tile_heights: [[i32; OBSERVATION_GRID_SIZE as usize]; OBSERVATION_GRID_SIZE as usize],
    pub player_pos: [i32; 3],  // x, y, z
    pub player_rot: i32,  // one of 4 values: 0, 1, 2, 3
    pub hp: i32,
    pub time: f32,
    pub inventory_state: [i32; INVENTORY_SIZE as usize],  // amount of storables of i-th type
    // player-relative x, player-relative y, health share (0 to 100), [one-hot-encoded type]. for the 16 closest mobs that are visible. mob type -1 is no mob
    pub mobs: [[i32; MOB_INFO_SIZE as usize]; MAX_MOBS as usize],
    // player-relative x, player-relative y, content (1: arrow, 2: other loot, 3: arrow and other loot). content -1 for no loot
    pub loot: [[i32; LOOT_INFO_SIZE as usize]; MAX_MOBS as usize],
    pub action_mask: [i32; NUM_ACTIONS as usize],
    pub score: i32,
    pub message: *mut c_char,
    pub done: bool,
}

impl Observation {
    pub fn new(
        vec_top_materials: Vec<Vec<i32>>, 
        vec_tile_heights: Vec<Vec<i32>>, 
        field: &Field, player: &Player, 
        close_mobs: Vec<[i32; MOB_INFO_SIZE as usize]>, 
        close_loot: [[i32; LOOT_INFO_SIZE as usize]; MAX_MOBS as usize],
        action_mask: [i32; NUM_ACTIONS as usize],
    ) -> Self {
        if vec_top_materials.len() != OBSERVATION_GRID_SIZE as usize || vec_tile_heights.len() != OBSERVATION_GRID_SIZE as usize {
            panic!("Invalid observation size");
        }
        let mut top_materials = [[0; OBSERVATION_GRID_SIZE as usize]; OBSERVATION_GRID_SIZE as usize];
        let mut tile_heights = [[0; OBSERVATION_GRID_SIZE as usize]; OBSERVATION_GRID_SIZE as usize];
        for i in 0..OBSERVATION_GRID_SIZE as usize {
            for j in 0..OBSERVATION_GRID_SIZE as usize {
                top_materials[i][j] = vec_top_materials[i][j];
                tile_heights[i][j] = vec_tile_heights[i][j];
            }
        }
        let player_pos = [player.x, player.y, player.z as i32];
        let player_rot = player.get_rotation() as i32;
        let hp = player.get_hp();
        let time = field.get_time();
        
        let mut inventory_state = [0; INVENTORY_SIZE as usize];
        for (storable, n) in player.get_inventory() {
            let idx = storable_to_inv_index(storable);
            inventory_state[idx] = *n as i32;
        }
        
        let mut mobs = [[0; MOB_INFO_SIZE as usize]; MAX_MOBS as usize];
        for i in 0..min(close_mobs.len(), MAX_MOBS as usize) {
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
            action_mask,
            score,
            message,
            done,
        }
    }
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            top_materials: [[0; OBSERVATION_GRID_SIZE as usize]; OBSERVATION_GRID_SIZE as usize],
            tile_heights: [[0; OBSERVATION_GRID_SIZE as usize]; OBSERVATION_GRID_SIZE as usize],
            player_pos: [0; 3],
            player_rot: 0,
            hp: 0,
            time: 0.0,
            inventory_state: [0; INVENTORY_SIZE as usize],
            mobs: [[0; MOB_INFO_SIZE as usize]; MAX_MOBS as usize],
            loot: [[0, 0, -1]; MAX_MOBS as usize],
            action_mask: [0; NUM_ACTIONS as usize],
            score: 0,
            message: CString::new(String::from("")).unwrap().into_raw(),
            done: false,
        }
    }
}


fn storable_to_inv_index(storable: &Storable) -> usize {
    ALL_STORABLES.iter().position(|s| s == storable).unwrap()
}


pub fn make_action_mask(game_state: &GameState) -> [i32; NUM_ACTIONS as usize] {
    let mut mask = [0; NUM_ACTIONS as usize];
    if !game_state.is_done() {
        for i in 0..NUM_ACTIONS as usize {
            if game_state.can_take_action(i as i32) {
                mask[i] = 1;
            }
        }
    }
    mask
}
