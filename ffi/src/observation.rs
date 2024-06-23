use std::cmp::min;
use std::ffi::CString;
use std::os::raw::c_char;
use interoptopus::ffi_type;
use game_logic::character::player::Player;
use game_logic::map_generation::field::Field;
use game_logic::settings::DEFAULT_SETTINGS;


pub const OBSERVATION_GRID_SIZE: usize = ((DEFAULT_SETTINGS.window.render_distance * 2) + 1) as usize;

#[ffi_type]
#[repr(C)]
pub struct Observation {
    pub top_materials: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub tile_heights: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub player_pos: [i32; 3],  // x, y, z
    pub player_rot: i32,  // one of 4 values: 0, 1, 2, 3
    pub hp: i32,
    pub time: f32,
    pub inventory_state: [i32; 128],  // amount of storables of i-th type
    // player-relative x, player-relative y, type, health. for the 16 closest mobs that are visible. mob type -1 is no mob
    pub mobs: [[i32; 4]; 16],
    pub message: *mut c_char,
}

impl Observation {
    pub fn new(vec_top_materials: Vec<Vec<i32>>, vec_tile_heights: Vec<Vec<i32>>, field: &Field, player: &Player, close_mobs: Vec<[i32; 4]>) -> Self {
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
        let inventory_state = [0; 128];
        let mut mobs = [[-1; 4]; 16];
        for i in 0..min(close_mobs.len(), 16) {
            for j in 0..4 {
                mobs[i][j] = close_mobs[i][j];
            }
        }
        let message = CString::new(player.message.clone()).unwrap().into_raw();
        Self {
            top_materials,
            tile_heights,
            player_pos,
            player_rot,
            hp,
            time,
            inventory_state,
            mobs,
            message,
        }
    }
}
