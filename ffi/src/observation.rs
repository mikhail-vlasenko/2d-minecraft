use game_logic::character::player::Player;
use game_logic::settings::DEFAULT_SETTINGS;


pub const OBSERVATION_GRID_SIZE: usize = ((DEFAULT_SETTINGS.window.render_distance * 2) + 1) as usize;

#[repr(C)]
pub struct Observation {
    pub top_materials: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub tile_heights: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub player_pos: (i32, i32, i32),
}

impl Observation {
    pub fn new(vec_top_materials: Vec<Vec<i32>>, vec_tile_heights: Vec<Vec<i32>>, player: &Player) -> Self {
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
        let player_pos = (player.x, player.y, player.z as i32);
        Self {
            top_materials,
            tile_heights,
            player_pos,
        }
    }
}
