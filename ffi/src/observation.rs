use game_logic::settings::DEFAULT_SETTINGS;


pub const OBSERVATION_GRID_SIZE: usize = ((DEFAULT_SETTINGS.window.render_distance * 2) + 1) as usize;

#[repr(C)]
pub struct Observation {
    pub top_materials: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
    pub tile_depths: [[i32; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE],
}

impl Observation {
    pub fn new(vec_top_materials: Vec<Vec<i32>>, vec_tile_depths: Vec<Vec<i32>>) -> Self {
        if vec_top_materials.len() != OBSERVATION_GRID_SIZE || vec_tile_depths.len() != OBSERVATION_GRID_SIZE {
            panic!("Invalid observation size");
        }
        let mut top_materials = [[0; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE];
        let mut tile_depths = [[0; OBSERVATION_GRID_SIZE]; OBSERVATION_GRID_SIZE];
        for i in 0..OBSERVATION_GRID_SIZE {
            for j in 0..OBSERVATION_GRID_SIZE {
                top_materials[i][j] = vec_top_materials[i][j];
                tile_depths[i][j] = vec_tile_depths[i][j];
            }
        }
        Self {
            top_materials,
            tile_depths,
        }
    }
}
