use lazy_static::lazy_static;
use game_logic::SETTINGS;
use game_logic::settings::{DEFAULT_SETTINGS, Settings};

pub struct FFIConfig {
    pub render_distance: usize,
    pub observation_grid_size: usize,
}

impl FFIConfig {
    pub fn new(settings: Settings) -> Self {
        if settings.window.render_distance != DEFAULT_SETTINGS.window.render_distance {
            panic!("Changing render distance for FFI requires rebuilding game_logic crate");
        }
        let render_distance = settings.window.render_distance as usize;
        let observation_grid_size = (render_distance * 2) + 1;
        Self {
            render_distance,
            observation_grid_size,
        }
    }
}

lazy_static! {
    pub static ref CONFIG: FFIConfig = FFIConfig::new(SETTINGS.read().unwrap().clone());
}
