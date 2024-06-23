use lazy_static::lazy_static;
use game_logic::SETTINGS;
use game_logic::settings::Settings;

pub struct GraphicalConfig {
    pub render_distance: usize,
    pub tiles_per_row: u32,
    pub disp_coef: f32,
}

impl GraphicalConfig {
    pub fn new(settings: Settings) -> Self {
        let render_distance = settings.window.render_distance as usize;
        let tiles_per_row = ((render_distance * 2) + 1) as u32;
        let disp_coef = 2.0 / tiles_per_row as f32;
        let a = SETTINGS.read().unwrap();
        Self {
            render_distance,
            tiles_per_row,
            disp_coef,
        }
    }
}

lazy_static! {
    pub static ref CONFIG: GraphicalConfig = GraphicalConfig::new(SETTINGS.read().unwrap().clone());
}
