use std::cell::RefCell;
use game_logic::auxiliary::actions::Action;
use game_logic::character::player::Player;
use game_logic::handle_action;
use game_logic::map_generation::field::Field;
use game_logic::map_generation::field::AbsolutePos;
use crate::ffi_config::CONFIG;
use crate::observation::Observation;


#[derive(Debug)]
pub struct GameState {
    field: Field,
    player: Player,
}

impl GameState {
    pub fn new() -> Self {
        let (field, player) = game_logic::init_field_player();
        Self {
            field,
            player,
        }
    }
    
    pub fn step(&mut self, action: &Action) {
        handle_action(action, &mut self.field, &mut self.player, &RefCell::new(false), &RefCell::new(false), None);
    }
    
    pub fn get_observation(&self) -> Observation {
        let mut top_materials = vec![vec![0; CONFIG.observation_grid_size]; CONFIG.observation_grid_size];
        let mut tile_depths = vec![vec![0; CONFIG.observation_grid_size]; CONFIG.observation_grid_size];
        for i in (self.player.x - CONFIG.render_distance as i32)..=(self.player.x + CONFIG.render_distance as i32) {
            for j in (self.player.y - CONFIG.render_distance as i32)..=(self.player.y + CONFIG.render_distance as i32) {
                let pos: AbsolutePos = (i, j);
                let idx = ((i - self.player.x + CONFIG.render_distance as i32) as usize, 
                                       (j - self.player.y + CONFIG.render_distance as i32) as usize);
                top_materials[idx.0][idx.1] = self.field.top_material_at(pos).into();
                tile_depths[idx.0][idx.1] = self.field.len_at(pos) as i32;
            }
        }
        Observation::new(top_materials, tile_depths)
    }
}
