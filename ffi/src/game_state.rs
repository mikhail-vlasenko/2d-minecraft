use std::cell::RefCell;
use strum::IntoEnumIterator;
use game_logic::auxiliary::actions::Action;
use game_logic::character::player::Player;
use game_logic::handle_action;
use game_logic::map_generation::field::Field;
use game_logic::map_generation::field::AbsolutePos;
use game_logic::map_generation::mobs::mob_kind::MobKind;
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

    pub fn step_i32(&mut self, action: i32) {
        let action = Action::from(action);
        self.step(&action);
    }

    pub fn get_observation(&self) -> Observation {
        let mut top_materials = vec![vec![0; CONFIG.observation_grid_size]; CONFIG.observation_grid_size];
        let mut tile_heights = vec![vec![0; CONFIG.observation_grid_size]; CONFIG.observation_grid_size];
        for i in (self.player.x - CONFIG.render_distance as i32)..=(self.player.x + CONFIG.render_distance as i32) {
            for j in (self.player.y - CONFIG.render_distance as i32)..=(self.player.y + CONFIG.render_distance as i32) {
                let pos: AbsolutePos = (i, j);
                let idx = ((i - self.player.x + CONFIG.render_distance as i32) as usize,
                                       (j - self.player.y + CONFIG.render_distance as i32) as usize);
                top_materials[idx.0][idx.1] = self.field.top_material_at(pos).into();
                tile_heights[idx.0][idx.1] = self.field.len_at(pos) as i32;
            }
        }
        Observation::new(top_materials, tile_heights, &self.field, &self.player, self.get_closest_mobs())
    }

    // Produces a vector of 4-arrays of mob information: x, y, type, health
    // The vector is sorted by manhattan distance from the player
    // x and y are player-relative
    pub fn get_closest_mobs(&self) -> Vec<[i32; 4]> {
        let mob_kinds = MobKind::iter().collect::<Vec<MobKind>>();
        let mut mobs = vec![];
        for i in 0..mob_kinds.len(){
            let these_mobs = self.field.mob_indices(&self.player, mob_kinds[i]);
            for j in 0..these_mobs.len(){
                let pos = these_mobs[j].0;
                // todo: hp
                mobs.push([pos.0, pos.1, i as i32, 1]);
            }
        }

        // Sorting the mobs by manhattan distance from the player
        mobs.sort_by(|a, b| {
            let dist_a = a[0].abs() + a[1].abs();
            let dist_b = b[0].abs() + b[1].abs();
            dist_a.cmp(&dist_b)
        });
        
        mobs
    }
}
