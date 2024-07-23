use std::cell::RefCell;
use strum::IntoEnumIterator;
use game_logic::auxiliary::actions::{Action, can_take_action};
use game_logic::character::player::Player;
use game_logic::{handle_action, is_game_over, SETTINGS};
use game_logic::auxiliary::replay::Replay;
use game_logic::map_generation::field::{absolute_to_relative, Field};
use game_logic::map_generation::field::AbsolutePos;
use game_logic::map_generation::field_observation::get_tile_observation;
use game_logic::map_generation::mobs::mob_kind::MobKind;
use crate::observation::Observation;


#[derive(Debug)]
pub struct GameState {
    field: Field,
    player: Player,
    replay: Replay
}

impl GameState {
    pub fn new() -> Self {
        let (field, player) = game_logic::init_field_player();
        // will not be recorded if record_replays is disabled in settings
        let replay = Replay::new();
        Self {
            field,
            player,
            replay,
        }
    }

    pub fn step(&mut self, action: &Action) {
        handle_action(action, &mut self.field, &mut self.player, &RefCell::new(false), &RefCell::new(false), None, &mut self.replay);
    }

    pub fn step_i32(&mut self, action: i32) {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        self.step(&action);
    }

    pub fn get_observation(&self) -> Observation {
        let (top_materials, tile_heights) = get_tile_observation(&self.field, &self.player);
        let top_materials = top_materials.iter().map(|row| row.iter().map(|mat| (*mat).into()).collect()).collect();
        Observation::new(top_materials, tile_heights, &self.field, &self.player, self.get_closest_mobs())
    }

    /// Produces a vector of 4-arrays of mob information: x, y, type, health
    /// The vector is sorted by manhattan distance from the player
    /// x and y are player-relative
    /// mob kind is an index in the MobKind enum
    /// health is integer - rounded percentage of max health, so from 0 to 100
    pub fn get_closest_mobs(&self) -> Vec<[i32; 4]> {
        let mob_kinds = MobKind::iter().collect::<Vec<MobKind>>();
        let mut mobs = self.field.close_mob_info(|mob| {
            let pos = absolute_to_relative((mob.pos.x, mob.pos.y), &self.player);
            [pos.0, pos.1, 
                mob_kinds.iter().position(| kind | { kind == mob.get_kind() }).unwrap() as i32, 
                (mob.get_hp_share() * 100.0) as i32]
        }, &self.player);

        // Sorting the mobs by manhattan distance from the player
        mobs.sort_by(|a, b| {
            let dist_a = a[0].abs() + a[1].abs();
            let dist_b = b[0].abs() + b[1].abs();
            dist_a.cmp(&dist_b)
        });
        
        mobs
    }
    
    pub fn is_done(&self) -> bool {
        is_game_over(&self.player)
    }
    
    pub fn reset(&mut self) {
        let (field, player) = game_logic::init_field_player();
        self.field = field;
        self.player = player;
    }
    
    pub fn can_take_action(&self, action: i32) -> bool {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        can_take_action(&action, &self.player, &self.field)
    }
}
