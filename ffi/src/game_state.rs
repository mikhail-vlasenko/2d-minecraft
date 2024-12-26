use std::cell::RefCell;
use std::cmp::min;
use std::path::PathBuf;
use strum::IntoEnumIterator;
use game_logic::auxiliary::actions::{Action, can_take_action};
use game_logic::character::player::Player;
use game_logic::{handle_action, is_game_over, SETTINGS};
use game_logic::auxiliary::replay::Replay;
use game_logic::character::milestones::MilestoneTracker;
use game_logic::map_generation::field::{absolute_to_relative, Field, RelativePos};
use game_logic::map_generation::field_observation::get_tile_observation;
use game_logic::map_generation::mobs::mob_kind::MobKind;
use game_logic::map_generation::save_load::load_game;
use crate::observation::{ActionMask, LOOT_INFO_SIZE, MAX_MOBS, MOB_INFO_SIZE, Observation};


#[derive(Debug)]
pub struct GameState {
    field: Field,
    player: Player,
    recorded_replay: Replay,
    milestone_tracker: MilestoneTracker,
}

impl GameState {
    pub fn new() -> Self {
        let (field, player, recorded_replay, milestone_tracker) = game_logic::init_game();
        Self {
            field,
            player,
            // will not be recorded if record_replays is disabled in settings
            recorded_replay,
            milestone_tracker,
        }
    }

    pub fn step(&mut self, action: &Action) {
        handle_action(
            action, &mut self.field, &mut self.player, 
            &RefCell::new(false), &RefCell::new(false), None, &mut self.recorded_replay, &mut self.milestone_tracker
        );
        if self.is_done() {
            // save the replay
            if SETTINGS.read().unwrap().record_replays && !self.recorded_replay.is_empty() {
                let path = PathBuf::from(SETTINGS.read().unwrap().replay_folder.clone().into_owned());
                let name = self.recorded_replay.make_save_name();
                let path = path.join(name.clone());
                self.recorded_replay.save(path.as_path());
                self.recorded_replay = Replay::new();
            }
        }
    }

    pub fn step_i32(&mut self, action: i32) {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        if !action.ffi_disabled() {
            self.step(&action);
        }
    }

    pub fn get_observation(&self) -> Observation {
        let (top_materials, tile_heights) = get_tile_observation(&self.field, &self.player);
        let top_materials = top_materials.iter().map(|row| row.iter().map(|mat| (*mat).into()).collect()).collect();
        Observation::new(top_materials, tile_heights, &self.field, &self.player, self.get_closest_mobs(), self.make_loot_array(), ActionMask::new(&self))
    }

    /// Produces a vector of 4-arrays of mob information: x, y, type, health
    /// The vector is sorted by manhattan distance from the player
    /// x and y are player-relative
    /// mob kind is an index in the MobKind enum
    /// health is integer - rounded percentage of max health, so from 0 to 100
    pub fn get_closest_mobs(&self) -> Vec<[i32; MOB_INFO_SIZE as usize]> {
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

    /// Produces a 2D array of loot information: x, y, content
    /// The array is sorted by manhattan distance from the player
    /// x and y are player-relative
    /// content (1: arrow, 2: other loot, 3: arrow and other loot). content -1 for no loot
    pub fn make_loot_array(&self) -> [[i32; LOOT_INFO_SIZE as usize]; MAX_MOBS as usize] {
        let mut loot = [[0, 0, -1]; MAX_MOBS as usize];
        let player_dist_cmp = |a: &RelativePos, b: &RelativePos| {
            let dist_a = a.0.abs() + a.1.abs();
            let dist_b = b.0.abs() + b.1.abs();
            dist_a.cmp(&dist_b)
        };
        let mut loot_indices = self.field.loot_indices(&self.player);
        loot_indices.sort_by(player_dist_cmp);
        let mut arrow_indices = self.field.arrow_indices(&self.player);
        arrow_indices.sort_by(player_dist_cmp);
        let mut min_empty_loot_position = 0;
        // record loot and loot+arrow positions
        for i in 0..min(loot_indices.len(), MAX_MOBS as usize) {
            let idx = loot_indices[i];
            if arrow_indices.contains(&idx) {
                loot[i] = [idx.0, idx.1, 3];
                let index_accounted_for = arrow_indices.iter().position(|x| *x == idx).unwrap();
                arrow_indices.remove(index_accounted_for);
            } else {
                loot[i] = [idx.0, idx.1, 2];
            }
            min_empty_loot_position = i + 1;
        }
        // in the remaining slots, record arrow-only positions
        for i in 0..min(arrow_indices.len(), MAX_MOBS as usize - min_empty_loot_position) {
            let idx = arrow_indices[i];
            loot[min_empty_loot_position + i] = [idx.0, idx.1, 1];
        }
        loot
    }
    
    pub fn is_done(&self) -> bool {
        is_game_over(&self.player)
    }
    
    pub fn reset(&mut self) {
        let (field, player, replay, milestone_tracker) = game_logic::init_game();
        self.field = field;
        self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
    }
    
    pub fn can_take_action(&self, action: i32) -> bool {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        can_take_action(&action, &self.player, &self.field)
    }
    
    pub fn reset_to_saved(&mut self, save_name: String) {
        let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
        path.push(&save_name);
        let (field, player, replay, milestone_tracker) = 
            load_game(path.as_path()).unwrap_or_else(|_| {
                println!("Could not load game {}. Initializing a new game instead", save_name);
                game_logic::init_game()
            });
        self.field = field;
        self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
    }
}
