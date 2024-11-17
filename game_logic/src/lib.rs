extern crate core;

extern crate lazy_static;

use std::cell::RefCell;
use std::sync::RwLock;
use lazy_static::lazy_static;
use crate::auxiliary::actions::Action;
use crate::auxiliary::animations::AnimationManager;
use crate::auxiliary::replay::Replay;
use crate::character::milestones::MilestoneTracker;
use crate::character::player::Player;
use crate::character::start_loadouts::apply_loadout;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::field::Field;
use crate::map_generation::read_chunk::read_file;
use crate::map_generation::save_load::autosave_game;
use crate::perform_action::act;
use crate::settings::Settings;

pub mod map_generation;
pub mod crafting;
pub mod character;
pub mod auxiliary;
pub mod perform_action;
pub mod settings;

lazy_static! {
    pub static ref SETTINGS: RwLock<Settings> = RwLock::new(Settings::load().into_owned());
}

pub fn init_game() -> (Field, Player, Replay, MilestoneTracker) {
    let start_chunk=  if SETTINGS.read().unwrap().field.from_test_chunk {
        Some(Chunk::from(read_file(String::from("res/chunks/test_chunk.txt"))))
    } else {
        None
    };
    let mut field = Field::new(SETTINGS.read().unwrap().window.render_distance as usize, start_chunk);
    let mut player = Player::new(&field);
    apply_loadout(&mut player, &mut field);

    // spawn some initial mobs
    let amount = (SETTINGS.read().unwrap().mobs.spawning.initial_hostile_per_chunk *
        (field.get_loading_distance() * 2 + 1).pow(2) as f32) as i32;
    field.spawn_mobs(&player, amount, true);
    field.spawn_mobs(&player, amount * 2, false);
    
    let replay = Replay::new();
    let milestone_tracker = MilestoneTracker::new();

    (field, player, replay, milestone_tracker)
}

pub fn handle_action(action: &Action, field: &mut Field, player: &mut Player, 
                     main_menu_open: &RefCell<bool>, craft_menu_open: &RefCell<bool>, 
                     animation_manager: Option<& mut AnimationManager>, replay: &mut Replay, 
                     milestone_tracker: &mut MilestoneTracker) {
    if action == &Action::ToggleMainMenu || (!is_game_over(player) && !*main_menu_open.borrow()) {
        player.message = String::new();
        // different actions take different time, so sometimes mobs are not allowed to step
        let passed_time = act(
                action,
                player, field,
                craft_menu_open,
                main_menu_open
        );
        field.step_time(passed_time, player);
        let completed_milestone = milestone_tracker.check_milestones(player, field.get_time());
        if completed_milestone && SETTINGS.read().unwrap().save_on_milestone {
            player.add_message(&format!("Milestone completed: {}", milestone_tracker.get_current_milestone_idx()));
            let save_name = autosave_game(field, player, milestone_tracker);
            player.add_message(&format!("Game saved as: {}", save_name));
        }
        
        if let Some(animation_manager) = animation_manager {
            if passed_time > 0. {
                // drop continuous animations if player made an action
                animation_manager.drop_continuous_animations();
                animation_manager.add_channeling_animations(field, player);
            }
            animation_manager.absorb_buffer(&mut field.animations_buffer);
            animation_manager.absorb_buffer(&mut player.animations_buffer);
        } else { 
            field.animations_buffer.clear();
            player.animations_buffer.clear();
        }
        if SETTINGS.read().unwrap().record_replays {
            replay.record_state(field, player);
        }
    }
}

pub fn is_game_over(player: &Player) -> bool {
    player.get_hp() <= 0
}
