use std::cell::RefCell;
use std::path::{Path, PathBuf};
use crate::auxiliary::actions::{Action, can_take_action};
use crate::character::player::Player;
use crate::SETTINGS;
use crate::auxiliary::animations::AnimationManager;
use crate::auxiliary::replay::Replay;
use crate::character::milestones::MilestoneTracker;
use crate::character::start_loadouts::apply_loadout;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::field::{Field};
use crate::map_generation::read_chunk::read_file;
use crate::map_generation::save_load::{autosave_game, load_game, save_game};
use crate::perform_action::act;

#[derive(Debug)]
pub struct GameState {
    pub field: Field,
    pub player: Player,
    pub recorded_replay: Replay,
    pub milestone_tracker: MilestoneTracker,
    pub animation_manager: AnimationManager,
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

impl GameState {
    pub fn new() -> Self {
        let (field, player, recorded_replay, milestone_tracker) = init_game();
        let animation_manager = AnimationManager::new();

        Self {
            field,
            player,
            // will not be recorded if record_replays is disabled in settings
            recorded_replay,
            milestone_tracker,
            animation_manager
        }
    }
    
    pub fn enable_animations(&mut self) {
        self.animation_manager.set_enabled(true);
    }

    pub fn step(&mut self, action: &Action, main_menu_open: &RefCell<bool>, craft_menu_open: &RefCell<bool>) {
        if action == &Action::ToggleMainMenu || (!self.is_done() && !*main_menu_open.borrow()) {
            self.player.reset_message();
            // different actions take different time, so sometimes mobs are not allowed to step
            let passed_time = act(
                action,
                &mut self.player, &mut self.field,
                craft_menu_open,
                main_menu_open
            );
            self.field.step_time(passed_time, &mut self.player);
            let completed_milestone = self.milestone_tracker.check_milestones(&self.player, self.field.get_time());
            if completed_milestone && SETTINGS.read().unwrap().save_on_milestone {
                self.player.add_message(&format!("Milestone completed: {}", self.milestone_tracker.get_current_milestone_idx()));
                let save_name = self.autosave_game();
                self.player.add_message(&format!("Game saved as: {}", save_name));
            }

            if self.animation_manager.is_enabled() {
                if passed_time > 0. {
                    // drop continuous animations if player made an action
                    self.animation_manager.drop_continuous_animations();
                    self.animation_manager.add_channeling_animations(&self.field, self.player.xy());
                }
                self.animation_manager.absorb_buffer(&mut self.field.animations_buffer);
                self.animation_manager.absorb_buffer(&mut self.player.animations_buffer);
            } else {
                self.field.animations_buffer.clear();
                self.player.animations_buffer.clear();
            }
            if SETTINGS.read().unwrap().record_replays {
                self.recorded_replay.record_state(&self.field, &self.player);
            }
        }
        if self.is_done() {
            // save the replay
            if SETTINGS.read().unwrap().record_replays && !self.recorded_replay.is_empty() {
                let path = PathBuf::from(SETTINGS.read().unwrap().replay_folder.clone().into_owned());
                let name = self.recorded_replay.make_save_name().unwrap();  // unwrap is safe here because record_replays
                let path = path.join(name.clone());
                self.recorded_replay.save(path.as_path());
                self.recorded_replay = Replay::new();
            }
        }
    }

    pub fn step_i32(&mut self, action: i32) {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        if !action.ffi_disabled() {
            self.step(&action, &RefCell::new(false), &RefCell::new(false));
        }
    }

    pub fn is_done(&self) -> bool {
        self.player.get_hp() <= 0
    }

    pub fn can_take_action(&self, action: i32) -> bool {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        can_take_action(&action, &self.player, &self.field)
    }

    pub fn reset(&mut self) {
        let (field, player, replay, milestone_tracker) = init_game();
        self.field = field;
        self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
    }

    pub fn reset_to_saved(&mut self, save_name: String) {
        let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
        path.push(&save_name);
        let (field, player, replay, milestone_tracker) =
            load_game(path.as_path()).unwrap_or_else(|_| {
                println!("Could not load game {}. Initializing a new game instead", save_name);
                init_game()
            });
        self.field = field;
        self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
    }

    pub fn get_time(&self) -> f32 {
        self.field.get_time()
    }

    pub fn save_game(&mut self, path: &Path) {
        save_game(&self.field, &self.player, &self.milestone_tracker, path);
    }
    
    pub fn autosave_game(&mut self) -> String {
        autosave_game(&self.field, &self.player, &self.milestone_tracker)
    }
    
    pub fn load_game(&mut self, path: &Path) {
        let (field, player, replay, milestone_tracker) = load_game(path).unwrap();
        self.field = field;
        self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
        self.animation_manager.clear();
    }
}
