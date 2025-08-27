use std::cell::RefCell;
use std::path::{Path, PathBuf};
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::auxiliary::actions::{Action, can_take_action};
use crate::character::player::Player;
use crate::SETTINGS;
use crate::auxiliary::animations::AnimationManager;
use crate::auxiliary::replay::Replay;
use crate::character::milestones::MilestoneTracker;
use crate::character::p2p_interactions::{apply_p2p_interactions, P2PInteraction};
use crate::character::start_loadouts::apply_loadout;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::field::{AbsolutePos, Field};
use crate::map_generation::mobs::mob::Position;
use crate::map_generation::read_chunk::read_file;
use crate::map_generation::save_load::{autosave_game, load_game, save_game};
use crate::perform_action::act;

#[derive(Debug)]
pub struct GameState {
    pub field: Field,
    pub player_register: PlayerRegister,
    pub recorded_replay: Replay,
    pub milestone_tracker: MilestoneTracker,
    pub animation_manager: AnimationManager,
    primary_player_id: u32,  // index of the primary player - the rendered one
    p2p_interactions_buffer: Vec<(P2PInteraction, AbsolutePos)>
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
    field.spawn_mobs(&vec![player.get_position().clone()], amount, true);
    field.spawn_mobs(&vec![player.get_position().clone()], amount * 2, false);

    let replay = Replay::new();
    let milestone_tracker = MilestoneTracker::new();

    (field, player, replay, milestone_tracker)
}

impl GameState {
    pub fn new() -> Self {
        let (mut field, player, recorded_replay, milestone_tracker) = init_game();
        let primary_player_id = player.get_id();
        // add more players
        let mut players = vec![player];
        for _ in 1..SETTINGS.read().unwrap().multiplayer.num_players {
            let x = rand::thread_rng().gen_range(-10..=10);
            let y = rand::thread_rng().gen_range(-10..=10);
            let mut new_player = Player::new_at(&field, (x, y));
            apply_loadout(&mut new_player, &mut field);
            players.push(new_player);
        }
        let player_register = PlayerRegister::new(players);
        let animation_manager = AnimationManager::new();
        let p2p_interactions_buffer = Vec::new();

        Self {
            field,
            player_register,
            // will not be recorded if record_replays is disabled in settings
            recorded_replay,
            milestone_tracker,
            animation_manager,
            primary_player_id,
            p2p_interactions_buffer,
        }
    }

    pub fn enable_animations(&mut self) {
        self.animation_manager.set_enabled(true);
    }

    pub fn step(&mut self, player_id: u32, action: &Action, main_menu_open: &RefCell<bool>, craft_menu_open: &RefCell<bool>) {
        let mut player = self.player_register.borrow_player(player_id).unwrap();
        let other_positions = self.player_register.get_player_positions();

        if action == &Action::ToggleMainMenu || (!player.is_done() && !*main_menu_open.borrow()) {
            player.reset_message();
            // different actions take different time, so sometimes mobs are not allowed to step
            let (passed_time, interactions) = act(
                action,
                &mut player,
                &mut self.field,
                &other_positions,
                craft_menu_open,
                main_menu_open
            );
            self.extend_p2p_interactions(interactions);
            player.add_time(passed_time);
            let completed_milestone = self.milestone_tracker.check_milestones(&player, self.field.get_time());
            if completed_milestone && SETTINGS.read().unwrap().save_on_milestone {
                player.add_message(&format!("Milestone completed: {}", self.milestone_tracker.get_current_milestone_idx()));
                let save_name = self.autosave_game();
                player.add_message(&format!("Game saved as: {}", save_name));
            }

            if self.animation_manager.is_enabled() {
                if passed_time > 0. {
                    // drop continuous animations if player made an action
                    self.animation_manager.drop_continuous_animations();
                    self.animation_manager.add_channeling_animations(&self.field, player.xy());
                }
                self.animation_manager.absorb_buffer(&mut self.field.animations_buffer);
                self.animation_manager.absorb_buffer(&mut player.animations_buffer);
            } else {
                self.field.animations_buffer.clear();
                player.animations_buffer.clear();
            }
            if SETTINGS.read().unwrap().record_replays {
                self.recorded_replay.record_state(&self.field, &player);
            }
        }

        // add back the player
        self.player_register.return_player(player);

        self.apply_p2p_interactions();

        while !self.is_done_game() && self.player_register.min_player_time() + 1.0 >= self.field.get_time() {
            let player_positions = self.player_register.get_player_positions();
            let inters = self.field.step_time(1.0, &player_positions);
            self.extend_p2p_interactions(inters);
            self.apply_p2p_interactions();
        }

        if self.is_done_primary() {
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

    pub fn step_i32(&mut self, player_id: u32, action: i32) {
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        if !action.ffi_disabled() {
            self.step(player_id, &action, &RefCell::new(false), &RefCell::new(false));
        }
    }

    pub fn is_done_primary(&self) -> bool {
        let player = self.player_register.get_by_id(self.primary_player_id);
        match player {
            None => true,
            Some(p) => p.is_done(),
        }
    }

    pub fn is_done_game(&self) -> bool {
        self.player_register.num_current_players() == 0
    }

    pub fn can_take_action(&self, action: i32) -> bool {
        // todo: take id as argument
        let action = Action::try_from(action).expect(format!("Invalid action with index {}", action).as_str());
        can_take_action(&action, self.player_register.get_by_id(self.primary_player_id).unwrap(), &self.field)
    }

    pub fn get_time(&self) -> f32 {
        self.field.get_time()
    }

    pub fn reset(&mut self) {
        *self = GameState::new();
    }

    // todo: fix save/load logic
    pub fn reset_to_saved(&mut self, save_name: String) {
        let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
        path.push(&save_name);
        let (field, player, replay, milestone_tracker) =
            load_game(path.as_path()).unwrap_or_else(|_| {
                println!("Could not load game {}. Initializing a new game instead", save_name);
                init_game()
            });
        self.field = field;
        // self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
    }

    pub fn save_game(&mut self, path: &Path) {
        // save_game(&self.field, &self.player, &self.milestone_tracker, path);
    }

    pub fn autosave_game(&mut self) -> String {
        // autosave_game(&self.field, &self.player, &self.milestone_tracker)
        String::new()
    }

    pub fn load_game(&mut self, path: &Path) {
        let (field, player, replay, milestone_tracker) = load_game(path).unwrap();
        self.field = field;
        // self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
        self.animation_manager.clear();
    }

    pub fn get_primary_player_id(&self) -> u32 {
        self.primary_player_id
    }

    pub fn get_id_next_to_move(&self) -> Option<u32> {
        // player Ord is based on time, so the one with the lowest time is next to move
        let player = self.player_register.players.iter().min();
        if player.is_none() {
            return None;
        }
        Some(player.unwrap().get_id())
    }


    pub fn get_primary_player(&self) -> &Player {
        self.player_register.get_by_id(self.primary_player_id).unwrap()
    }

    pub fn borrow_primary_player(&mut self) -> Player {
        self.player_register.borrow_player(self.primary_player_id).unwrap()
    }

    pub fn return_primary_player(&mut self, player: Player) {
        assert_eq!(player.get_id(), self.primary_player_id, "Returning a player with wrong id");
        self.player_register.return_player(player);
    }

    pub fn get_all_rendering_info(&self) -> Vec<(AbsolutePos, u32)> {
        self.player_register.get_all_rendering_info()
    }

    pub fn extend_p2p_interactions(&mut self, interactions: Vec<(P2PInteraction, AbsolutePos)>) {
        self.p2p_interactions_buffer.extend(interactions);
    }

    pub fn apply_p2p_interactions(&mut self) {
        let players = self.player_register.take_players();
        let interactions = std::mem::take(&mut self.p2p_interactions_buffer);
        let players= apply_p2p_interactions(interactions, players);
        self.player_register.set_players(players);
    }
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct PlayerRegister {
    players: Vec<Player>,
}

impl PlayerRegister {
    pub fn new(players: Vec<Player>) -> PlayerRegister {
        Self { players }
    }

    pub fn take_players(&mut self) -> Vec<Player> {
        std::mem::take(&mut self.players)
    }

    pub fn set_players(&mut self, players: Vec<Player>) {
        self.players = players.into_iter().filter(|player| !player.is_done()).collect();
    }

    /// Removes the player with the given id from the register and returns it.
    /// The player can then be freely mutated.
    pub fn borrow_player(&mut self, id: u32) -> Option<Player> {
        let idx = self.players.iter().position(|player| player.get_id() == id);
        match idx {
            None => None,
            Some(i) => Some(self.players.swap_remove(i)) // pop and replace with the last element (order does not matter)
        }
    }

    /// After the player is done being mutated, it should be returned to the register.
    pub fn return_player(&mut self, player: Player) {
        self.players.push(player);
    }

    pub fn get_by_id(&self, id: u32) -> Option<&Player> {
        self.players.iter().find(|player| player.get_id() == id)
    }

    pub fn get_player_positions(&self) -> Vec<Position> {
        self.players.iter().map(|player| player.get_position().clone()).collect()
    }

    pub fn get_all_rendering_info(&self) -> Vec<((i32, i32), u32)> {
        self.players.iter().map(|player| (player.xy(), player.get_rotation())).collect()
    }

    pub fn num_current_players(&self) -> usize {
        self.players.len()
    }

    pub fn min_player_time(&self) -> f32 {
        self.players.iter().map(|p| p.get_time()).fold(f32::INFINITY, |a, b| a.min(b))
    }
}
