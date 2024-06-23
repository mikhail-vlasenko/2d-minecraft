use std::cell::RefCell;
use rand::Rng;
use minecraft::input_decoding::act;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::character::player::Player;
use egui_winit::winit::keyboard::KeyCode;
use egui_winit::winit::keyboard::KeyCode::*;
use minecraft::crafting::consumable::Consumable;
use minecraft::crafting::storable::Storable;


pub struct Data {
    pub field: Field,
    pub player: Player
}

impl Data {
    pub fn create_with_chunk(test_chunk: Chunk) -> Self {
        // Mobs dont step here, and the 0, 0 chunk doesnt spawn a mob
        let field = Field::new(8, Some(test_chunk));
        let player = Player::new(&field);
        Self {
            field,
            player
        }
    }
    pub fn new() -> Self {
        let test_chunk = Chunk::from(read_file(String::from("res/chunks/test_chunk.txt")));
        Self::create_with_chunk(test_chunk)
    }

    pub fn maze() -> Self {
        let test_chunk = Chunk::from(read_file(String::from("res/chunks/maze_chunk.txt")));
        Self::create_with_chunk(test_chunk)
    }

    pub fn act(&mut self, key: KeyCode) -> f32 {
        act(&key, &mut self.player, &mut self.field,
            &RefCell::new(false), &RefCell::new(false))
    }
    pub fn mine(&mut self) {
        self.act(KeyQ);
    }
    pub fn place(&mut self, m: Storable) {
        self.player.placement_storable = m;
        self.act(KeyE);
    }
    pub fn craft(&mut self, s: Storable) {
        self.player.crafting_item = s;
        self.act(KeyC);
    }
    pub fn consume(&mut self, consumable: Consumable) {
        self.player.consumable = consumable;
        self.act(KeyF);
    }
    pub fn step_mobs(&mut self) {
        self.field.step_mobs(&mut self.player);
    }
    pub fn spawn_mobs(&mut self, amount: i32, hostile: bool) {
        spawn_mobs(&mut self.field, &mut self.player, amount, hostile)
    }
    pub fn step_time(&mut self) {
        self.field.step_time(1., &mut self.player);
    }
    pub fn step_interactables(&mut self, turns: i32) {
        for _ in 0..turns {
            self.field.step_interactables(&mut self.player);
        }
    }
    
    pub fn random_action(&mut self) {
        let mut rng = rand::thread_rng();
        let mut actions = vec![KeyW, KeyA, KeyS, KeyD, KeyQ, KeyE, KeyC, KeyF];
        let key = actions.remove(rng.gen_range(0..actions.len()));
        self.act(key);
    }
}

fn spawn_mobs(field: &mut Field, player: &mut Player, amount: i32, hostile: bool) {
    field.spawn_mobs(player, amount, hostile)
}
