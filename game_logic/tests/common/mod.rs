use std::cell::RefCell;
use rand::Rng;
use game_logic::auxiliary::actions::Action;
use game_logic::perform_action::act;
use game_logic::map_generation::chunk::Chunk;
use game_logic::map_generation::field::Field;
use game_logic::map_generation::read_chunk::read_file;
use game_logic::character::player::Player;
use game_logic::crafting::consumable::Consumable;
use game_logic::crafting::storable::Storable;

#[cfg(test)]
#[allow(dead_code)]
pub struct Data {
    pub field: Field,
    pub player: Player
}

#[cfg(test)]
#[allow(dead_code)]
impl Data {
    pub fn create_with_chunk(test_chunk: Chunk) -> Self {
        // Mobs don't step here, and the 0, 0 chunk doesn't spawn a mob
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

    pub fn act(&mut self, action: Action) -> f32 {
        act(action, &mut self.player, &mut self.field,
            &RefCell::new(false), &RefCell::new(false))
    }

    pub fn mine(&mut self) {
        self.act(Action::Mine);
    }

    pub fn place(&mut self, m: Storable) {
        self.player.placement_storable = m;
        self.act(Action::Place);
    }

    pub fn craft(&mut self, s: Storable) {
        self.player.crafting_item = s;
        self.act(Action::Craft);
    }

    pub fn consume(&mut self, consumable: Consumable) {
        self.player.consumable = consumable;
        self.act(Action::Consume);
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
        let actions = vec![
            Action::WalkNorth, Action::WalkWest, Action::WalkSouth, Action::WalkEast,
            Action::Mine, Action::Place, Action::Craft, Action::Consume
        ];
        let action = actions[rng.gen_range(0..actions.len())];
        self.act(action);
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn spawn_mobs(field: &mut Field, player: &mut Player, amount: i32, hostile: bool) {
    field.spawn_mobs(player, amount, hostile)
}
