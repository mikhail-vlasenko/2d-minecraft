use minecraft::input_decoding::act;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::player::Player;
use winit::event::VirtualKeyCode;
use winit::event::VirtualKeyCode::*;
use minecraft::crafting::consumable::Consumable;
use minecraft::crafting::material::Material;
use minecraft::crafting::storable::Storable;


pub struct Data {
    pub field: Field,
    pub player: Player
}

impl Data {
    pub fn new() -> Self {
        let test_chunk = Chunk::from(read_file(String::from("res/chunks/test_chunk.txt")));

        // Mobs dont step here, and the 0, 0 chunk doesnt spawn a mob
        let field = Field::new(8, Some(test_chunk));
        let player = Player::new(&field);
        Self {
            field,
            player
        }
    }

    pub fn act(&mut self, key: VirtualKeyCode) {
        act(&Some(key), &mut self.player, &mut self.field);
    }
    pub fn mine(&mut self) {
        self.act(Q)
    }
    pub fn place(&mut self, m: Material) {
        self.player.placement_material = m;
        self.act(E)
    }
    pub fn craft(&mut self, s: Storable) {
        self.player.crafting_item = s;
        self.act(C)
    }
    pub fn consume(&mut self, consumable: Consumable) {
        self.player.consumable = consumable;
        self.act(F)
    }
    pub fn step_mobs(&mut self) {
        self.field.step_mobs(&mut self.player);
    }
    pub fn spawn_mobs(&mut self, amount: usize) {
        spawn_mobs(&mut self.field, &mut self.player, amount)
    }
}

fn spawn_mobs(field: &mut Field, player: &mut Player, amount: usize) {
    field.spawn_mobs(player, amount)
}
