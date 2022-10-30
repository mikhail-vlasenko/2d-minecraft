use minecraft::input_decoding::act;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::player::Player;
use winit::event::VirtualKeyCode;
use winit::event::VirtualKeyCode::*;
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
        let field = Field::new(Some(test_chunk));
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
}
