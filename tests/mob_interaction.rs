mod common;
use crate::common::Data;

use minecraft::crafting::items::Item;
use minecraft::crafting::items::Item::*;
use minecraft::crafting::material::Material;
use minecraft::crafting::material::Material::*;
use minecraft::crafting::storable::Storable;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::player::Player;
use winit::event::VirtualKeyCode::*;

#[test]
fn test_mobs_get_you_eventually() {
    let mut data = Data::new();

    assert!(data.player.get_hp() > 0);
    for _ in 0..100 {
        data.step_mobs();
    }
    assert!(data.player.get_hp() <= 0);
}
