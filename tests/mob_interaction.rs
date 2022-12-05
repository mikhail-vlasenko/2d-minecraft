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
use minecraft::crafting::consumable::Consumable::RawMeat;

#[test]
fn test_mobs_get_you_eventually() {
    let mut data = Data::new();
    data.spawn_mobs(20, true);

    assert!(data.player.get_hp() > 0);
    for _ in 0..100 {
        data.step_mobs();
    }
    assert!(data.player.get_hp() <= 0);
}

#[test]
fn test_healing() {
    let mut data = Data::new();

    let init_hp = data.player.get_hp();
    data.player.pickup(Storable::C(RawMeat), 1);

    data.player.receive_damage(10);
    assert!(data.player.get_hp() < init_hp);

    data.consume(RawMeat);
    assert_eq!(data.player.get_hp(), init_hp);
}
