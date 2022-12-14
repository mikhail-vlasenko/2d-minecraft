mod common;
use crate::common::Data;

use minecraft::crafting::items::Item;
use minecraft::crafting::items::Item::*;
use minecraft::crafting::material::Material;
use minecraft::crafting::material::Material::*;
use minecraft::crafting::storable::Storable::*;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::player::Player;
use winit::event::VirtualKeyCode::*;
use minecraft::crafting::consumable::Consumable::RawMeat;
use minecraft::crafting::ranged_weapon::RangedWeapon::Bow;
use minecraft::crafting::storable::Storable;
use minecraft::map_generation::mobs::mob::{Mob, Position};
use minecraft::map_generation::mobs::mob_kind::MobKind::{Cow, Zergling, Zombie};

#[test]
fn test_mobs_get_you_eventually() {
    let mut data = Data::new();
    data.spawn_mobs(50, true);

    assert!(data.player.get_hp() > 0);
    for _ in 0..200 {
        data.step_mobs();
        if data.player.get_hp() <= 0 {
            break;
        }
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

#[test]
fn test_killing_and_looting() {
    let mut data = Data::new();
    let pos = Position {
        x: 1,
        y: 0,
        z: 2,
    };
    let kind = Cow;
    let mob = Mob::new(pos, kind);
    data.field.get_chunk(0, 1).add_mob(mob);

    let init_meat = data.player.inventory_count(&Storable::C(RawMeat));
    data.act(S);
    data.act(S);
    data.act(S);
    data.act(S);
    assert!(data.player.inventory_count(&Storable::C(RawMeat)) > init_meat)
}

#[test]
fn test_shooting() {
    let mut data = Data::new();
    let pos = Position {
        x: 2,
        y: 0,
        z: 2,
    };
    let kind = Zergling;
    let mob = Mob::new(pos.clone(), kind);
    data.field.get_chunk(0, 2).add_mob(mob);

    data.act(Right);
    data.act(Right);

    // cant shoot
    data.act(X);
    data.act(X);
    assert!(data.field.is_occupied((2, 0)));

    data.player.pickup(RW(Bow), 1);
    // no ammo
    data.act(X);
    data.act(X);
    assert!(data.field.is_occupied((2, 0)));

    data.player.pickup(Storable::I(Arrow), 2);
    data.act(X);
    data.act(X);
    // killed
    assert!(!data.field.is_occupied((2, 0)));

    let mob = Mob::new(pos.clone(), kind);
    data.field.get_chunk(0, 2).add_mob(mob);
    // out of ammo
    data.act(X);
    data.act(X);
    assert!(data.field.is_occupied((2, 0)));
}
