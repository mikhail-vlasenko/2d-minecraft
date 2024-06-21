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
use minecraft::character::player::Player;
use egui_winit::winit::keyboard::KeyCode::*;
use minecraft::crafting::consumable::Consumable::RawMeat;
use minecraft::crafting::ranged_weapon::RangedWeapon::Bow;
use minecraft::crafting::storable::Storable;
use minecraft::map_generation::mobs::mob::{Mob, Position};
use minecraft::map_generation::mobs::mob_kind::MobKind::{Cow, Zergling, Zombie};


#[test]
fn test_zombie_hits() {
    let mut data = Data::new();
    let pos = Position {
        x: 3,
        y: 2,
        z: data.field.len_at((3, 2)),
    };
    let kind = Zombie;
    let mob = Mob::new(pos, kind);
    data.field.get_chunk(3, 2).add_mob(mob);

    let init_hp = data.player.get_hp();
    for _ in 0..10 {
        data.step_mobs();
    }

    assert!(data.player.get_hp() < init_hp);
}

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
    data.act(KeyS);
    data.act(KeyS);
    data.act(KeyS);
    data.act(KeyS);
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

    data.act(ArrowRight);
    data.act(ArrowRight);

    // cant shoot
    data.act(KeyX);
    data.act(KeyX);
    assert!(data.field.is_occupied((2, 0)));

    data.player.pickup(RW(Bow), 1);
    // no ammo
    data.act(KeyX);
    data.act(KeyX);
    assert!(data.field.is_occupied((2, 0)));

    data.player.pickup(Storable::I(Arrow), 2);
    data.act(KeyX);
    data.act(KeyX);
    // killed
    assert!(!data.field.is_occupied((2, 0)));

    let mob = Mob::new(pos.clone(), kind);
    data.field.get_chunk(0, 2).add_mob(mob);
    // out of ammo
    data.act(KeyX);
    data.act(KeyX);
    assert!(data.field.is_occupied((2, 0)));
}

#[test]
fn test_one_ling_doesnt_engage() {
    let mut data = Data::new();
    let pos = Position {
        x: 3,
        y: 2,
        z: data.field.len_at((3, 2)),
    };
    let kind = Zergling;
    let mob = Mob::new(pos, kind);
    data.field.get_chunk(3, 2).add_mob(mob);

    let init_hp = data.player.get_hp();
    for _ in 0..10 {
        data.step_mobs();
    }

    assert_eq!(data.player.get_hp(), init_hp);
}

#[test]
fn test_three_lings_engage() {
    let mut data = Data::new();

    // clear tile with tree
    data.act(ArrowRight);
    data.mine();
    data.mine();

    let pos = Position {
        x: 3,
        y: 2,
        z: data.field.len_at((3, 2)),
    };
    let kind = Zergling;
    let mob = Mob::new(pos, kind);
    data.field.get_chunk(3, 2).add_mob(mob);

    let pos = Position {
        x: -3,
        y: 0,
        z: data.field.len_at((-3, 0)),
    };
    let mob = Mob::new(pos, kind);
    data.field.get_chunk(-3, 0).add_mob(mob);

    let pos = Position {
        x: -1,
        y: -4,
        z: data.field.len_at((-1, -4)),
    };
    let mob = Mob::new(pos, kind);
    data.field.get_chunk(-1, -4).add_mob(mob);

    let init_hp = data.player.get_hp();
    for _ in 0..10 {
        if data.player.get_hp() <= 0 {
            break;
        }
        data.step_mobs();
    }

    assert!(data.player.get_hp() < init_hp);

    // check that three tiles around the player are occupied (means 3 lings engaged)
    let mut occupied = 0;
    for x in -2..=2 {
        for y in -2..=2 {
            if x == 0 && y == 0 {
                continue;
            }
            if data.field.is_occupied((x, y)) {
                occupied += 1;
            }
        }
    }
    assert_eq!(occupied, 3);
}
