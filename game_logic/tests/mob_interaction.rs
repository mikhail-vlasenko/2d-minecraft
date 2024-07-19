mod common;
use crate::common::Data;

use game_logic::crafting::items::Item::*;
use game_logic::crafting::storable::Storable::*;
use game_logic::crafting::consumable::Consumable::RawMeat;
use game_logic::crafting::ranged_weapon::RangedWeapon::Bow;
use game_logic::crafting::storable::Storable;
use game_logic::map_generation::mobs::mob::{Mob, Position};
use game_logic::map_generation::mobs::mob_kind::MobKind::{Cow, Zergling, Zombie};
use game_logic::auxiliary::actions::Action::*;

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
    data.act(WalkSouth);
    data.act(WalkSouth);
    data.act(WalkSouth);
    data.act(WalkSouth);
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

    data.act(TurnRight);
    data.act(TurnRight);

    // can't shoot
    data.act(Shoot);
    data.act(Shoot);
    assert!(data.field.is_occupied((2, 0)));

    data.player.pickup(RW(Bow), 1);
    // no ammo
    data.act(Shoot);
    data.act(Shoot);
    assert!(data.field.is_occupied((2, 0)));

    data.player.pickup(Storable::I(Arrow), 2);
    data.act(Shoot);
    data.act(Shoot);
    // killed
    assert!(!data.field.is_occupied((2, 0)));

    let mob = Mob::new(pos.clone(), kind);
    data.field.get_chunk(0, 2).add_mob(mob);
    // out of ammo
    data.act(Shoot);
    data.act(Shoot);
    assert!(data.field.is_occupied((2, 0)));
}
