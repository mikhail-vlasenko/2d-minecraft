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
use winit::event::VirtualKeyCode::*;
use minecraft::crafting::consumable::Consumable::RawMeat;
use minecraft::crafting::interactable::InteractableKind;
use minecraft::crafting::interactable::InteractableKind::CrossbowTurret;
use minecraft::crafting::ranged_weapon::RangedWeapon::Bow;
use minecraft::crafting::storable::Storable;
use minecraft::map_generation::mobs::mob::{Mob, Position};
use minecraft::map_generation::mobs::mob_kind::MobKind::{Cow, Zergling, Zombie};


#[test]
fn test_placed_turret_kills() {
    let mut data = Data::new();
    data.player.x = 10;
    data.player.y = 9;
    let pos = Position {
        x: 8,
        y: 8,
        z: 2,
    };
    let mob = Mob::new(pos, Zergling);
    data.field.get_chunk(0, 0).add_mob(mob);
    assert!(data.field.is_occupied((8, 8)));
    data.player.pickup(CrossbowTurret.into(), 1);
    data.place(CrossbowTurret.into());
    // load it (later)
    data.step_interactables();
    assert!(!data.field.is_occupied((8, 8)));
}

#[test]
fn test_turret_limited_range() {
    let mut data = Data::new();
    data.player.x = 15;
    data.player.y = 15;
    let pos = Position {
        x: 0,
        y: 0,
        z: 2,
    };
    let mob = Mob::new(pos, Zergling);
    data.field.get_chunk(0, 0).add_mob(mob);
    data.player.pickup(CrossbowTurret.into(), 1);
    data.place(CrossbowTurret.into());
    // load it (later)
    data.step_interactables();
    assert!(data.field.is_occupied((0, 0)));
}