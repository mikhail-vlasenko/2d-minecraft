mod common;
use crate::common::Data;

use game_logic::crafting::items::Item::*;
use game_logic::crafting::interactable::InteractableKind::CrossbowTurret;
use game_logic::map_generation::mobs::mob::{Mob, Position};
use game_logic::map_generation::mobs::mob_kind::MobKind::{Baneling, Zergling};


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
    data.field.load_interactable_at((9, 9), Arrow.into(), 10);
    data.step_interactables(1);
    // crossbow turret has speed < 1 so it does not kill immediately
    assert!(data.field.is_occupied((8, 8)));
    data.step_interactables(10);
    assert!(!data.field.is_occupied((8, 8)));
}

#[test]
fn test_placed_turret_targets_only_banes() {
    let mut data = Data::new();
    data.player.x = 10;
    data.player.y = 9;
    let pos = Position {
        x: 8,
        y: 8,
        z: 2,
    };
    let pos2 = Position {
        x: 7,
        y: 8,
        z: 2,
    };
    let mob = Mob::new(pos, Zergling);
    data.field.get_chunk(0, 0).add_mob(mob);
    assert!(data.field.is_occupied((8, 8)));
    data.player.pickup(CrossbowTurret.into(), 1);
    data.place(CrossbowTurret.into());
    data.field.load_interactable_at((9, 9), Arrow.into(), 10);
    data.field.set_interactable_targets_at((9, 9), vec![Baneling]);
    data.step_interactables(10);
    // zergling is still alive cause it is not targeted
    assert!(data.field.is_occupied((8, 8)));

    let mob = Mob::new(pos2, Baneling);
    data.field.get_chunk(0, 0).add_mob(mob);
    assert!(data.field.is_occupied((7, 8)));

    data.step_interactables(10);
    // baneling is dead
    assert!(!data.field.is_occupied((7, 8)));
}

#[test]
fn test_loading_turret() {
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
    data.step_interactables(10);
    assert!(data.field.is_occupied((8, 8)));
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
    data.field.load_interactable_at((14, 15), Arrow.into(), 10);
    data.step_interactables(10);
    assert!(data.field.is_occupied((0, 0)));
}