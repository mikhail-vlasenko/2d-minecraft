mod common;
use crate::common::Data;

use game_logic::crafting::items::Item::*;
use game_logic::crafting::storable::Storable::*;
use game_logic::crafting::consumable::Consumable::RawMeat;
use game_logic::crafting::ranged_weapon::RangedWeapon::Bow;
use game_logic::crafting::storable::Storable;
use game_logic::map_generation::mobs::mob::{Mob, Position};
use game_logic::map_generation::mobs::mob_kind::MobKind::{Baneling, Cow, GelatinousCube, Zergling, Zombie};
use game_logic::auxiliary::actions::Action::*;


#[test]
fn test_mobs_get_you_eventually() {
    let mut data = Data::new();
    data.spawn_mobs(50, true);

    assert!(data.player.get_hp() > 0);
    for _ in 0..256 {
        data.step_mobs();
        if data.player.get_hp() <= 0 {
            break;
        }
    }
    assert!(data.player.get_hp() <= 0);
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
    data.act(TurnRight);
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

#[test]
fn test_gelatinous_cubes_jump_and_kill() {
    let mut data = Data::maze();

    let mut initial_heights = vec![];
    for i in 0..20 {
        for j in 0..20 {
            initial_heights.push(data.field.len_at((i, j)));
        }
    }

    data.player.x = 5;
    data.player.y = 6;
    for i in 0..3 {
        let pos = Position { x: 4, y: -i, z: 2 };
        let mob = Mob::new(pos, GelatinousCube);
        data.field.get_chunk(i, 0).add_mob(mob);
    }
    for i in 17..20 {
        let pos = Position { x: 8, y: -i, z: 2 };
        let mob = Mob::new(pos, GelatinousCube);
        data.field.get_chunk(i, 0).add_mob(mob);
    }
    for _ in 0..100 {
        data.step_mobs();
        if data.player.get_hp() <= 0 {
            break;
        }
    }
    assert!(data.player.get_hp() <= 0);

    // check no blocks were added or removed
    for i in 0..20 {
        for j in 0..20 {
            assert_eq!(data.field.len_at((i, j)), initial_heights[(i * 20 + j) as usize]);
        }
    }
}
