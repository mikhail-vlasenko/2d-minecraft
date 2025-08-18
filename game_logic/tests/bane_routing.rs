mod common;
use crate::common::Data;

use game_logic::map_generation::mobs::mob::{Mob, Position};
use game_logic::map_generation::mobs::mob_kind::MobKind::{Baneling};


#[test]
fn test_horizontal_wall_explode() {
    let mut data = Data::maze();
    data.place_player((2, 3));
    let pos = Position {
        x: 0,
        y: 3,
        z: 2,
    };
    let mob = Mob::new(pos, Baneling);
    data.field.get_chunk(0, 0).add_mob(mob);

    assert_eq!(data.field.len_at((1, 3)), 5);
    data.step_mobs();
    data.step_mobs();
    assert!(data.field.len_at((1, 3)) < 5)
}

#[test]
fn test_vertical_wall_explode() {
    let mut data = Data::maze();
    data.place_player((3, 13));
    let pos = Position {
        x: 3,
        y: 15,
        z: 2,
    };
    let mob = Mob::new(pos, Baneling);
    data.field.get_chunk(0, 0).add_mob(mob);

    assert_eq!(data.field.len_at((3, 14)), 5);
    data.step_mobs();
    data.step_mobs();
    assert!(data.field.len_at((3, 14)) < 5)
}

#[test]
fn test_shorter_horizontal_wall_explode() {
    let mut data = Data::maze();
    data.place_player((2, 2));
    let pos = Position {
        x: 0,
        y: 2,
        z: 2,
    };
    let mob = Mob::new(pos, Baneling);
    data.field.get_chunk(0, 0).add_mob(mob);

    assert_eq!(data.field.len_at((1, 2)), 5);
    data.step_mobs();
    data.step_mobs();
    assert!(data.field.len_at((1, 2)) < 5)
}

#[test]
fn test_horizontal_wall_go_around() {
    let mut data = Data::maze();
    data.place_player((2, 1));
    let pos = Position {
        x: 0,
        y: 1,
        z: 2,
    };
    let mob = Mob::new(pos, Baneling);
    data.field.get_chunk(0, 0).add_mob(mob);

    assert_eq!(data.field.len_at((1, 1)), 5);
    data.step_mobs();
    data.step_mobs();
    assert_eq!(data.field.len_at((1, 1)), 5)
}

#[test]
fn test_banes_kill() {
    let mut data = Data::maze();

    for i in 0..10 {
        let pos = Position { x: i, y: 2, z: 2 };
        let mob = Mob::new(pos, Baneling);
        data.field.get_chunk(i, 0).add_mob(mob);
    }
    for _ in 0..100 {
        data.step_mobs();
        if data.player.get_hp() <= 0 {
            break;
        }
    }
    assert!(data.player.get_hp() <= 0);
}

#[test]
fn test_banes_bust_and_kill() {
    let mut data = Data::maze();
    data.place_player((5, 6));
    for i in 0..20 {
        let pos = Position { x: 4, y: -i, z: 2 };
        let mob = Mob::new(pos, Baneling);
        data.field.get_chunk(i, 0).add_mob(mob);
    }
    for i in 0..20 {
        let pos = Position { x: 8, y: -i, z: 2 };
        let mob = Mob::new(pos, Baneling);
        data.field.get_chunk(i, 0).add_mob(mob);
    }
    for _ in 0..100 {
        data.step_mobs();
        if data.player.get_hp() <= 0 {
            break;
        }
    }
    assert!(data.player.get_hp() <= 0);
}
