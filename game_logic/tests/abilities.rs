mod common;

use game_logic::auxiliary::actions::Action;
use game_logic::character::abilities::Ability::WhirlAttack;
use game_logic::map_generation::mobs::mob::{Mob, Position};
use game_logic::map_generation::mobs::mob_kind::MobKind::Zombie;
use crate::common::Data;


#[test]
fn whirlwind_hits() {
    let mut data = Data::new();
    data.place_player((1, 1));
    let kind = Zombie;

    let pos = Position::new((0, 0), &mut data.field);
    let mob = Mob::new(pos, kind);
    data.field.place_mob(mob);

    let pos = Position::new((2, 2), &mut data.field);
    let mob = Mob::new(pos, kind);
    data.field.place_mob(mob);

    let pos = Position::new((1, 2), &mut data.field);
    let mob = Mob::new(pos, kind);
    data.field.place_mob(mob);

    let mob_hp = data.field.close_mob_info(|mob| {
        mob.get_hp_share()},
        &data.player,
    );
    for hp in mob_hp {
        assert_eq!(hp, 1.0);
    }
    data.act(Action::UseAbility(WhirlAttack));
    let mobs = data.field.close_mob_info(
        |mob| {
            let pos = (mob.pos.x, mob.pos.y);
            (pos, mob.get_hp_share())
        },
        &data.player,
    );
    for (pos, hp) in mobs {
        if pos == (1, 2) {
            assert_eq!(hp, 1.0, "Mob is too high to get hit");
        } else if (pos == (2, 2)) | (pos == (0, 0)) {
            assert!(hp < 1.0, "Whirlwind did not hit mob at position {:?}", pos);
        } else {
            panic!("Unexpected mob at position {:?} with hp {}", pos, hp);
        }
    }
}

#[test]
fn abilities_cooldown() {
    let mut data = Data::new();
    data.place_player((1, 1));
    data.act(Action::UseAbility(WhirlAttack));
    data.step_time();
    assert!(!data.player.is_ability_ready(&WhirlAttack));
    
    let kind = Zombie;
    let pos = Position::new((0, 0), &mut data.field);
    let mob = Mob::new(pos, kind);
    data.field.place_mob(mob);

    data.act(Action::UseAbility(WhirlAttack));
    let mobs = data.field.close_mob_info(
        |mob| {
            let pos = (mob.pos.x, mob.pos.y);
            (pos, mob.get_hp_share())
        },
        &data.player,
    );
    for (pos, hp) in mobs {
        if pos == (0, 0) {
            assert_eq!(hp, 1.0, "Ability on cooldown");
        } else {
            panic!("Unexpected mob at position {:?} with hp {}", pos, hp);
        }
    }
}
