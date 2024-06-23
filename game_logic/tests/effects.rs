mod common;

use game_logic::crafting::consumable::Consumable::SpeedPotion;
use crate::common::Data;
use game_logic::auxiliary::actions::Action::*;

#[test]
fn test_speed_potion_makes_you_faster() {
    let mut data = Data::new();
    data.player.pickup(SpeedPotion.into(), 1);
    let time_taken = data.act(WalkSouth);
    assert_eq!(time_taken, 1.);
    data.consume(SpeedPotion);
    let time_taken = data.act(WalkSouth);
    assert!(time_taken < 1.);
}

#[test]
fn test_speed_potion_ends() {
    let mut data = Data::new();
    data.player.pickup(SpeedPotion.into(), 1);
    data.consume(SpeedPotion);
    let time_taken = data.act(WalkSouth);
    assert!(time_taken < 1.);
    for _ in 0..100 {
        data.step_time();
    }
    let time_taken = data.act(WalkSouth);
    assert_eq!(time_taken, 1.);
}
