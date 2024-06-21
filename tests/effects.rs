mod common;

use minecraft::crafting::consumable::Consumable::SpeedPotion;
use crate::common::Data;
use egui_winit::winit::keyboard::KeyCode::*;


#[test]
fn test_speed_potion_makes_you_faster() {
    let mut data = Data::new();
    data.player.pickup(SpeedPotion.into(), 1);
    let time_taken = data.act(KeyS);
    assert_eq!(time_taken, 1.);
    data.consume(SpeedPotion);
    let time_taken = data.act(KeyS);
    assert!(time_taken < 1.);
}

#[test]
fn test_speed_potion_ends() {
    let mut data = Data::new();
    data.player.pickup(SpeedPotion.into(), 1);
    data.consume(SpeedPotion);
    let time_taken = data.act(KeyS);
    assert!(time_taken < 1.);
    for _ in 0..100 {
        data.step_time();
    }
    let time_taken = data.act(KeyS);
    assert_eq!(time_taken, 1.);
}
