use std::cell::RefCell;
use std::ops::Not;
use winit::event::VirtualKeyCode;
use crate::player::Player;
use crate::map_generation::field::Field;

/// Makes an action, corresponding to the key.
/// Returns how much turn was used.
pub fn act(key: &Option<VirtualKeyCode>, player: &mut Player, field: &mut Field, craft_menu_open: &RefCell<bool>) -> f32 {
    match key {
        None => { println!("Unrecognized virtual key"); 0. },
        Some(VirtualKeyCode::W) => player.walk("w", field),
        Some(VirtualKeyCode::A) => player.walk("a", field),
        Some(VirtualKeyCode::S) => player.walk("s", field),
        Some(VirtualKeyCode::D) => player.walk("d", field),
        Some(VirtualKeyCode::Left) => player.turn(1),
        Some(VirtualKeyCode::Right) => player.turn(-1),
        Some(VirtualKeyCode::Q) => player.mine_infront(field),
        Some(VirtualKeyCode::M) => player.mine_infront(field),
        Some(VirtualKeyCode::E) => player.place_current(field),
        Some(VirtualKeyCode::P) => player.place_current(field),
        Some(VirtualKeyCode::C) => player.craft_current(field),
        Some(VirtualKeyCode::F) => player.consume_current(),
        Some(VirtualKeyCode::X) => player.shoot_current(field),
        Some(VirtualKeyCode::Space) => { craft_menu_open.replace(!craft_menu_open.take()); 0. },
        _ => { println!("Unknown action"); 0. }
    }
}