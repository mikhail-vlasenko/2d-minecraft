use std::cell::RefCell;
use std::ops::Not;
use winit::keyboard::KeyCode;
use crate::character::player::Player;
use crate::map_generation::field::Field;

/// Makes an action, corresponding to the key.
/// Returns how much turn was used.
pub fn act(key: &KeyCode, player: &mut Player, field: &mut Field, craft_menu_open: &RefCell<bool>, main_menu_open: &RefCell<bool>) -> f32 {
    match key {
        KeyCode::KeyW => player.walk("w", field),
        KeyCode::KeyA => player.walk("a", field),
        KeyCode::KeyS => player.walk("s", field),
        KeyCode::KeyD => player.walk("d", field),
        KeyCode::ArrowLeft => player.turn(1),
        KeyCode::ArrowRight => player.turn(-1),
        KeyCode::KeyQ => player.mine_infront(field),
        KeyCode::KeyE => player.place_current(field),
        KeyCode::KeyC => player.craft_current(field),
        KeyCode::KeyF => player.consume_current(),
        KeyCode::KeyX => player.shoot_current(field),
        KeyCode::KeyM => { player.toggle_map(); 0. },
        KeyCode::Space => { craft_menu_open.replace(!craft_menu_open.take()); 0. },
        KeyCode::Escape => { main_menu_open.replace(!main_menu_open.take()); 0. },
        _ => { println!("Unknown action"); 0. }
    }
}