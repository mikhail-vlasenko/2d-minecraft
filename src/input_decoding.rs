use winit::event::VirtualKeyCode;
use crate::{Field, Player};

pub fn act(key: &Option<VirtualKeyCode>, player: &mut Player, field: &mut Field) {
    match key {
        None => println!("Unrecognized virtual key"),
        Some(VirtualKeyCode::W) => player.walk("w", field),
        Some(VirtualKeyCode::A) => player.walk("a", field),
        Some(VirtualKeyCode::S) => player.walk("s", field),
        Some(VirtualKeyCode::D) => player.walk("d", field),
        Some(VirtualKeyCode::M) => player.mine(field, -1, 0),
        Some(VirtualKeyCode::Q) => player.mine(field, -1, 0),
        Some(VirtualKeyCode::P) => player.place_current(field, -1, 0),
        Some(VirtualKeyCode::E) => player.place_current(field, -1, 0),
        Some(VirtualKeyCode::C) => player.craft_current(),
        _ => println!("Unknown action")
    }
}