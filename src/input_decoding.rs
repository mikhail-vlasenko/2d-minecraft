use winit::event::VirtualKeyCode;
use crate::{Field, Player};

pub fn act(key: &Option<VirtualKeyCode>, player: &mut Player, field: &mut Field) {
    match key {
        None => println!("Unrecognized virtual key"),
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
        _ => println!("Unknown action")
    }
}