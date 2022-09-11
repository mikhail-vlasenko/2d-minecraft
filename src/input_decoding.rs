use winit::event::VirtualKeyCode;
use crate::Player;

pub fn act(key: &Option<VirtualKeyCode>, player: &mut Player) {
    match key {
        None => println!("Unrecognized virtual key"),
        Some(VirtualKeyCode::W) => player.walk("w"),
        Some(VirtualKeyCode::A) => player.walk("a"),
        Some(VirtualKeyCode::S) => player.walk("s"),
        Some(VirtualKeyCode::D) => player.walk("d"),
        _ => println!("Unknown action")
    }
}