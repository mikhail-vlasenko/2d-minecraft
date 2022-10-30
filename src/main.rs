extern crate core;

use map_generation::field::Field;
use crate::player::Player;
use crate::graphics::event_loop::run;

mod player;
mod graphics;
mod input_decoding;
mod map_generation;
mod crafting;


fn main() {
    pollster::block_on(run());
}
