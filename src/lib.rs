extern crate core;

use crate::graphics::event_loop::run;

pub mod player;
pub mod map_generation;
pub mod crafting;
mod graphics;
pub mod input_decoding;

pub fn lib_main() {
    pollster::block_on(run());
}
