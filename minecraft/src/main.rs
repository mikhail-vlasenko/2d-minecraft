extern crate core;

use crate::graphics::event_loop::run;
mod graphics;
mod graphical_config;
mod input_decoding;


pub fn main() {
    pollster::block_on(run());
}
