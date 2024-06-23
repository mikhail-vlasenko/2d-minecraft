extern crate core;

use crate::graphics::event_loop::run;
mod graphics;
mod graphical_config;


pub fn main() {
    pollster::block_on(run());
}
