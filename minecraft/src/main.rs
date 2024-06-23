extern crate core;

use crate::graphics::event_loop::run;
mod graphics;


pub fn main() {
    pollster::block_on(run());
}
