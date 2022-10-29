#[macro_use] extern crate text_io;
extern crate core;

use map_generation::field::Field;
use crate::player::Player;
use crate::cli_event_loop::cli_event_loop;
use crate::graphics::event_loop::run;

mod player;
mod graphics;
mod cli_event_loop;
mod input_decoding;
mod map_generation;
mod crafting;


fn main() {
    // cli_event_loop();

    pollster::block_on(run());

}
