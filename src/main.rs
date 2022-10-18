#[macro_use] extern crate text_io;
extern crate core;

use map_generation::field::Field;
use crate::material::Material;
use crate::player::Player;
use crate::storable::Storable;
use crate::cli_event_loop::cli_event_loop;
use crate::graphics::event_loop::run;

mod material;
mod player;
mod inventory;
mod items;
mod storable;
mod graphics;
mod cli_event_loop;
mod input_decoding;
mod map_generation;


fn main() {
    // cli_event_loop();

    pollster::block_on(run());

}
