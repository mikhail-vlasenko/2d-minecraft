extern crate core;

extern crate lazy_static;
use lazy_static::lazy_static;
use crate::graphics::event_loop::run;
use crate::settings::Settings;

pub mod player;
pub mod map_generation;
pub mod crafting;
mod graphics;
pub mod input_decoding;
pub mod character;
mod settings;

lazy_static! {
    static ref SETTINGS: Settings = Settings::load().into_owned();
}

pub fn lib_main() {
    pollster::block_on(run());
}
