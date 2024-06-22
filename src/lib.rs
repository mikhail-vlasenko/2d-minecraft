extern crate core;

extern crate lazy_static;

use std::sync::RwLock;
use lazy_static::lazy_static;
use crate::graphics::event_loop::run;
use crate::settings::Settings;

pub mod map_generation;
pub mod crafting;
mod graphics;
pub mod input_decoding;
pub mod character;
pub mod auxiliary;
mod settings;

lazy_static! {
    static ref SETTINGS: RwLock<Settings> = RwLock::new(Settings::load().into_owned());
}

pub fn lib_main() {
    pollster::block_on(run());
}
