extern crate core;

#[macro_use]
extern crate lazy_static;
use lazy_static::lazy_static;
use config::Config;
use std::sync::RwLock;
use crate::graphics::event_loop::run;

pub mod player;
pub mod map_generation;
pub mod crafting;
mod graphics;
pub mod input_decoding;

lazy_static! {
    static ref SETTINGS: Config = Config::builder()
    .add_source(config::File::with_name("settings.yaml"))
    .build().unwrap();
}

pub fn lib_main() {
    pollster::block_on(run());
}
