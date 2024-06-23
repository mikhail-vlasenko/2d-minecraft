extern crate core;

extern crate lazy_static;

use std::sync::RwLock;
use lazy_static::lazy_static;
use crate::settings::Settings;

pub mod map_generation;
pub mod crafting;
pub mod input_decoding;
pub mod character;
pub mod auxiliary;
pub mod gym_interface;
pub mod settings;

lazy_static! {
    pub static ref SETTINGS: RwLock<Settings> = RwLock::new(Settings::load().into_owned());
}
