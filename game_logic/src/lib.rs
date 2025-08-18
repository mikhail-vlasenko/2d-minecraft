extern crate core;

extern crate lazy_static;

use std::sync::RwLock;
use lazy_static::lazy_static;
use crate::settings::Settings;

pub mod map_generation;
pub mod crafting;
pub mod character;
pub mod auxiliary;
pub mod perform_action;
pub mod settings;
pub mod game_state;

lazy_static! {
    pub static ref SETTINGS: RwLock<Settings> = RwLock::new(Settings::load().into_owned());
}
