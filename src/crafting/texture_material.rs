use std::fmt;
use std::fmt::{Display, Formatter};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Storable::*;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::*;


/// A non-destructible block that displays some texture.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum TextureMaterial {
    Unknown,
    RobotTL,
    RobotTR,
    RobotBL,
    RobotBR,
}

impl TextureMaterial {
    pub fn glyph(&self) -> String {
        use TextureMaterial::*;
        match self {
            Unknown => String::from("?"),
            RobotTL => String::from("╔"),
            RobotTR => String::from("╗"),
            RobotBL => String::from("╚"),
            RobotBR => String::from("╝"),
        }
    }
}

impl Default for TextureMaterial {
    fn default() -> Self {
        TextureMaterial::Unknown
    }
}
