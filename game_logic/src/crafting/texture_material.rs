use strum_macros::EnumIter;
use serde::{Serialize, Deserialize};
use crate::crafting::storable::{Storable};


/// A non-destructible block that displays some texture.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum TextureMaterial {
    Unknown,
    RobotTL,
    RobotTR,
    RobotBL,
    RobotBR,
}

use TextureMaterial::*;
use crate::crafting::items::Item;

impl TextureMaterial {
    pub fn glyph(&self) -> String {
        match self {
            Unknown => String::from("?"),
            RobotTL => String::from("╔"),
            RobotTR => String::from("╗"),
            RobotBL => String::from("╚"),
            RobotBR => String::from("╝"),
        }
    }

    pub fn required_mining_power(&self) -> i32 {
        match self {
            Unknown => 999,
            _ => 2,
        }
    }

    pub fn drop_item(&self) -> Option<Storable> {
        match self {
            RobotTL => Some(Item::IronIngot.into()),
            RobotTR => Some(Item::IronIngot.into()),
            RobotBL => Some(Item::IronIngot.into()),
            RobotBR => Some(Item::IronIngot.into()),
            _ => None,
        }
    }
}

impl Default for TextureMaterial {
    fn default() -> Self {
        Unknown
    }
}

impl Into<i32> for TextureMaterial {
    fn into(self) -> i32 {
        match self {
            Unknown => 0,
            RobotTL => 1,
            RobotTR => 2,
            RobotBL => 3,
            RobotBR => 4,
        }
    }
}
