use std::fmt;
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use serde::{Serialize, Deserialize};
use Storable::*;
use crate::character::status_effects::StatusEffect;
use crate::crafting::storable::{Craftable, Storable};
use crate::crafting::consumable::Consumable::*;
use crate::crafting::material::Material;
use crate::character::player::Player;


#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum Consumable {
    Apple,
    RawMeat,
    SpeedPotion,
}

impl Consumable {
    pub fn apply_effect(&self, player: &mut Player) {
        match self {
            Apple => player.heal(20),
            RawMeat => player.heal(20),
            SpeedPotion => player.add_status_effect(StatusEffect::Speedy, 25),
        }
    }
}

impl Craftable for Consumable {
    fn name(&self) -> &str {
        match self {
            Apple => "apple",
            RawMeat => "raw meat",
            SpeedPotion => "speed potion",
        }
    }

    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            _ => &[]
        }
    }

    fn craft_yield(&self) -> u32 {
        match self {
            _ => 0
        }
    }
}

impl Display for Consumable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Into<Storable> for Consumable {
    fn into(self) -> Storable {
        C(self)
    }
}
