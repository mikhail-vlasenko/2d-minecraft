use std::fmt;
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Storable::*;
use crate::crafting::storable::{Craftable, Storable};
use crate::crafting::consumable::Consumable::*;
use crate::crafting::material::Material;


#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum Consumable {
    Apple,
    RawMeat,
}

impl Consumable { }

impl Craftable for Consumable {
    fn name(&self) -> &str {
        match self {
            Apple => "apple",
            RawMeat => "raw meat",
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
