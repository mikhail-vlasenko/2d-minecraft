use std::fmt;
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Storable::*;
use crate::crafting::items::Item::*;
use crate::crafting::material::Material;
use crate::crafting::material::Material::Diamond;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::*;
use crate::crafting::interactable::Interactable::*;


/// Something that can't be placed, but can be in the inventory.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum Interactable {
    CrossbowTurret,
}

impl Craftable for Interactable {
    fn name(&self) -> &str {
        match self {
            CrossbowTurret => "crossbow turret",
        }
    }
    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            CrossbowTurret => &[(&I(Stick), 6), (&I(IronIngot), 1), (&M(Diamond), 1)],
        }
    }
    fn craft_yield(&self) -> u32 {
        match self {
            _ => 1,
        }
    }
    fn required_crafter(&self) -> Option<&Material> {
        match self {
            CrossbowTurret => Some(&Material::CraftTable),
        }
    }

    fn menu_section(&self) -> CraftMenuSection {
        match self {
            _ => Interactables
        }
    }
}

impl Display for Interactable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Into<Storable> for Interactable {
    fn into(self) -> Storable {
        IN(self)
    }
}
