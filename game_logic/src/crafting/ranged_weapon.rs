use std::fmt;
use std::fmt::Display;
use strum_macros::EnumIter;
use serde::{Serialize, Deserialize};
use Storable::*;
use crate::crafting::items::Item;
use crate::crafting::items::Item::*;
use crate::crafting::ranged_weapon::RangedWeapon::*;
use crate::crafting::material::Material;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::Weapons;


#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum RangedWeapon {
    Bow
}

impl RangedWeapon {
    pub fn ammo(&self) -> &Item {
        match self {
            Bow => &Arrow,
        }
    }
    pub fn range(&self) -> i32 {
        match self {
            Bow => 5,
        }
    }
    pub fn damage(&self) -> i32 {
        match self {
            Bow => 30,
        }
    }
}

impl Craftable for RangedWeapon {
    fn name(&self) -> &str {
        match self {
            Bow => "bow",
        }
    }
    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            Bow =>  &[(&I(Stick), 6), (&I(IronIngot), 1)],
        }
    }
    fn craft_yield(&self) -> u32 {
        match self {
            _ => 1,
        }
    }
    fn required_crafter(&self) -> Option<&Material> {
        match self {
            Bow => Some(&Material::CraftTable),
        }
    }

    fn menu_section(&self) -> CraftMenuSection {
        match self {
            _ => Weapons
        }
    }
}

impl Display for RangedWeapon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Into<Storable> for RangedWeapon {
    fn into(self) -> Storable {
        RW(self)
    }
}
