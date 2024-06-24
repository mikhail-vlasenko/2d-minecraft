use std::fmt;
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use serde::{Serialize, Deserialize};
use Storable::*;
use crate::crafting::items::Item::*;
use crate::crafting::material::Material;
use crate::crafting::material::Material::Diamond;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::*;


/// Something that can't be placed, but can be in the inventory.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum Item {
    Stick,
    WoodenPickaxe,
    IronPickaxe,
    IronIngot,
    IronSword,
    DiamondSword,
    Arrow,
    TargetingModule,
}

impl Item {}

impl Craftable for Item {
    fn name(&self) -> &str {
        match self {
            Stick => "stick",
            WoodenPickaxe => "wooden pickaxe",
            IronPickaxe => "iron pickaxe",
            IronIngot => "iron ingot",
            IronSword => "iron sword",
            DiamondSword => "diamond sword",
            Arrow => "arrow",
            TargetingModule => "targeting module",
        }
    }

    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            Stick => &[(&M(Material::Plank), 1)],
            WoodenPickaxe => &[(&M(Material::Plank), 3), (&I(Stick), 2)],
            IronPickaxe => &[(&I(IronIngot), 3), (&I(Stick), 2)],
            IronIngot => &[(&M(Material::IronOre), 1)],
            IronSword => &[(&I(IronIngot), 2), (&I(Stick), 1)],
            DiamondSword => &[(&M(Diamond), 2), (&I(Stick), 1)],
            Arrow => &[(&I(IronIngot), 1), (&I(Stick), 2)],
            TargetingModule => &[],
        }
    }

    fn craft_yield(&self) -> u32 {
        match self {
            Stick => 2,
            Arrow => 2,
            TargetingModule => 0,
            _ => 1,
        }
    }

    fn required_crafter(&self) -> Option<&Material> {
        match self {
            WoodenPickaxe => Some(&Material::CraftTable),
            IronPickaxe => Some(&Material::CraftTable),
            IronSword => Some(&Material::CraftTable),
            DiamondSword => Some(&Material::CraftTable),
            _ => None
        }
    }

    fn menu_section(&self) -> CraftMenuSection {
        match self {
            Stick => Ingredients,
            WoodenPickaxe => Tools,
            IronPickaxe => Tools,
            IronIngot => Ingredients,
            IronSword => Weapons,
            DiamondSword => Weapons,
            Arrow => Weapons,
            TargetingModule => Uncraftable,
        }
    }
}

impl TryFrom<String> for Item {
    type Error = &'static str;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        for i in Item::iter() {
            if i.name() == value {
                return Ok(i)
            }
        }
        return Err("unknown item")
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Into<Storable> for Item {
    fn into(self) -> Storable {
        I(self)
    }
}
