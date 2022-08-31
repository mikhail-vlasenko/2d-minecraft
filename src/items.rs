use std::fmt;
use std::fmt::Display;
use crate::items::Item::*;
use crate::Material;


#[derive(PartialEq, Copy, Clone, Hash)]
pub enum Item {
    TreeLog,
    Plank,
    Stick,
    WoodenPickaxe
}

impl Item {
    pub fn name(&self) -> &str {
        match self {
            TreeLog => "tree log",
            Plank => "plank",
            Stick => "stick",
            WoodenPickaxe => "wooden pickaxe"
        }
    }

    pub fn is_placeable(&self) -> bool {
        match self {
            TreeLog => true,
            Plank => true,
            _ => false
        }
    }

    pub fn craft_requirements(&self) -> &[(&Item, u32)] {
        match self {
            Plank => &[(&TreeLog, 1)],
            Stick => &[(&Plank, 1)],
            WoodenPickaxe => &[(&Plank, 3), (&Stick, 2)],
            _ => &[]
        }
    }

    pub fn craft_yield(&self) -> u32 {
        match self {
            Plank => 4,
            Stick => 2,
            WoodenPickaxe => 1,
            _ => 0
        }
    }

    pub fn is_craftable(&self) -> bool {
        self.craft_yield() > 0
    }
}

impl TryFrom<Material> for Item {
    type Error = &'static str;

    fn try_from(value: Material) -> Result<Self, Self::Error> {
        match value {
            Material::TreeLog => Ok(TreeLog),
            Material::Plank => Ok(Plank),
            _ => Err("no corresponding item for the material")
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub fn item_by_name(name: &str) -> Option<Item> {
    // none means probably a block
    match name {
        "tree log" => Some(TreeLog),
        "plank" => Some(Plank),
        "stick" => Some(Stick),
        "wooden pickaxe" => Some(WoodenPickaxe),
        _ => None
    }
}
