use std::fmt;
use std::fmt::{Display, Formatter};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Material::*;
use Storable::*;
use crate::crafting::storable::Storable;


/// What a block on the field can be made of.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum Material {
    Dirt,
    TreeLog,
    Plank,
    Stone,
    Bedrock,
    IronOre,
    CraftTable,
}

pub enum MaterialCategory {
    Stone,
    Soil,
    Wood,
    Mineral,
}

impl Material {
    pub fn name(&self) -> &str {
        match self {
            Dirt => "dirt",
            Stone => "stone",
            TreeLog => "tree log",
            Bedrock => "bedrock",
            Plank => "plank",
            IronOre => "iron ore",
            CraftTable => "crafting table",
        }
    }

    pub fn glyph(&self) -> String {
        use Material::*;
        match self {
            Dirt => String::from("d"),
            Stone => String::from("s"),
            TreeLog => String::from("T"),
            Bedrock => String::from("b"),
            Plank => String::from("w"),
            IronOre => String::from("i"),
            CraftTable => String::from("C"),
        }
    }

    pub fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            Plank => &[(&M(TreeLog), 1)],
            CraftTable => &[(&M(Plank), 4)],
            _ => &[]
        }
    }

    pub fn craft_yield(&self) -> u32 {
        match self {
            Plank => 4,
            CraftTable => 1,
            _ => 0
        }
    }

    pub fn required_crafter(&self) -> Option<&Material> {
        match self {
            _ => None
        }
    }
}

impl TryFrom<String> for Material {
    type Error = &'static str;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        for m in Material::iter() {
            if m.name() == value {
                return Ok(m)
            }
        }
        return Err("unknown material")
    }
}

impl Display for Material {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
