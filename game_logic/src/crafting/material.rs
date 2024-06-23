use std::fmt;
use std::fmt::{Display, Formatter};
use serde::{Serialize, Deserialize};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Material::*;
use Storable::*;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::*;
use crate::crafting::texture_material::TextureMaterial;


/// What a block on the field can be made of.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum Material {
    Dirt,
    TreeLog,
    Plank,
    Stone,
    Bedrock,
    IronOre,
    CraftTable,
    Diamond,
    Texture(TextureMaterial),  // unbreakable material used to display textures of spawnable structures
}

pub enum MaterialCategory {
    Stone,
    Soil,
    Wood,
    Mineral,
}

impl Material {
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
            Diamond => String::from("D"),
            Texture(t) => t.glyph(),
        }
    }

    pub fn required_mining_power(&self) -> i32 {
        match self {
            Bedrock => 999,
            Texture(t) => t.required_mining_power(),
            Stone => 1,
            IronOre => 1,
            Diamond => 2,
            _ => 0
        }
    }

    pub fn drop_item(&self) -> Option<Storable> {
        match self {
            Texture(t) => t.drop_item(),
            _ => None
        }
    }
}

impl Craftable for Material {
    fn name(&self) -> &str {
        match self {
            Dirt => "dirt",
            Stone => "stone",
            TreeLog => "tree log",
            Bedrock => "bedrock",
            Plank => "plank",
            IronOre => "iron ore",
            CraftTable => "crafting table",
            Diamond => "diamond",
            Texture(_) => "some texture",
        }
    }

    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            Plank => &[(&M(TreeLog), 1)],
            CraftTable => &[(&M(Plank), 4)],
            _ => &[]
        }
    }

    fn craft_yield(&self) -> u32 {
        match self {
            Plank => 4,
            CraftTable => 1,
            _ => 0
        }
    }

    fn menu_section(&self) -> CraftMenuSection {
        match self {
            Plank => Placeables,
            CraftTable => Placeables,
            _ => Uncraftable,
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

impl Into<Storable> for Material {
    fn into(self) -> Storable {
        M(self)
    }
}
