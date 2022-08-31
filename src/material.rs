use std::fmt;
use std::fmt::{Display, Formatter};
use crate::items::{Item, Storable};


#[derive(PartialEq, Copy, Clone, Hash)]
pub enum Material {
    Dirt,
    TreeLog,
    Plank,
    Stone,
    Bedrock
}

pub enum MaterialCategory {
    Stone,
    Soil,
    Wood
}

impl Material {
    pub fn from_string(name: &str) -> Option<Material> {
        use Material::*;
        match name {
            "dirt" => Some(Dirt),
            "stone" => Some(Stone),
            "tree log" => Some(TreeLog),
            "bedrock" => Some(Bedrock),
            "plank" => Some(Plank),
            _ => None
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
        }
    }
}

impl Display for Material {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use Material::*;
        write!(f, "{}", match self {
            Dirt => "dirt",
            Stone => "stone",
            TreeLog => "tree log",
            Bedrock => "bedrock",
            Plank => "plank",
        })
    }
}
