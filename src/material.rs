use std::fmt;
use std::fmt::{Display, Formatter};

#[derive(Clone, PartialEq, Hash)]
pub struct Material<'a> {
    pub name: &'a str,
    pub category: &'a str,
    pub display_symbol: &'a str,
}

impl Display for Material<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub mod materials {
    use crate::Material;

    pub static AIR: Material = Material {
        name: "air",
        category: "air",
        display_symbol: " ",
    };

    pub static DIRT: Material = Material {
        name: "dirt",
        category: "soil",
        display_symbol: "d",
    };

    pub static TREE_LOG: Material = Material {
        name: "tree log",
        category: "wood",
        display_symbol: "T",
    };

    pub static WOOD_PLANKS: Material = Material {
        name: "wood planks",
        category: "wood",
        display_symbol: "W",
    };

    pub static STONE: Material = Material {
        name: "stone",
        category: "stone",
        display_symbol: "s",
    };

    pub static BEDROCK: Material = Material {
        name: "bedrock",
        category: "stone",
        display_symbol: "b",
    };
}
