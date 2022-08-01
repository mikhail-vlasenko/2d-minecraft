#[derive(Clone, PartialEq)]
pub struct Material<'a> {
    pub name: &'a str,
    pub category: &'a str,
    pub display_symbol: &'a str,
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
