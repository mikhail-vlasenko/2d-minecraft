use std::fmt;
use std::fmt::Display;
use crate::block::Block;
use crate::crafting::item_by_name;
use crate::Material;

pub trait Storable: Eq + Display {
    fn name(&self) -> &str;
    fn is_placeable(&self) -> bool;
    fn is_craftable(&self) -> bool;
    fn craft_requirements(&self) -> &[(&Item, u32)];
}

#[derive(Clone)]
pub struct Item<'a> {
    name: &'a str,
    is_placeable: bool,
    is_craftable: bool,
    tool_type: &'a str,
    craft_requirements: &'a [(&'a Item<'a>, u32)],
}

impl Item<'_> {
    pub fn new() -> Self {
        Self {
            name: "",
            is_placeable: false,
            is_craftable: false,
            tool_type: "",
            craft_requirements: &[]
        }
    }
    pub fn from_material(material: &'static Material) -> Self {
        Self {
            name: material.name,
            is_placeable: true,
            is_craftable: false,
            tool_type: "",
            craft_requirements: &[]
        }
    }
}

impl PartialEq<Self> for Item<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Item<'_> {}

impl Display for Item<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// unsuccessful better solution
// impl From<Block<'_>> for Item<'_> {
    // fn from(block: Block) -> Self {
    //     let item = item_by_name(block.material.name);
    //     match item {
    //         Some(i) => panic!(),
    //         None => Self {
    //             name: block.material.name,
    //             is_placeable: true,
    //             is_craftable: false,
    //             tool_type: "",
    //             craft_requirements: &[]
    //         }
    //     }
    // }
// }

impl Storable for Item<'_> {
    fn name(&self) -> &str {
        self.name
    }
    fn is_placeable(&self) -> bool {
        self.is_placeable
    }
    fn is_craftable(&self) -> bool {
        self.is_craftable
    }
    fn craft_requirements(&self) -> &[(&Item, u32)] {
        &self.craft_requirements
    }
}

pub mod possible_items {
    use crate::items::Item;

    pub static WOOD: Item = Item {
        name: "wood",
        is_placeable: true,
        is_craftable: false,
        tool_type: "",
        craft_requirements: &[],
    };
    pub static PLANK: Item = Item {
        name: "planks",
        is_placeable: true,
        is_craftable: true,
        tool_type: "",
        craft_requirements: &[(&WOOD, 1)],
    };
    pub static STICK: Item = Item {
        name: "stick",
        is_placeable: false,
        is_craftable: true,
        tool_type: "",
        craft_requirements: &[(&PLANK, 1)],
    };
    pub static WOOD_PICKAXE: Item = Item {
        name: "wooden pickaxe",
        is_placeable: false,
        is_craftable: true,
        tool_type: "pickaxe",
        craft_requirements: &[(&PLANK, 3), (&STICK, 2)]
    };
}