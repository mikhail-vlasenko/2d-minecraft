use std::fmt;
use std::fmt::Display;

pub trait Storable {
    fn name(&self) -> &str;
    fn is_placeable(&self) -> bool;
    fn is_craftable(&self) -> bool;
    fn craft_requirements(&self) -> &[(&Item, u32)];
}

#[derive(Clone, Hash)]
pub struct Item<'a> {
    name: &'a str,
    is_placeable: bool,
    is_craftable: bool,
    craft_requirements: &'a [(&'a Item<'a>, u32)],
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

#[derive(Clone, PartialEq, Hash)]
pub struct Tool<'a> {
    name: &'a str,
    tool_type: &'a str,
    craft_requirements: &'a [(&'a Item<'a>, u32)],
}

impl Display for Tool<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Storable for Tool<'_> {
    fn name(&self) -> &str { self.name }
    fn is_placeable(&self) -> bool { false }
    fn is_craftable(&self) -> bool { true }
    fn craft_requirements(&self) -> &[(&Item, u32)] { &self.craft_requirements }
}

pub mod possible_items {
    use crate::items::{Item, Tool};

    pub static WOOD: Item = Item {
        name: "wood",
        is_placeable: true,
        is_craftable: false,
        craft_requirements: &[],
    };
    pub static PLANK: Item = Item {
        name: "planks",
        is_placeable: true,
        is_craftable: true,
        craft_requirements: &[(&WOOD, 1)],
    };
    pub static STICK: Item = Item {
        name: "stick",
        is_placeable: false,
        is_craftable: true,
        craft_requirements: &[(&PLANK, 1)],
    };
    pub static WOOD_PICKAXE: Tool = Tool {
        name: "wooden pickaxe",
        tool_type: "pickaxe",
        craft_requirements: &[(&PLANK, 3), (&STICK, 2)]
    };
}