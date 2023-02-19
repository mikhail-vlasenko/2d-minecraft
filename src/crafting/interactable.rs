use std::fmt;
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Storable::*;
use crate::crafting::items::Item::*;
use crate::crafting::material::Material;
use crate::crafting::material::Material::Diamond;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::*;
use crate::crafting::interactable::InteractableKind::*;
use crate::crafting::inventory::Inventory;
use crate::map_generation::field::Field;


#[derive(Clone)]
pub struct Interactable {
    kind: InteractableKind,
    inventory: Inventory,
    position: (i32, i32),
}

impl Interactable {
    pub fn new(kind: InteractableKind, position: (i32, i32)) -> Self {
        Interactable {
            kind,
            inventory: Inventory::new(),
            position,
        }
    }
    pub fn step(&mut self, field: &mut Field, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        match self.kind {
            CrossbowTurret => { println!("Turret is shooting!"); }
        }
    }
    pub fn load_item(&mut self, item: Storable, amount: u32) {
        self.inventory.pickup(item, amount);
    }
    pub fn get_kind(&self) -> InteractableKind {
        self.kind
    }
    pub fn get_position(&self) -> (i32, i32) {
        self.position
    }
}

/// Something that can be placed and interacted with.
#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum InteractableKind {
    CrossbowTurret,
}

impl InteractableKind {
    pub fn is_turret(&self) -> bool {
        match self {
            CrossbowTurret => true,
        }
    }
}

impl Craftable for InteractableKind {
    fn name(&self) -> &str {
        match self {
            CrossbowTurret => "crossbow turret",
        }
    }
    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            CrossbowTurret => &[(&I(Stick), 6), (&I(IronIngot), 1), (&M(Diamond), 1)],
        }
    }
    fn craft_yield(&self) -> u32 {
        match self {
            _ => 1,
        }
    }
    fn required_crafter(&self) -> Option<&Material> {
        match self {
            CrossbowTurret => Some(&Material::CraftTable),
        }
    }

    fn menu_section(&self) -> CraftMenuSection {
        match self {
            _ => Interactables
        }
    }
}

impl Display for InteractableKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Into<Storable> for InteractableKind {
    fn into(self) -> Storable {
        IN(self)
    }
}
