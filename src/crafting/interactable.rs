use std::{fmt, mem};
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Storable::*;
use crate::character::acting_with_speed::ActingWithSpeed;
use crate::character::player::Player;
use crate::crafting::items::Item::*;
use crate::crafting::material::Material;
use crate::crafting::material::Material::Diamond;
use crate::crafting::storable::{Craftable, CraftMenuSection, Storable};
use crate::crafting::storable::CraftMenuSection::*;
use crate::crafting::interactable::InteractableKind::*;
use crate::crafting::inventory::Inventory;
use crate::crafting::turret::TargetingData;
use crate::map_generation::field::Field;


#[derive(Clone)]
pub struct Interactable {
    kind: InteractableKind,
    inventory: Inventory,
    position: (i32, i32),
    /// targeting data can be changed by the player, so it's not part of the kind
    pub(super) targeting_data: Option<TargetingData>,
    logs: Vec<String>,
    /// when this reaches 1, the interactable is eligible to step
    speed_buffer: f32,
}

impl Interactable {
    pub fn new(kind: InteractableKind, position: (i32, i32)) -> Self {
        Interactable {
            kind,
            inventory: Inventory::new(),
            position,
            targeting_data: kind.get_targeting_data(),
            logs: Vec::new(),
            speed_buffer: 0.0,
        }
    }
    pub fn load_item(&mut self, item: Storable, amount: u32) {
        self.inventory.pickup(item, amount);
    }
    pub fn unload_item(&mut self, item: &Storable, amount: u32) -> bool {
        self.inventory.drop(item, amount)
    }
    pub fn take_all(&mut self) -> Vec<(Storable, u32)> {
        let items = self.inventory.get_all().clone();
        self.inventory = Inventory::new();
        items
    }
    pub fn get_inventory(&self) -> &Inventory {
        &self.inventory
    }
    pub fn get_kind(&self) -> InteractableKind {
        self.kind
    }
    pub fn get_position(&self) -> (i32, i32) {
        self.position
    }
    pub fn get_targeting_data(&self) -> &TargetingData {
        self.targeting_data.as_ref().unwrap()
    }
}

impl ActingWithSpeed for Interactable {
    fn act(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        match self.kind {
            CrossbowTurret => {
                self.act_turret(field, player, min_loaded, max_loaded);
            }
        }
    }
    fn get_speed(&self) -> f32 {
        self.kind.speed()
    }
    fn get_speed_buffer(&self) -> f32 {
        self.speed_buffer
    }
    fn add_to_speed_buffer(&mut self, amount: f32) {
        self.speed_buffer += amount;
    }
    fn decrement_speed_buffer(&mut self) {
        self.speed_buffer -= 1.0;
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
    // how often the step() function should be called
    pub fn speed(&self) -> f32 {
        match self {
            CrossbowTurret => 0.5,
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
            CrossbowTurret => &[(&I(Stick), 6), (&M(Diamond), 1), (&I(TargetingModule), 1)]
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
