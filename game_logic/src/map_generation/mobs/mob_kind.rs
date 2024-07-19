use strum_macros::EnumIter;
use serde::{Serialize, Deserialize};
use crate::crafting::consumable::Consumable::{RawMeat, SpeedPotion};
use crate::crafting::items::Item::IronIngot;
use crate::crafting::storable::Storable;
use crate::map_generation::mobs::mob_kind::MobKind::*;

#[derive(PartialEq, Copy, Clone, EnumIter, Serialize, Deserialize, Debug)]
pub enum MobKind {
    Zombie,
    Zergling,
    Baneling,
    GelatinousCube,
    Cow,
}

impl MobKind {
    pub fn get_max_hp(&self) -> i32 {
        match self {
            Zombie => 40,
            Zergling => 20,
            Baneling => 20,
            GelatinousCube => 100,
            Cow => 20,
        }
    }
    pub fn get_melee_damage(&self) -> i32 {
        match self {
            Zombie => 10,
            Zergling => 10,
            Baneling => 0,
            GelatinousCube => 15,
            Cow => 5,
        }
    }
    /// 1 - acts every turn
    /// 0.5 - acts every second turn
    /// 2 - acts twice per player turn
    pub fn speed(&self) -> f32 {
        match self {
            Zombie => 0.5,
            Zergling => 1.5,
            Baneling => 0.75,
            GelatinousCube => 0.5,
            Cow => 0.75,
        }
    }

    pub fn hostile(&self) -> bool {
        match self {
            Cow => false,
            _ => true
        }
    }

    pub fn loot(&self) -> Vec<Storable> {
        match self {
            Zombie => vec![],
            Zergling => {
                let mut loot = Vec::new();
                add_loot_with_probability(&mut loot, RawMeat.into(), 0.5);
                add_loot_with_probability(&mut loot, SpeedPotion.into(), 0.25);
                loot
            }
            Baneling => vec![],
            GelatinousCube => {
                let mut loot = Vec::new();
                loot.push(IronIngot.into());
                add_loot_with_probability(&mut loot, IronIngot.into(), 0.5);
                loot
            },
            Cow => vec![Storable::C(RawMeat)],
        }
    }
    
    pub fn jump_distance(&self) -> i32 {
        match self {
            GelatinousCube => 5,
            _ => 0,
        }
    }
}

fn add_loot_with_probability(loot: &mut Vec<Storable>, item: Storable, probability: f32) {
    let rng: f32 = rand::random();
    if rng < probability {
        loot.push(item);
    }
}

#[derive(PartialEq, Copy, Clone, Serialize, Deserialize, Debug)]
pub enum MobState {
    Searching,
    Attacking,
    Channeling(u32),  // the number is amount of turns remaining. GelatinousCube preparing for a jump, for example
}

impl Default for MobState {
    fn default() -> Self {
        MobState::Searching
    }
}

pub const BANELING_EXPLOSION_RAD: i32 = 1;
pub const BANELING_EXPLOSION_PWR: i32 = 2;

pub const ZERGLING_ATTACK_RANGE: i32 = 8;
