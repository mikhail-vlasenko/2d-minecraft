use strum_macros::EnumIter;
use crate::crafting::consumable::Consumable::RawMeat;
use crate::crafting::storable::Storable;
use crate::map_generation::mobs::mob_kind::MobKind::*;

#[derive(PartialEq, Copy, Clone, EnumIter, Debug)]
pub enum MobKind {
    Zombie,
    Zergling,
    Baneling,
    Cow,
}

impl MobKind {
    pub fn get_max_hp(&self) -> i32 {
        match self {
            Zombie => 40,
            Zergling => 20,
            Baneling => 20,
            Cow => 20,
        }
    }
    pub fn get_melee_damage(&self) -> i32 {
        match self {
            Zombie => 10,
            Zergling => 10,
            Baneling => 0,
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
        let rng: f32 = rand::random();
        match self {
            Zombie => vec![],
            Zergling => if rng > 0.7 { vec![Storable::C(RawMeat)] } else { vec![] },
            Baneling => vec![],
            Cow => vec![Storable::C(RawMeat)],
        }
    }
}

pub const BANELING_EXPLOSION_RAD: i32 = 1;
pub const BANELING_EXPLOSION_PWR: i32 = 2;
