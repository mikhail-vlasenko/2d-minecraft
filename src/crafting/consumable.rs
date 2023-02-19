use std::fmt;
use std::fmt::Display;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use Storable::*;
use crate::crafting::storable::{Craftable, Storable};
use crate::crafting::consumable::Consumable::*;
use crate::crafting::material::Material;
use crate::map_generation::field::Field;
use crate::mechanics::delayed_action::DelayedAction;
use crate::player::Player;


#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum Consumable {
    Apple,
    RawMeat,
    SpeedPotion,
}

impl Consumable {
    pub fn apply_effect(&self, player: &mut Player, field: &mut Field) {
        match self {
            Apple => player.heal(20),
            RawMeat => player.heal(20),
            SpeedPotion => {
                let reset_speed = DelayedAction::new(
                    50,
                    Player::set_speed_multiplier,
                    Box::new(1.));
                player.add_delayed_action(reset_speed);
            },
        }
    }
}

impl Craftable for Consumable {
    fn name(&self) -> &str {
        match self {
            Apple => "apple",
            RawMeat => "raw meat",
            SpeedPotion => "speed potion",
        }
    }

    fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            _ => &[]
        }
    }

    fn craft_yield(&self) -> u32 {
        match self {
            _ => 0
        }
    }
}

impl Display for Consumable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Into<Storable> for Consumable {
    fn into(self) -> Storable {
        C(self)
    }
}
