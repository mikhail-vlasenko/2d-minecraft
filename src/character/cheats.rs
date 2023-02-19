use crate::crafting::consumable::Consumable::{RawMeat, SpeedPotion};
use crate::crafting::items::Item::{Arrow, DiamondSword, IronIngot, IronPickaxe, Stick};
use crate::crafting::material::Material::{CraftTable, Diamond, Plank};
use crate::crafting::ranged_weapon::RangedWeapon::Bow;
use crate::character::player::Player;

impl Player {
    pub fn receive_cheat_package(&mut self) {
        self.pickup(Plank.into(), 100);
        self.pickup(Stick.into(), 100);
        self.pickup(CraftTable.into(), 1);
        self.pickup(IronIngot.into(), 100);
        self.pickup(Diamond.into(), 10);
        self.pickup(Bow.into(), 1);
        self.pickup(Arrow.into(), 100);
        self.pickup(DiamondSword.into(), 1);
        self.pickup(IronPickaxe.into(), 1);
        self.pickup(RawMeat.into(), 10);
        self.pickup(SpeedPotion.into(), 10);
    }
}