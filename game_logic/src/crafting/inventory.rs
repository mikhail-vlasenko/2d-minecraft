use serde::{Serialize, Deserialize};
use crate::crafting::items::Item::{DiamondSword, IronPickaxe, IronSword, WoodenPickaxe};
use crate::crafting::storable::Storable;
use crate::crafting::storable::Storable::{I};


#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct Inventory {
    items: Vec<(Storable, u32)>
}

impl Inventory {
    pub fn new() -> Self {
        Self {
            items: Vec::new()
        }
    }

    fn get(&self, storable: &Storable) -> Option<u32> {
        for pair in self.items.iter() {
            if *storable == pair.0 {
                return Some(pair.1);
            }
        }
        None
    }

    pub fn get_all(&self) -> &Vec<(Storable, u32)> {
        &self.items
    }

    fn get_idx(&self, storable: &Storable) -> Option<usize> {
        for i in 0..self.items.len() {
            if self.items[i].0 == *storable {
                return Some(i);
            }
        }
        None
    }

    pub fn pickup(&mut self, storable: Storable, amount: u32) {
        let idx = self.get_idx(&storable);
        match idx {
            None => self.items.push((storable, amount)),
            Some(i) => self.items[i].1 += amount
        }
    }

    pub fn contains(&self, storable: &Storable) -> bool {
        match self.get(storable) {
            Some(amount) => amount > 0,
            None => false
        }
    }

    pub fn count(&self, storable: &Storable) -> u32 {
        self.get(storable).unwrap_or_else(|| 0)
    }

    /// Returns whether the drop was successful
    pub fn drop(&mut self, storable: &Storable, amount: u32) -> bool {
        let idx = self.get_idx(storable);
        match idx {
            None => false,
            Some(i) => {
                if self.items[i].1 < amount {
                    false
                } else {
                    self.items[i].1 -= amount;
                    true
                }
            }
        }
    }
}

/// Player's stats, dependent on inventory content.
impl Inventory {
    pub fn damage_modifier(&self) -> i32 {
        if self.contains(&I(DiamondSword)) { 30 }
        else if self.contains(&I(IronSword)) { 15 }
        else if self.contains(&I(IronPickaxe)) { 10 }
        else if self.contains(&I(WoodenPickaxe)) { 5 }
        else { 0 }
    }
    pub fn mining_power(&self) -> i32 {
        if self.contains(&I(IronPickaxe)) { 2 }
        else if self.contains(&I(WoodenPickaxe)) { 1 }
        else { 0 }
    }
}