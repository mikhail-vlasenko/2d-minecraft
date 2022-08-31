use std::fmt::Display;
use crate::items::{Item};


pub struct Inventory {
    items: Vec<(Item, u32)>
}

impl Inventory {
    pub fn new() -> Self {
        Self {
            items: Vec::new()
        }
    }

    fn get(&self, item: &Item) -> Option<u32> {
        for pair in self.items.iter() {
            if *item == pair.0 {
                return Some(pair.1);
            }
        }
        None
    }

    fn get_idx(&self, item: &Item) -> Option<usize> {
        for i in 0..self.items.len() {
            if self.items[i].0 == *item {
                return Some(i);
            }
        }
        None
    }

    pub fn pickup(&mut self, item: Item, amount: u32) {
        let idx = self.get_idx(&item);
        match idx {
            None => self.items.push((item, amount)),
            Some(i) => self.items[i].1 += amount
        }
    }

    pub fn contains(&self, item: Item) -> bool {
        match self.get(&item) {
            Some(_) => true,
            None => false
        }
    }

    pub fn count(&self, item: &Item) -> u32 {
        match self.get(item) {
            Some(value) => value,
            None => 0
        }
    }

    pub fn drop(&mut self, item: &Item, amount: u32) -> bool {
        let idx = self.get_idx(item);
        match idx {
            None => panic!("you dont have enough to drop"),
            Some(i) => {
                if self.items[i].1 < amount {
                    panic!("you dont have enough to drop");
                } else {
                    self.items[i].1 -= amount;
                }
                true
            }
        }
    }

    pub fn render(&self) {
        println!("You have: ");
        for item in self.items.iter() {
            if item.1 != 0 {
                println!("{}: {}", item.0, item.1);
            }
        }
    }
}