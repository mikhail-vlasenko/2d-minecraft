use std::collections::{HashMap, HashSet};
use std::any::{Any, TypeId};
use std::collections::hash_map::{DefaultHasher, Entry};
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use crate::block::Block;
use crate::crafting::item_by_name;
use crate::items::{Item, Storable};


pub struct Inventory<'a> {
    items: Vec<(Item<'a>, u32)>
}

impl Inventory<'_> {
    pub fn new() -> Self {
        Self {
            items: Vec::new()
        }
    }

    fn get(&self, item: Item) -> Option<u32> {
        for mut pair in self.items.iter() {
            if item == pair.0 {
                return Some(pair.1);
            }
        }
        None
    }

    fn get_idx(&self, item: Item) -> Option<usize> {
        for i in 0..self.items.len() {
            if self.items[i].0 == item {
                return Some(i);
            }
        }
        None
    }

    // this just doesnt work as i hoped it would
    // fn get_mut(&mut self, item: Item) -> Option<&mut u32> {
    //     for i in 0..self.items.len() {
    //         if self.items[i].0 == item {
    //             return Some(&mut self.items[i].1);
    //         }
    //     }
    //     None
    // }

    pub fn pickup(&mut self, item: Item) {
        let idx = self.get_idx(item);
        match idx {
            None => self.items.push((Item::new(), 1)),
            Some(i) => self.items[i].1 += 1
        }
    }

    pub fn contains(&self, item: Item) -> bool {
        match self.get(item) {
            Some(_) => true,
            None => false
        }
    }

    pub fn count(&self, item: Item) -> u32 {
        match self.get(item) {
            Some(value) => value,
            None => 0
        }
    }

    pub fn drop(&mut self, item: Item, count: u32) -> bool {
        let idx = self.get_idx(item);
        match idx {
            None => panic!("you dont have enough to drop"),
            Some(i) => {
                if self.items[i].1 > count {
                    panic!("you dont have enough to drop");
                } else {
                    self.items[i].1 -= count;
                }
                true
            }
        }
    }

    pub fn render(&self) {
        println!("You have: ");
        for item in self.items.iter() {
            println!("{}: {}", item.0, item.1);
        }
    }
}