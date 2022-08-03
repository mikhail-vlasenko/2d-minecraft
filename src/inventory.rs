use std::collections::{HashMap, HashSet};
use std::any::{Any, TypeId};
use std::collections::hash_map::{DefaultHasher, Entry};
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use crate::hash_map_storable::{into_key, Key};


pub struct Inventory {
    items: HashMap<Box<dyn Key>, u32>
}

// pub trait Storable: Hash + Eq + 'static {
//     fn store(&self);
// }
//
// impl PartialEq for Box<dyn Storable> {
//     fn eq(&self, other: &Self) -> bool {
//         Storable::eq(self.as_ref(), other.as_ref())
//     }
// }
//
// impl Eq for Box<dyn Storable> {}
//
// impl Hash for Box<dyn Storable> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         let key_hash = Storable::hash(self.as_ref());
//         state.write_u64(key_hash);
//     }
// }
//
// fn into_key(key: impl Eq + Hash + 'static) -> Box<dyn Storable> {
//     Box::new(key)
// }

impl Inventory {
    pub fn new() -> Self {
        Self {
            items: HashMap::new()
        }
    }
    pub fn pickup(&mut self, item: impl Key + Eq + Hash + 'static + Display) {
        self.items.entry(into_key(item))
            .and_modify(|e| { *e += 1 })
            .or_insert(1);
    }

    pub fn contains(&self, item: impl Key + Eq + Hash + 'static + Display) -> bool {
        self.items.contains_key(&into_key(item))
    }

    pub fn drop(&mut self, item: impl Key + Eq + Hash + 'static + Display) -> bool {
        let entry = self.items.entry(into_key(item));
        match entry {
            Entry::Vacant(_) => false,
            Entry::Occupied(mut o) => {
                let value = o.get_mut();
                if *value == 1 {
                    o.remove();
                } else {
                    *value -= 1;
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