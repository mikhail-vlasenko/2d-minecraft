use std::collections::HashMap;
use crate::hash_map_storable::{into_key, Key};
use crate::items::{possible_items, Storable};


fn item_by_name(name: &str) -> Box<dyn Storable> {
    match name {
        "wood" => Box::new(possible_items::WOOD.clone()),
        "plank" => Box::new(possible_items::PLANK.clone()),
        _ => panic!()
    }
}
