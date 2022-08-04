use crate::items::{Item, possible_items, Storable};


pub fn item_by_name(name: &str) -> Option<Item> {
    match name {
        "wood" => Some(possible_items::WOOD.clone()),
        "plank" => Some(possible_items::PLANK.clone()),
        _ => None
    }
}
