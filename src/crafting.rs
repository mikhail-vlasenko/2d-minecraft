use crate::items::{Item, possible_items, Storable};


pub fn item_by_name(name: &str) -> Option<Item> {
    // none means probably a block
    match name {
        "wood" => Some(possible_items::WOOD.clone()),
        "plank" => Some(possible_items::PLANK.clone()),
        "stick" => Some(possible_items::STICK.clone()),
        "wooden pickaxe" => Some(possible_items::WOOD_PICKAXE.clone()),
        _ => None
    }
}
