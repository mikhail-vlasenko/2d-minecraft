use strum::IntoEnumIterator;
use crate::character::player::Player;
use crate::crafting::consumable::Consumable;
use crate::crafting::items::Item;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::Storable::{C, I, RW};
use crate::SETTINGS;


pub fn apply_loadout(player: &mut Player) {
    let binding = SETTINGS.read().unwrap().player.start_inventory.loadout.clone();
    let loadout = binding.as_ref();
    match loadout {
        "empty" => {},
        "apples" => {
            player.pickup(C(Consumable::Apple), 3);
        },
        "fighter" => {
            player.pickup(I(Item::IronSword), 1);
            player.pickup(C(Consumable::RawMeat), 3);
        },
        "archer" => {
            player.pickup(RW(RangedWeapon::Bow), 1);
            player.pickup(I(Item::Arrow), 10);
        },
        _ => {
            panic!("Unknown loadout: {}", loadout);
        }
    }
    if SETTINGS.read().unwrap().player.start_inventory.cheating_start {
        player.receive_cheat_package();
    }
    // set active consumable
    let mut consumables = vec![];
    for consumable in Consumable::iter() {
        if player.has(&C(consumable)) {
            consumables.push(consumable);
        }
    }
    if consumables.is_empty() {
        player.consumable = Consumable::RawMeat;
        return;
    } else {
        player.consumable = consumables[0];
    }
}