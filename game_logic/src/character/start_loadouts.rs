use rand::{random, Rng, thread_rng};
use strum::IntoEnumIterator;
use itertools::Itertools;
use crate::character::player::Player;
use crate::crafting::consumable::Consumable;
use crate::crafting::items::Item;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::Storable::{C, I, RW};
use crate::map_generation::field::{Field, relative_to_absolute, RelativePos};
use crate::map_generation::mobs::mob::{Mob, Position};
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;


pub fn apply_loadout(player: &mut Player, field: &mut Field) {
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
            spawn_close_zombies(field, player, 3);
        },
        "archer" => {
            player.pickup(RW(RangedWeapon::Bow), 1);
            player.pickup(I(Item::Arrow), 10);
            spawn_close_zombies(field, player, 3);
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

/// Give a relative position for a mob spawn.
fn choose_close_mob_position() -> RelativePos {
    let mut x = thread_rng().gen_range(2..4);
    let mut y = thread_rng().gen_range(2..4);
    if random() {
        x = -x;
    }
    if random() {
        y = -y;
    }
    (x, y)
}

/// Spawn some zombies close to the player.
/// Some of them are damaged.
fn spawn_close_zombies(field: &mut Field, player: &Player, amount: usize) {
    let mut positions = Vec::new();
    for _ in 0..amount {
        positions.push(choose_close_mob_position());
    }
    positions = positions.into_iter().unique().collect();
    for pos in positions {
        let abs_pos = relative_to_absolute(pos, player);
        let mut mob = Mob::new(Position::new(abs_pos, field), MobKind::Zombie);
        if random() {
            mob.receive_damage(MobKind::Zombie.get_max_hp() / 2);
        }
        field.place_mob(mob);
    }
}
