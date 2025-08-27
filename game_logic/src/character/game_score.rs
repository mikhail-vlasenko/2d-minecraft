use crate::crafting::interactable::InteractableKind;
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;
use crate::settings::_Config__scoring;

pub fn score_values() -> _Config__scoring {
    SETTINGS.read().unwrap().scoring.clone()
}

pub fn score_mined(material: &Material) -> i32 {
    match material {
        Material::DiamondOre => score_values().blocks.mined.diamond,
        _ => 0
    }
}

pub fn score_crafted(storable: &Storable, amount: u32) -> i32 {
    let score_per_item = match storable {
        Storable::I(Item::IronIngot) => score_values().blocks.crafted.iron_ingot,
        Storable::I(Item::DiamondSword) => score_values().blocks.crafted.diamond_sword,
        Storable::IN(InteractableKind::CrossbowTurret) => score_values().blocks.crafted.crossbow_turret,
        Storable::I(Item::Arrow) => score_values().blocks.crafted.arrow,
        _ => 0,
    };
    score_per_item * amount as i32
}

pub fn score_killed_mob(mob_kind: &MobKind) -> i32 {
    match mob_kind {
        MobKind::Zombie => score_values().killed_mobs.zombie,
        MobKind::Zergling => score_values().killed_mobs.zergling,
        MobKind::Baneling => score_values().killed_mobs.baneling,
        MobKind::GelatinousCube => score_values().killed_mobs.gelatinous_cube,
        MobKind::Cow => score_values().killed_mobs.cow,
    }
}

pub fn score_passed_time(time_passed: f32, time_total: f32) -> i32 {
    // a little awkward function because score is integer and this handles when time is any float
    let mut score = 0;
    let turned_over_whole = time_total.floor() as i32 - (time_total - time_passed).floor() as i32;
    if turned_over_whole > 0 {
        score += score_values().time.turn * turned_over_whole;
    }
    let turned_over_day = time_total.floor() as i32 / 100 - (time_total - time_passed).floor() as i32 / 100;
    if turned_over_day > 0{
        score += score_values().time.day * turned_over_day;
    }
    score
}
