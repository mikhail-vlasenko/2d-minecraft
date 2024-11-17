use crate::character::player::Player;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;
use crate::settings::_Config__scoring;

impl Player {
    pub fn get_score(&self) -> i32 {
        self.score
    }

    pub fn add_to_score(&mut self, score: i32) {
        self.score += score;
    }
    
    pub fn score_values(&self) -> _Config__scoring {
        SETTINGS.read().unwrap().scoring.clone()
    }
    
    pub fn score_mined(&mut self, material: &Material) {
        match material {
            Material::Diamond => self.add_to_score(self.score_values().blocks.mined.diamond),
            _ => {}
        }
    }
    
    pub fn score_crafted(&mut self, storable: &Storable, amount: u32) {
        let score_per_item = match storable {
            Storable::I(Item::IronIngot) => self.score_values().blocks.crafted.iron_ingot,
            Storable::I(Item::DiamondSword) => self.score_values().blocks.crafted.diamond_sword,
            Storable::IN(InteractableKind::CrossbowTurret) => self.score_values().blocks.crafted.crossbow_turret,
            Storable::I(Item::Arrow) => self.score_values().blocks.crafted.arrow,
            _ => 0,
        };
        self.add_to_score(score_per_item * amount as i32);
    }
    
    pub fn score_killed_mob(&mut self, mob_kind: &MobKind) {
        match mob_kind {
            MobKind::Zombie => self.add_to_score(self.score_values().killed_mobs.zombie),
            MobKind::Zergling => self.add_to_score(self.score_values().killed_mobs.zergling),
            MobKind::Baneling => self.add_to_score(self.score_values().killed_mobs.baneling),
            MobKind::GelatinousCube => self.add_to_score(self.score_values().killed_mobs.gelatinous_cube),
            MobKind::Cow => self.add_to_score(self.score_values().killed_mobs.cow),
        }
    }
    
    pub fn score_passed_time(&mut self, time_passed: f32, time_total: f32) {
        // a little awkward function because score is integer and this handles when time is any float
        let turned_over_whole = time_total.floor() as i32 - (time_total - time_passed).floor() as i32;
        if turned_over_whole > 0 {
            self.add_to_score(self.score_values().time.turn * turned_over_whole);
        }
        let turned_over_day = time_total.floor() as i32 / 100 - (time_total - time_passed).floor() as i32 / 100;
        if turned_over_day > 0{
            self.add_to_score(self.score_values().time.day * turned_over_day);
        }
    }
}