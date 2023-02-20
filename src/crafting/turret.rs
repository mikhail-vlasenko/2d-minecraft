use std::cmp::{max, min};
use crate::character::player::Player;
use crate::crafting::interactable::{Interactable, InteractableKind};
use crate::crafting::interactable::InteractableKind::*;
use crate::map_generation::field::Field;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;

#[derive(Clone)]
/// Data that is used by turrets
pub struct TargetingData {
    /// which kinds of mobs this turret targets
    target_mobs: Vec<MobKind>,
    /// how far this turret shoots
    range: usize,
    damage: i32,
}

impl Interactable {
    pub fn act_turret(&mut self, field: &mut Field, player: &mut Player,
                      min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let targeting = self.get_targeting_data();
        let middle_idx = (field.chunk_pos(self.get_position().0),
                                   field.chunk_pos(self.get_position().1));
        let size = SETTINGS.field.chunk_size;
        let chunk_distance =
            (targeting.range as f32 /
                SETTINGS.field.chunk_size as f32).ceil() as i32;

        // iterate through close chunks not going out of loaded chunks
        for i in max(min_loaded.0 / size, middle_idx.0 - chunk_distance)
            ..=min(max_loaded.0 / size, middle_idx.0 + chunk_distance) {
            for j in max(min_loaded.1 / size, middle_idx.1 - chunk_distance)
                ..=min(max_loaded.1 / size, middle_idx.1 + chunk_distance) {
                let mut mob_pos = None;
                // get the chunk at this index (get_chunk_immut uses tile coords)
                for mob in field.get_chunk_immut(i * size, j * size).get_mobs() {
                    // check if the mob is in range and is a target
                    if ((mob.pos.x - self.get_position().0).abs() + (mob.pos.y - self.get_position().1).abs()) <=
                        targeting.range as i32 &&
                        targeting.target_mobs.contains(&mob.get_kind()) {
                        mob_pos = Some((mob.pos.x, mob.pos.y));
                        break;
                    }
                }
                if let Some(mob_pos) = mob_pos {
                    field.damage_mob(mob_pos, targeting.damage);
                    player.add_message(&format!("Turret shot mob at ({}, {})", mob_pos.0, mob_pos.1));
                    return;
                }
            }
        }
    }
}

impl InteractableKind {
    pub fn get_targeting_data(&self) -> Option<TargetingData> {
        match self {
            CrossbowTurret => Some(TargetingData {
                target_mobs: vec![MobKind::Zombie, MobKind::Zergling, MobKind::Baneling],
                range: 5,
                damage: 30,
            }),
        }
    }
}
