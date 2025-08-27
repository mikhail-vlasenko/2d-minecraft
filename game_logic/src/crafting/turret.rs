use std::cmp::{max, min};
use serde::{Serialize, Deserialize};
use strum::IntoEnumIterator;
use crate::auxiliary::animations::ProjectileType;
use crate::character::acting_with_speed::ActingWithSpeed;
use crate::character::game_score::score_killed_mob;
use crate::character::p2p_interactions::P2PInteraction;
use crate::crafting::interactable::{Interactable, InteractableKind};
use crate::crafting::interactable::InteractableKind::*;
use crate::crafting::items::Item;
use crate::crafting::items::Item::Arrow;
use crate::map_generation::field::{AbsolutePos, Field};
use crate::map_generation::mobs::mob::Position;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
/// Data that is used by turrets
pub struct TargetingData {
    /// which kinds of mobs this turret targets
    target_mobs: Vec<MobKind>,
    /// how far this turret shoots
    range: usize,
    damage: i32,
    ammo: Item,
}

impl Interactable {
    pub fn act_turret(&mut self, field: &mut Field, player_pos: &Position,
                      min_loaded: (i32, i32), max_loaded: (i32, i32)) -> Vec<(P2PInteraction, AbsolutePos)> {
        let mut interactions = Vec::new();
        let targeting = self.get_targeting_data();

        if !self.get_inventory().contains(&targeting.ammo.into()) {
            // no ammo, do nothing
            return interactions;
        }

        let middle_idx = (field.chunk_pos(self.get_position().0),
                                   field.chunk_pos(self.get_position().1));
        let size = SETTINGS.read().unwrap().field.chunk_size;
        let chunk_distance =
            (targeting.range as f32 /
                SETTINGS.read().unwrap().field.chunk_size as f32).ceil() as i32;

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
                // found a valid target to shoot
                if let Some(mob_pos) = mob_pos {
                    let mob_kind = field.get_mob_kind_at(mob_pos).unwrap();
                    let died = field.damage_mob(mob_pos, targeting.damage);
                    if died {
                        // todo: add score to the player who placed the turret, not the player standing next to it
                        interactions.push((P2PInteraction::IncreaseScore(score_killed_mob(&mob_kind)), player_pos.xy()));
                    }
                    // put the arrow on the field
                    if targeting.ammo == Arrow {
                        let rng: f32 = rand::random();
                        if rng > SETTINGS.read().unwrap().player.arrow_break_chance {
                            field.add_loot_at(vec![Arrow.into()], mob_pos);
                        }
                    }
                    // remove the arrow from the turret
                    self.unload_item(&targeting.ammo.into(), 1);
                    field.animations_buffer.add_projectile_animation(ProjectileType::Arrow, self.get_position(), mob_pos);
                    interactions.push(
                        (P2PInteraction::AddMessage(
                            format!("Turret shot mob at ({}, {})", mob_pos.0, mob_pos.1)
                        ),
                         player_pos.xy())
                    );
                    return interactions;
                }
            }
        }
        // nothing was shot, so it should be able to make a turn on the next tick
        self.add_to_speed_buffer(1. - self.get_speed());
        interactions
    }

    pub fn get_targets(&self) -> Vec<MobKind> {
        self.targeting_data.as_ref().unwrap().target_mobs.clone()
    }

    pub fn set_targets(&mut self, targets: Vec<MobKind>) {
        self.targeting_data.as_mut().unwrap().set_targets(targets)
    }
}

impl InteractableKind {
    pub fn get_targeting_data(&self) -> Option<TargetingData> {
        let hostiles = MobKind::iter().filter(|m| m.hostile()).collect();
        match self {
            CrossbowTurret => Some(TargetingData {
                target_mobs: hostiles,
                range: 5,
                damage: 30,
                ammo: Arrow,
            }),
        }
    }
    pub fn get_ammo(&self) -> Option<Item> {
        if self.is_turret() {
            Some(self.get_targeting_data().unwrap().ammo)
        } else {
            None
        }
    }
}

impl TargetingData {
    pub fn set_targets(&mut self, targets: Vec<MobKind>) {
        self.target_mobs = targets;
    }
}
