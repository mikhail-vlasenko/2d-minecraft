use std::cmp::min;
use strum::IntoEnumIterator;
use game_logic::game_state::GameState;
use game_logic::map_generation::field::{absolute_to_relative, Field, RelativePos};
use game_logic::map_generation::field_observation::get_tile_observation;
use game_logic::map_generation::mobs::mob_kind::MobKind;
use crate::observation::{LOOT_INFO_SIZE, make_action_mask, MAX_MOBS, MOB_INFO_SIZE, Observation};


pub trait ObservationProvider {
    fn get_observation(&self) -> Observation;
    fn get_closest_mobs(&self) -> Vec<[i32; MOB_INFO_SIZE as usize]>;
    fn make_loot_array(&self) -> [[i32; LOOT_INFO_SIZE as usize]; MAX_MOBS as usize];
}


impl ObservationProvider for GameState {
    fn get_observation(&self) -> Observation {
        let (top_materials, tile_heights) = get_tile_observation(&self.field, &self.player);
        let top_materials = top_materials.iter().map(|row| row.iter().map(|mat| (*mat).into()).collect()).collect();
        Observation::new(top_materials, tile_heights, self.get_time(), &self.player, self.is_done(), self.get_closest_mobs(), self.make_loot_array(), make_action_mask(&self))
    }

    /// Produces a vector of 4-arrays of mob information: x, y, type, health
    /// The vector is sorted by manhattan distance from the player
    /// x and y are player-relative
    /// mob kind is an index in the MobKind enum
    /// health is integer - rounded percentage of max health, so from 0 to 100
    fn get_closest_mobs(&self) -> Vec<[i32; MOB_INFO_SIZE as usize]> {
        let mob_kinds = MobKind::iter().collect::<Vec<MobKind>>();
        let mut mobs = self.field.close_mob_info(|mob| {
            let pos = absolute_to_relative((mob.pos.x, mob.pos.y), self.player.xy());
            let mut arr = [0; MOB_INFO_SIZE as usize];
            let idx = mob_kinds.iter().position(| kind | { kind == mob.get_kind() }).unwrap();
            arr[0] = pos.0;
            arr[1] = pos.1;
            arr[2] = (mob.get_hp_share() * 100.0) as i32;
            arr[3 + idx] = 1;
            arr
        }, self.player.xy());

        // Sorting the mobs by manhattan distance from the player
        mobs.sort_by(|a, b| {
            let dist_a = a[0].abs() + a[1].abs();
            let dist_b = b[0].abs() + b[1].abs();
            dist_a.cmp(&dist_b)
        });
        
        mobs
    }

    /// Produces a 2D array of loot information: x, y, content
    /// The array is sorted by manhattan distance from the player
    /// x and y are player-relative
    /// content (1: arrow, 2: other loot, 3: arrow and other loot). content -1 for no loot
    fn make_loot_array(&self) -> [[i32; LOOT_INFO_SIZE as usize]; MAX_MOBS as usize] {
        let mut loot = [[0, 0, -1]; MAX_MOBS as usize];
        let player_dist_cmp = |a: &RelativePos, b: &RelativePos| {
            let dist_a = a.0.abs() + a.1.abs();
            let dist_b = b.0.abs() + b.1.abs();
            dist_a.cmp(&dist_b)
        };
        let mut loot_indices = self.field.loot_indices(self.player.xy());
        loot_indices.sort_by(player_dist_cmp);
        let mut arrow_indices = self.field.arrow_indices(self.player.xy());
        arrow_indices.sort_by(player_dist_cmp);
        let mut min_empty_loot_position = 0;
        // record loot and loot+arrow positions
        for i in 0..min(loot_indices.len(), MAX_MOBS as usize) {
            let idx = loot_indices[i];
            if arrow_indices.contains(&idx) {
                loot[i] = [idx.0, idx.1, 3];
                let index_accounted_for = arrow_indices.iter().position(|x| *x == idx).unwrap();
                arrow_indices.remove(index_accounted_for);
            } else {
                loot[i] = [idx.0, idx.1, 2];
            }
            min_empty_loot_position = i + 1;
        }
        // in the remaining slots, record arrow-only positions
        for i in 0..min(arrow_indices.len(), MAX_MOBS as usize - min_empty_loot_position) {
            let idx = arrow_indices[i];
            loot[min_empty_loot_position + i] = [idx.0, idx.1, 1];
        }
        loot
    }
}
