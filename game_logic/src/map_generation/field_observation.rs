use crate::character::player::Player;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::items::Item::Arrow;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::map_generation::field::{AbsolutePos, Field, RelativePos};
use crate::map_generation::mobs::mob::Mob;
use crate::map_generation::mobs::mob_kind::MobKind;


/// Rendering-related
impl Field {
    pub fn min_loaded_idx(&self) -> AbsolutePos {
        let x = (self.get_central_chunk().0 - self.get_loading_distance() as i32) * self.get_chunk_size() as i32;
        let y = (self.get_central_chunk().1 - self.get_loading_distance() as i32) * self.get_chunk_size() as i32;
        (x, y)
    }

    pub fn max_loaded_idx(&self) -> AbsolutePos {
        let x = (self.get_central_chunk().0 + self.get_loading_distance() as i32 + 1) * self.get_chunk_size() as i32 - 1;
        let y = (self.get_central_chunk().1 + self.get_loading_distance() as i32 + 1) * self.get_chunk_size() as i32 - 1;
        (x, y)
    }

    pub fn loaded_tiles_size(&self) -> usize {
        ((2 * self.get_loading_distance()) + 1) * self.get_chunk_size()
    }

    /// Makes a list of positions of blocks of given material around the player.
    /// Useful for rendering if blocks of same type are rendered simultaneously.
    /// Positions are centered on the player.
    /// [positive, positive] is bottom right.
    /// first coord is vertical.
    ///
    /// # Arguments
    ///
    /// * `player`: the player
    /// * `material`: index only blocks of this material
    /// * `radius`: how far field from player is included
    ///
    /// returns: (2d Vector: the list of positions)
    pub fn texture_indices(&self, player: &Player, material: Material, radius: i32) -> Vec<RelativePos> {
        let cond = |(i, j)| { self.top_material_at((i, j)) == material };
        self.indices_around_player(player, cond, radius)
    }

    fn indices_around_player<F: Fn(AbsolutePos) -> bool>(&self, player: &Player, condition: F, radius: i32) -> Vec<RelativePos> {
        let mut res = Vec::new();
        for i in (player.x - radius)..=(player.x + radius) {
            for j in (player.y - radius)..=(player.y + radius) {
                if condition((i, j)) {
                    res.push((i - player.x, j - player.y));
                }
            }
        }
        res
    }

    /// Makes a list of player-centered positions of blocks of given height around the player.
    pub fn depth_indices(&self, player: &Player, height: usize) -> Vec<RelativePos> {
        let cond = |(i, j)| { self.len_at((i, j)) == height };
        self.indices_around_player(player, cond, self.get_render_distance() as i32)
    }

    /// Makes a list of positions of blocks that have loot on them.
    /// Does not count arrows as loot.
    pub fn loot_indices(&self, player: &Player) -> Vec<RelativePos> {
        let cond = |(i, j)| {
            let chunk = self.get_chunk_immut(i, j);
            let loot = chunk.get_loot_at(i, j);
            for l in loot {
                if l != &Storable::I(Arrow) {
                    return true;
                }
            }
            false
        };
        self.indices_around_player(player, cond, self.get_render_distance() as i32)
    }

    /// Makes a list of positions of blocks that have loot on them
    pub fn arrow_indices(&self, player: &Player) -> Vec<RelativePos> {
        let cond = |(i, j)| {
            self.get_chunk_immut(i, j).get_loot_at(i, j).contains(&Storable::I(Arrow))
        };
        self.indices_around_player(player, cond, self.get_render_distance() as i32)
    }

    pub fn interactable_indices(&self, player: &Player, interactable: InteractableKind) -> Vec<RelativePos> {
        // todo: can rewrite like mob_indices for speed
        let cond = |(i, j)| {
            self.get_interactable_kind_at((i, j)) == Some(interactable)
        };
        self.indices_around_player(player, cond, self.get_render_distance() as i32)
    }

    /// Makes a list of positions with mobs of this kind on them, and their corresponding rotations.
    /// Positions are centered on the player.
    /// Checks stray mobs, so can be used during mob turns.
    pub fn mob_indices(&self, player: &Player, kind: MobKind) -> Vec<(RelativePos, u32)> {
        let mut res= self.conditional_close_mob_info(|m| {
            ((m.pos.x - player.x, m.pos.y - player.y), m.get_rotation())
        }, player, |m: &Mob| { m.get_kind() == &kind });

        for m in self.get_stray_mobs() {
            if m.get_kind() == &kind && self.is_position_visible(player, (m.pos.x, m.pos.y)) {
                res.push(((m.pos.x - player.x, m.pos.y - player.y), m.get_rotation()));
            }
        }
        res
    }

    /// Makes a list of infos for mobs that are close enough to be visible.
    /// Can't be used during mob turns, as it doesn't account for stray mobs
    pub fn conditional_close_mob_info<F: Fn(&Mob) -> T, T, C: Fn(&Mob) -> bool>(&self, info_extractor: F, player: &Player, condition: C) -> Vec<T> {
        let mut res= Vec::new();

        // selects a square of chunks around the player that are close enough to have some tiles in view
        let (min_idx, max_idx) = self.get_close_chunk_indices();

        for i in min_idx..=max_idx {
            for j in min_idx..=max_idx {
                let middle_idx = self.get_loading_distance() as i32;

                // a position on the target chunk
                let absolute_pos: AbsolutePos = (
                    (i as i32 - middle_idx) * self.get_chunk_size() as i32 + player.x, 
                    (j as i32 - middle_idx) * self.get_chunk_size() as i32 + player.y);
                for m in self.get_chunk_immut(absolute_pos.0, absolute_pos.1).get_mobs() {
                    if self.is_position_visible(player, (m.pos.x, m.pos.y)) && condition(m) {
                        res.push(info_extractor(m));
                    }
                }
            }
        }
        res
    }

    pub fn close_mob_info<F: Fn(&Mob) -> T, T>(&self, info_extractor: F, player: &Player) -> Vec<T> {
        self.conditional_close_mob_info(info_extractor, player, |_| { true })
    }

    /// Index boundaries of chunks that are close to the player.
    fn get_close_chunk_indices(&self) -> (usize, usize) {
        // self.loaded_chunks is centered on the player, so loaded_chunks[get_loading_distance()][get_loading_distance()] is the player's chunk
        // the loaded_chunks also are square, so the returned indices are the same for both dimensions
        let middle_idx = self.get_loading_distance();
        // how many chunks are in the render distance
        let chunk_distance = (self.get_render_distance() as f32 / self.get_chunk_size() as f32).ceil() as usize;
        let min_idx = middle_idx - chunk_distance;
        let max_idx = middle_idx + chunk_distance;
        (min_idx, max_idx)
    }

    fn is_position_visible(&self, player: &Player, pos: AbsolutePos) -> bool {
        (pos.0 - player.x).abs() <= self.get_render_distance() as i32 &&
            (pos.1 - player.y).abs() <= self.get_render_distance() as i32
    }
}