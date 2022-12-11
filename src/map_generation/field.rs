use std::cell::{Ref, RefCell, RefMut};
use std::mem::swap;
use std::ops::DerefMut;
use std::panic;
use std::rc::Rc;
use rand::Rng;
use rand::rngs::ThreadRng;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::player::Player;

use crate::map_generation::tile::{randomly_augment, Tile};
use crate::map_generation::block::Block;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::chunk_loader::ChunkLoader;
use crate::map_generation::mobs::a_star::AStar;
use crate::map_generation::mobs::mob::{Mob, Position};
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::map_generation::mobs::spawning::create_mob;


/// The playing grid
pub struct Field {
    /// hashmap for all generated chunks. key: encoded xy position, value: the chunk
    chunk_loader: ChunkLoader,
    /// tiles from these chunks can be accessed
    loaded_chunks: Vec<Vec<Rc<RefCell<Chunk>>>>,
    /// position of the center of the currently loaded chunks. (usually the player's chunk)
    central_chunk: (i32, i32),
    chunk_size: usize,
    /// how far from the player's chunk the chunks are loaded
    loading_distance: usize,
    /// what the player sees
    render_distance: usize,
    /// struct for pathing
    a_star: AStar,
    /// mobs that have been extracted from their chunks, and are currently (in queue for) acting
    stray_mobs: Vec<Mob>,
    /// Number of turns passed. Time of the day is from 0 to 99. Night is from 50 to 99.
    time: f32,
    accumulated_time: f32,
    rng: ThreadRng,
}

impl Field {
    pub fn new(render_distance: usize, starting_chunk: Option<Chunk>) -> Self {
        let loading_distance = 4;
        let chunk_size = 16;

        let chunk_loader = if starting_chunk.is_some() {
            ChunkLoader::with_starting_chunk(loading_distance, starting_chunk.unwrap())
        } else {
            ChunkLoader::new(loading_distance)
        };
        let loaded_chunks = Vec::new();
        let central_chunk = (0, 0);
        let stray_mobs = Vec::new();

        let a_star_radius = 20;
        let a_star = AStar::new(a_star_radius);

        let time = 0.;
        let accumulated_time = 0.;
        let rng = rand::thread_rng();

        let mut field = Self{
            chunk_loader,
            loaded_chunks,
            central_chunk,
            chunk_size,
            loading_distance,
            render_distance,
            a_star,
            stray_mobs,
            time,
            accumulated_time,
            rng
        };
        field.load(central_chunk.0, central_chunk.1);
        field
    }

    /// Load a new set of loaded_chunks around the given chunk
    pub fn load(&mut self, chunk_x: i32, chunk_y:i32) {
        self.chunk_loader.generate_close_chunks(chunk_x, chunk_y);
        self.loaded_chunks = self.chunk_loader.load_around(chunk_x, chunk_y);
        self.central_chunk = (chunk_x, chunk_y);
    }

    /// Steps the field for the given time.
    /// Time increases, mobs may step.
    /// Everything steps only when the turn switches (accumulated_time >= 1).
    ///
    /// # Arguments
    ///
    /// * `passed_time`: for how long the field should be ran
    /// * `player`: the player
    pub fn step_time(&mut self, passed_time: f32, player: &mut Player) {
        self.accumulated_time += passed_time;
        while self.accumulated_time >= 1. {
            let rng: f32 = self.rng.gen();
            if self.is_night() {
                if rng > 0.9 {
                    self.spawn_mobs(player, 5, true)
                }
            } else {
                if rng > 0.9 {
                    self.spawn_mobs(player, 2, false)
                }
            }
            self.step_mobs(player);
            self.accumulated_time -= 1.;
            self.time += 1.;
        }
    }

    pub fn is_night(&self) -> bool {
        (self.time as i32 % 100) >= 50
    }

    pub fn get_time(&self) -> f32 {
        self.time + self.accumulated_time
    }

    pub fn step_mobs(&mut self, player: &mut Player) {
        for i in 0..self.loaded_chunks.len() {
            for j in 0..self.loaded_chunks[i].len() {
                self.stray_mobs.extend(self.loaded_chunks[i][j].borrow_mut().transfer_mobs());
            }
        }
        for _ in 0..self.stray_mobs.len() {
            let optional_mob = self.stray_mobs.pop();
            if optional_mob.is_none() {
                println!("amount of mobs decreased during their turn");
                break;
            }
            let mut m = optional_mob.unwrap();
            m.act_with_speed(self, player, self.min_loaded_idx(), self.max_loaded_idx());
            let (x_chunk, y_chunk) = self.chunk_idx_from_pos(m.pos.x, m.pos.y);

            if m.is_alive() {
                // mobs can self-destruct, dont add them in that case.
                self.loaded_chunks[x_chunk][y_chunk].borrow_mut().add_mob(m);
            }
        }
    }

    /// Spawns mobs on the loaded chunks.
    /// Positions are chosen randomly.
    /// The more mobs exists, the less will be spawned for the given fraction
    ///
    /// # Arguments
    ///
    /// * `player`: the player (for its position)
    /// * `amount`: how many mobs should be spawn (at most)
    pub fn spawn_mobs(&mut self, player: &Player, amount: usize, hostile: bool) {
        let game_time = self.time;
        for _ in 0..amount {
            let tile = self.pick_tile();
            // not too close
            if ((tile.0 - player.x).abs() + (tile.1 - player.y).abs()) >
                (2 * self.render_distance) as i32 {

                let chunk_pos = (self.chunk_pos(tile.0), self.chunk_pos(tile.1));
                let mut chunk = self.get_chunk(tile.0, tile.1);
                // not too crowded, allow more for friendly
                if chunk.get_mobs().len() < 3 {
                    create_mob(chunk.deref_mut(), chunk_pos, tile, game_time, hostile);
                }
            }
        }
    }

    /// Removes top layer of the tiles in the radius.
    ///
    /// # Arguments
    ///
    /// * `center`: x and y of tile that is the epicenter
    /// * `radius`: radius in manhattan distance
    /// * `destruction_power`: mining power applied to remove blocks
    ///
    /// returns: ()
    pub fn explosion(&mut self, center: (i32, i32), radius: i32, destruction_power: i32, player: &mut Player) {
        let start_height = self.len_at(center);
        let damage = destruction_power * 10;
        for i in (center.0 - radius)..=(center.0 + radius) {
            for j in (center.1 - radius)..=(center.1 + radius) {
                if self.top_material_at((i, j)).required_mining_power() <= destruction_power {
                    self.pop_at((i, j));
                    // pop twice if this tile is high
                    if self.len_at((i, j)) > start_height &&
                        self.top_material_at((i, j)).required_mining_power() <= destruction_power {
                        self.pop_at((i, j));
                    }
                }
                self.damage_mob((i, j), damage);
                if player.x == i && player.y == j {
                    player.receive_damage(damage);
                }
            }
        }
        player.add_message(&format!("BOOM!!! (at {}, {})", center.0, center.1));
    }

    pub fn full_pathing(&mut self,
                        source: (i32, i32), destination: (i32, i32),
                        player: (i32, i32), max_detour: Option<i32>) -> ((i32, i32), i32) {
        let detour =
            if max_detour.is_none() {
                10
            } else {
                max_detour.unwrap()
            };
        self.a_star.reset(player.0, player.1);
        let mut secondary_a_star = AStar::default();
        swap(&mut secondary_a_star, &mut self.a_star);
        // now secondary_a_star is the actual self.a_star now
        let res = secondary_a_star.full_pathing(self, source, destination, detour);
        swap(&mut secondary_a_star, &mut self.a_star);
        res
    }

    pub fn get_a_star_radius(&self) -> i32 {
        self.a_star.get_radius()
    }

    pub fn get_towards_player_radius(&self) -> i32 {
        if self.get_time() < 200. { 35 } else { 45 }
    }
}

/// Chunk logic
impl Field {
    /// Gives the Chunk owning the absolute map position, from the loaded chunks only.
    ///
    /// # Arguments
    ///
    /// * `x`: absolute x position on the map
    /// * `y`: absolute y position on the map
    ///
    /// returns: mutable reference to the Chunk at this position, or panics if the Chunk is not loaded
    pub fn get_chunk(&mut self, x: i32, y: i32) -> RefMut<Chunk> {
        let chunk_idx = self.chunk_idx_from_pos(x, y);
        return self.loaded_chunks[chunk_idx.0][chunk_idx.1].as_ref().borrow_mut();
    }

    pub fn get_chunk_immut(&self, x: i32, y: i32) -> Ref<Chunk> {
        let chunk_idx = self.chunk_idx_from_pos(x, y);
        return self.loaded_chunks[chunk_idx.0][chunk_idx.1].as_ref().borrow();
    }

    /// Chunk's index in the loaded_chunks vector.
    ///
    /// /// # Arguments
    ///
    /// * `x`: absolute x position on the map
    /// * `y`: absolute y position on the map
    fn chunk_idx_from_pos(&self, x: i32, y: i32) -> (usize, usize) {
        (self.compute_coord(x, self.central_chunk.0),
         self.compute_coord(y, self.central_chunk.1))
    }

    /// Finds chunk's index along an axis from the absolute map coordinate.
    /// panics for unloaded chunks
    fn compute_coord(&self, coord: i32, center: i32) -> usize {
        let chunk_coord = self.chunk_pos(coord);
        let left_top = center - self.loading_distance as i32;
        let idx = chunk_coord - left_top;
        if idx < 0 || idx as usize > self.loading_distance * 2 {
            panic!("Attempted access to an unloaded chunk (idx = {})", idx);
        }
        idx as usize
    }

    /// Chunk's index for this absolute map coordinate
    /// /// # Arguments
    ///
    /// * `coord`: absolute x or y position on the map
    ///
    /// returns: absolute chunk index (x or y) on this axis corresponding to this coord
    pub fn chunk_pos(&self, coord: i32) -> i32 {
        let mut new_coord = coord;
        if new_coord < 0 {
            new_coord -= self.chunk_size as i32 - 1;
        }
        new_coord / self.chunk_size as i32
    }

    /// Absolute index of the current central chunk
    pub fn get_central_chunk(&self) -> (i32, i32) {
        self.central_chunk
    }
}

/// Rendering-related
impl Field {
    pub fn min_loaded_idx(&self) -> (i32, i32) {
        let x = (self.central_chunk.0 - self.loading_distance as i32) * self.chunk_size as i32;
        let y = (self.central_chunk.1 - self.loading_distance as i32) * self.chunk_size as i32;
        (x, y)
    }

    pub fn max_loaded_idx(&self) -> (i32, i32) {
        let x = (self.central_chunk.0 + self.loading_distance as i32 + 1) * self.chunk_size as i32 - 1;
        let y = (self.central_chunk.1 + self.loading_distance as i32 + 1) * self.chunk_size as i32 - 1;
        (x, y)
    }

    pub fn loaded_tiles_size(&self) -> usize {
        ((2 * self.loading_distance) + 1) * self.chunk_size
    }

    /// Display as glyphs
    pub fn render(&self, player: &Player) {
        for i in 0..self.chunk_size {
            for j in 0..self.chunk_size {
                if i as i32 == player.x && j as i32 == player.y {
                    print!("P");
                } else {
                    print!("{}", self.loaded_chunks[self.loading_distance][self.loading_distance]
                        .borrow().top_at(i as i32, j as i32));
                }
            }
            println!();
        }
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
    /// * `RADIUS`: how far field from player is included
    ///
    /// returns: (2d Vector: the list of positions)
    pub fn texture_indices(&self, player: &Player, material: Material, radius: usize) -> Vec<(i32, i32)> {
        let r = radius as i32;
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x - r)..=(player.x + r) {
            for j in (player.y - r)..=(player.y + r) {
                if self.top_material_at((i, j)) == material {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
    }

    /// Makes a list of positions of blocks of given height around the player.
    pub fn depth_indices(&self, player: &Player, height: usize, radius: usize) -> Vec<(i32, i32)> {
        let r = radius as i32;
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x - r)..=(player.x + r) {
            for j in (player.y - r)..=(player.y + r) {
                if self.len_at((i, j)) == height {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
    }

    /// Makes a list of positions of blocks that have loot on them
    pub fn loot_indices(&self, player: &Player, radius: usize) -> Vec<(i32, i32)> {
        let r = radius as i32;
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x - r)..=(player.x + r) {
            for j in (player.y - r)..=(player.y + r) {
                if self.get_chunk_immut(i, j).get_loot_at(i, j).len() > 0 {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
    }

    pub fn mob_indices(&self, player: &Player, kind: MobKind) -> Vec<(i32, i32)> {
        let mut res: Vec<(i32, i32)> = Vec::new();

        for i in 0..self.loaded_chunks.len() {
            for j in 0..self.loaded_chunks[i].len() {
                for m in self.loaded_chunks[i][j].borrow().get_mobs() {
                    if m.get_kind() == &kind {
                        res.push((m.pos.x - player.x, m.pos.y - player.y))
                    }
                }
            }
        }
        res
    }
}

/// API for Tile interaction, x and y are absolute map positions.
impl Field {
    pub fn len_at(&self, xy: (i32, i32)) -> usize {
        self.get_chunk_immut(xy.0, xy.1).len_at(xy.0, xy.1)
    }
    pub fn push_at(&mut self, block: Block, xy: (i32, i32)) {
        self.get_chunk(xy.0, xy.1).push_at(block, xy.0, xy.1)
    }
    pub fn top_material_at(&self, xy: (i32, i32)) -> Material {
        self.get_chunk_immut(xy.0, xy.1).top_material_at(xy.0, xy.1)
    }
    pub fn pop_at(&mut self, xy: (i32, i32)) -> Option<Block> {
        self.get_chunk(xy.0, xy.1).pop_at(xy.0, xy.1)
    }
    pub fn full_at(&self, xy: (i32, i32)) -> bool {
        self.get_chunk_immut(xy.0, xy.1).full_at(xy.0, xy.1)
    }
    pub fn add_loot_at(&mut self, new: Vec<Storable>, xy: (i32, i32)) {
        self.get_chunk(xy.0, xy.1).add_loot_at(new, xy.0, xy.1)
    }
    pub fn gather_loot_at(&mut self, xy: (i32, i32)) -> Vec<Storable> {
        self.get_chunk(xy.0, xy.1).gather_loot_at(xy.0, xy.1)
    }
    /// This function needs to take stray mobs into account,
    /// as it gets called during the mob movement stage,
    /// when (some) of the mobs are extracted from chunks
    pub fn is_occupied(&self, xy: (i32, i32)) -> bool {
        self.get_chunk_immut(xy.0, xy.1).is_occupied(xy.0, xy.1) || {
            for m in &self.stray_mobs {
                if m.pos.x == xy.0 && m.pos.y == xy.1 {
                    return true;
                }
            }
            false
        }
    }
    pub fn damage_mob(&mut self, xy: (i32, i32), damage: i32) -> bool {
        if self.stray_mobs.len() > 0 {
            for i in 0..self.stray_mobs.len() {
                // found the mob in strays
                if self.stray_mobs[i].pos.x == xy.0 && self.stray_mobs[i].pos.y == xy.1 {
                    return if self.stray_mobs[i].receive_damage(damage) {
                        self.add_loot_at(self.stray_mobs[i].get_kind().loot(),
                                         (self.stray_mobs[i].pos.x, self.stray_mobs[i].pos.y));
                        self.stray_mobs.remove(i);
                        println!("stray mob removed");
                        true
                    } else {
                        false
                    }
                }
            }
        }
        self.get_chunk(xy.0, xy.1).damage_mob(xy.0, xy.1, damage)
    }
    pub fn pick_tile(&mut self) -> (i32, i32) {
        let min = self.min_loaded_idx();
        let max = self.max_loaded_idx();
        let x = self.rng.gen_range(min.0..=max.0);
        let y = self.rng.gen_range(min.1..=max.1);
        (x, y)
    }
}

pub const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
