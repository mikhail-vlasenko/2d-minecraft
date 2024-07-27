use std::sync::{Arc, Mutex};
use std::cmp::max;
use std::mem::swap;
use std::ops::DerefMut;
use std::panic;
use rand::Rng;
use serde::{Serialize, Deserialize};
use derivative::Derivative;
use crate::auxiliary::animations::{AnimationsBuffer, TileAnimationType};
use crate::character::acting_with_speed::ActingWithSpeed;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::character::player::Player;
use crate::crafting::interactable::{Interactable, InteractableKind};

use crate::map_generation::block::Block;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::chunk_loader::ChunkLoader;
use crate::map_generation::mobs::a_star::AStar;
use crate::map_generation::mobs::mob::{Mob};
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::map_generation::mobs::spawning::create_mob;
use crate::SETTINGS;

/// The playing grid
#[derive(Serialize, Deserialize, Debug, Derivative)]
#[derivative(PartialEq)]
pub struct Field {
    /// hashmap for all generated chunks. key: encoded xy position, value: the chunk
    chunk_loader: ChunkLoader,
    /// tiles from these chunks can be accessed
    /// shape is ((loading_distance * 2 + 1), (loading_distance * 2 + 1))
    #[serde(skip)]
    #[derivative(PartialEq = "ignore")]
    loaded_chunks: Vec<Vec<Arc<Mutex<Chunk>>>>,
    /// position of the center of the currently loaded chunks. (usually the player's chunk)
    central_chunk: AbsoluteChunkPos,
    /// how far from the player's chunk the chunks are loaded, in chunks
    loading_distance: usize,
    /// what the player sees, in tiles
    render_distance: usize,
    /// radius of the map view, in tiles
    map_render_distance: usize,
    /// struct for pathing
    #[derivative(PartialEq = "ignore")]
    a_star: AStar,
    /// mobs that have been extracted from their chunks, and are currently (in queue for) acting
    stray_mobs: Vec<Mob>,
    /// animations that have to be started as a result of events
    pub animations_buffer: AnimationsBuffer,
    /// Number of turns passed. Time of the day is from 0 to 99. Night is from 50 to 99.
    time: f32,
    accumulated_time: f32,
}

impl Field {
    pub fn new(render_distance: usize, starting_chunk: Option<Chunk>) -> Self {
        // in chunks
        let loading_distance = SETTINGS.read().unwrap().field.loading_distance as usize;

        let chunk_loader = if starting_chunk.is_some() {
            ChunkLoader::with_starting_chunk(loading_distance, starting_chunk.unwrap())
        } else {
            ChunkLoader::new(loading_distance)
        };
        
        let map_render_distance = max(SETTINGS.read().unwrap().field.map_radius as usize, loading_distance * chunk_loader.get_chunk_size());

        let loaded_chunks = Vec::new();
        let central_chunk = (0, 0);
        let stray_mobs = Vec::new();
        let animations_buffer = AnimationsBuffer::new();

        let a_star = AStar::new(SETTINGS.read().unwrap().pathing.a_star_radius);

        let time = 0.;
        let accumulated_time = 0.;

        let mut field = Self{
            chunk_loader,
            loaded_chunks,
            central_chunk,
            loading_distance,
            render_distance,
            map_render_distance,
            a_star,
            stray_mobs,
            animations_buffer,
            time,
            accumulated_time,
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
            player.step_status_effects();
            self.step_interactables(player);
            let rng: f32 = rand::random();
            if rng > 0.9 {
                self.spawn_mobs(player,
                                self.get_mob_spawn_amount(),
                                self.is_night())
            }
            self.step_mobs(player);
            self.accumulated_time -= 1.;
            self.time += 1.;
            player.score_passed_time(1., self.get_time());
        }
    }
    
    fn detach_mobs(&mut self) {
        for i in 0..self.loaded_chunks.len() {
            for j in 0..self.loaded_chunks[i].len() {
                self.stray_mobs.extend(self.loaded_chunks[i][j].lock().unwrap().transfer_mobs());
            }
        }
    }

    pub fn step_mobs(&mut self, player: &mut Player) {
        self.detach_mobs();
        for _ in 0..self.stray_mobs.len() {
            let optional_mob = self.stray_mobs.pop();
            if optional_mob.is_none() {
                break;
            }
            let mut mob = optional_mob.unwrap();
            mob.act_with_speed(self, player, self.min_loaded_idx(), self.max_loaded_idx());

            if mob.is_alive() {
                // mobs can self-destruct, dont add them in that case.
                self.place_mob(mob);
            }
        }
    }

    pub fn step_turrets(&mut self, player: &mut Player) {
        let mut turrets = Vec::new();
        for i in 0..self.loaded_chunks.len() {
            for j in 0..self.loaded_chunks[i].len() {
                turrets.extend(self.loaded_chunks[i][j].lock().unwrap().transfer_turrets());
            }
        }
        for _ in 0..turrets.len() {
            let mut turret = turrets.pop().unwrap();
            turret.act_with_speed(self, player, self.min_loaded_idx(), self.max_loaded_idx());
            let (x_chunk, y_chunk) =
                self.chunk_idx_from_pos(turret.get_position().0, turret.get_position().1);
            self.loaded_chunks[x_chunk][y_chunk].lock().unwrap().add_interactable(turret);
        }
    }

    pub fn step_interactables(&mut self, player: &mut Player) {
        self.step_turrets(player);
    }

    /// Spawns mobs on the loaded chunks.
    /// Positions are chosen randomly.
    /// The more mobs exists, the less will be spawned for the given fraction
    ///
    /// # Arguments
    ///
    /// * `player`: the player (for its position)
    /// * `amount`: how many mobs should be spawn (at most)
    pub fn spawn_mobs(&mut self, player: &Player, amount: i32, hostile: bool) {
        let game_time = self.time;
        for _ in 0..amount {
            let tile = self.pick_tile();
            // not too close
            if ((tile.0 - player.x).abs() + (tile.1 - player.y).abs()) >
                (2 * self.render_distance) as i32 {

                let is_red_moon = self.is_red_moon();
                let chunk_pos = (self.chunk_pos(tile.0), self.chunk_pos(tile.1));
                let mut chunk = self.get_chunk(tile.0, tile.1);
                // limit so it is not too crowded, but allow more mobs during red moon
                if chunk.get_mobs().len() <
                    (SETTINGS.read().unwrap().mobs.spawning.max_mobs_on_chunk as usize + game_time as usize / 700)
                    || is_red_moon {
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
    pub fn explosion(&mut self, center: AbsolutePos, radius: i32, destruction_power: i32, player: &mut Player) {
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
                if self.get_interactable_kind_at((i, j)).is_some() {
                    self.break_interactable_at((i, j));
                }
                if self.is_occupied((i, j)) {
                    self.damage_mob((i, j), damage);
                }
                if player.x == i && player.y == j {
                    player.receive_damage(damage);
                }
            }
        }
        player.add_message(&format!("BOOM!!! (at {}, {})", center.0, center.1));
    }

    /// Perform a full A* pathing from source to destination.
    pub fn full_pathing(&mut self,
                        source: AbsolutePos, destination: AbsolutePos,
                        player: AbsolutePos, max_detour: Option<i32>) -> (AbsolutePos, i32) {
        let detour =
            if max_detour.is_none() {
                SETTINGS.read().unwrap().pathing.default_detour
            } else {
                max_detour.unwrap()
            };
        self.a_star.reset(player.0, player.1);
        let mut secondary_a_star = AStar::default();
        swap(&mut secondary_a_star, &mut self.a_star);
        // write_to_state_spy("Swapped A* and starting search");
        // now secondary_a_star is the actual self.a_star now
        let res = secondary_a_star.full_pathing(self, source, destination, detour);
        // write_to_state_spy("Finished A* full pathing");
        swap(&mut secondary_a_star, &mut self.a_star);
        res
    }
}

/// Getters
impl Field {
    /// Absolute index of the current central chunk
    pub fn get_central_chunk(&self) -> AbsoluteChunkPos {
        self.central_chunk
    }
    
    pub fn get_chunk_size(&self) -> usize {
        self.chunk_loader.get_chunk_size()
    }
    
    pub fn is_night(&self) -> bool {
        (self.time as i32 % 100) >= 50 || self.is_red_moon()
    }

    pub fn is_red_moon(&self) -> bool {
        (self.time as i32 % 700) >= 550
    }

    pub fn get_time(&self) -> f32 {
        self.time + self.accumulated_time
    }
    
    pub fn get_mob_spawn_amount(&self) -> i32 {
        let settings = SETTINGS.read().unwrap();
        if !self.is_night() {
            settings.mobs.spawning.base_day_amount
        } else {
            let mut amount = settings.mobs.spawning.base_night_amount;
            if self.is_red_moon() {
                amount += settings.mobs.spawning.base_night_amount / 2;
            }
            amount += self.time as i32 / 100 * settings.mobs.spawning.increase_amount_every;
            amount
        }
    }
    
    /// From how many tiles away do mobs start to use full a pathing algorithm
    pub fn get_a_star_radius(&self) -> i32 {
        self.a_star.get_radius()
    }

    /// From how many tiles away do mobs start to move towards the player ('sense' him).
    pub fn get_towards_player_radius(&self) -> i32 {
        if self.get_time() < 200. {
            SETTINGS.read().unwrap().pathing.towards_player_radius.early
        } else if self.is_red_moon() {
            SETTINGS.read().unwrap().pathing.towards_player_radius.red_moon
        } else {
            SETTINGS.read().unwrap().pathing.towards_player_radius.usual
        }
    }
    
    pub fn get_stray_mobs(&self) -> &Vec<Mob> {
        &self.stray_mobs
    }
    
    /// How far from the player's chunk the chunks are loaded
    pub fn get_loading_distance(&self) -> usize {
        self.loading_distance
    }

    /// How many tiles from player to a side appear on the screen
    pub fn get_render_distance(&self) -> usize {
        self.render_distance
    }

    pub fn get_map_render_distance(&self) -> usize {
        self.map_render_distance
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
    pub fn get_chunk(&mut self, x: i32, y: i32) -> std::sync::MutexGuard<Chunk> {
        let chunk_idx = self.chunk_idx_from_pos(x, y);
        return self.loaded_chunks[chunk_idx.0][chunk_idx.1].lock().unwrap();
    }

    pub fn get_chunk_immut(&self, x: i32, y: i32) -> std::sync::MutexGuard<Chunk> {
        let chunk_idx = self.chunk_idx_from_pos(x, y);
        return self.loaded_chunks[chunk_idx.0][chunk_idx.1].lock().unwrap();
    }

    /// Chunk's index in the loaded_chunks vector.
    ///
    /// /// # Arguments
    ///
    /// * `x`: absolute x position on the map
    /// * `y`: absolute y position on the map
    pub fn chunk_idx_from_pos(&self, x: i32, y: i32) -> (usize, usize) {
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
    /// 
    /// # Arguments
    ///
    /// * `coord`: absolute x or y position on the map
    ///
    /// returns: absolute chunk index (x or y) on this axis corresponding to this coord
    pub fn chunk_pos(&self, coord: i32) -> i32 {
        let mut new_coord = coord;
        if new_coord < 0 {
            new_coord -= self.chunk_loader.get_chunk_size() as i32 - 1;
        }
        new_coord / self.chunk_loader.get_chunk_size() as i32
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
    pub fn non_texture_material_at(&self, xy: (i32, i32)) -> Material {
        self.get_chunk_immut(xy.0, xy.1).non_texture_material_at(xy.0, xy.1)
    }
    pub fn pop_at(&mut self, xy: (i32, i32)) -> Option<Block> {
        self.animations_buffer.add_tile_animation(TileAnimationType::mining(), xy);
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
    pub fn get_interactable_kind_at(&self, xy: (i32, i32)) -> Option<InteractableKind> {
        self.get_chunk_immut(xy.0, xy.1).get_interactable_kind_at(xy.0, xy.1)
    }
    pub fn get_interactable_inventory_at(&self, xy: (i32, i32)) -> Vec<(Storable, u32)> {
        self.get_chunk_immut(xy.0, xy.1).get_interactable_inventory_at(xy.0, xy.1).unwrap().clone()
    }
    pub fn load_interactable_at(&mut self, xy: (i32, i32), item: Storable, amount: u32) {
        self.get_chunk(xy.0, xy.1).load_interactable_at(xy.0, xy.1, item, amount)
    }
    pub fn unload_interactable_at(&mut self, xy: (i32, i32), item: &Storable, amount: u32) -> bool {
        self.get_chunk(xy.0, xy.1).unload_interactable_at(xy.0, xy.1, item, amount)
    }
    pub fn add_interactable(&mut self, inter: Interactable) -> bool {
        self.get_chunk(inter.get_position().0, inter.get_position().1)
            .add_interactable(inter)
    }
    pub fn get_interactable_targets_at(&self, xy: (i32, i32)) -> Vec<MobKind> {
        self.get_chunk_immut(xy.0, xy.1).get_interactable_targets_at(xy.0, xy.1)
    }
    pub fn set_interactable_targets_at(&mut self, xy: (i32, i32), targets: Vec<MobKind>) {
        self.get_chunk(xy.0, xy.1).set_interactable_targets_at(xy.0, xy.1, targets)
    }
    pub fn break_interactable_at(&mut self, xy: (i32, i32)) -> InteractableKind {
        self.get_chunk(xy.0, xy.1).break_interactable_at(xy.0, xy.1)
    }
    pub fn place_mob(&mut self, mob: Mob) {
        let xy = (mob.pos.x, mob.pos.y);
        self.get_chunk(xy.0, xy.1).add_mob(mob)
    }
    /// This function needs to take stray mobs into account,
    /// as it gets called during the mob movement stage,
    /// when (some of) the mobs are extracted from chunks
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
    /// Deals damage to the mob at the given position and removes them if they die.
    /// The mob must exist at the given position.
    /// Returns true if the mob was removed.
    pub fn damage_mob(&mut self, xy: AbsolutePos, damage: i32) -> bool {
        self.animations_buffer.add_tile_animation(TileAnimationType::receive_damage(), xy);
        if self.stray_mobs.len() > 0 {
            for i in 0..self.stray_mobs.len() {
                // found the mob in strays
                if self.stray_mobs[i].pos.x == xy.0 && self.stray_mobs[i].pos.y == xy.1 {
                    return if self.stray_mobs[i].receive_damage(damage) {
                        self.add_loot_at(self.stray_mobs[i].get_kind().loot(),
                                         (self.stray_mobs[i].pos.x, self.stray_mobs[i].pos.y));
                        self.stray_mobs.remove(i);
                        true
                    } else {
                        false
                    }
                }
            }
        }
        self.get_chunk(xy.0, xy.1).damage_mob(xy.0, xy.1, damage)
    }
    pub fn get_mob_kind_at(&self, xy: (i32, i32)) -> Option<MobKind> {
        for m in &self.stray_mobs {
            if m.pos.x == xy.0 && m.pos.y == xy.1 {
                return Some(*m.get_kind());
            }
        }
        self.get_chunk_immut(xy.0, xy.1).get_mob_kind_at(xy.0, xy.1)
    }
    pub fn pick_tile(&mut self) -> (i32, i32) {
        let min = self.min_loaded_idx();
        let max = self.max_loaded_idx();
        let x = rand::thread_rng().gen_range(min.0..=max.0);
        let y = rand::thread_rng().gen_range(min.1..=max.1);
        (x, y)
    }
}

/// Replays
impl Field {
    pub (crate) fn set_time(&mut self, time: f32) {
        self.time = time;
    }
    
    pub (crate) fn set_visible_tiles(&mut self, top_materials: &Vec<Vec<Material>>, tile_heights: &Vec<Vec<i32>>, player_pos: AbsolutePos) {
        // assuming square vectors of equal shape
        let center = (top_materials.len() - 1) / 2;
        let to_absolute_pos = |i: usize, j: usize| -> AbsolutePos {
            let relative_pos = (i as i32 - center as i32, j as i32 - center as i32);
            (relative_pos.0 + player_pos.0, relative_pos.1 + player_pos.1)
        };
        let curr_chunk: AbsoluteChunkPos = (self.chunk_pos(player_pos.0), self.chunk_pos(player_pos.1));
        self.load(curr_chunk.0, curr_chunk.1);

        for i in 0..top_materials.len() {
            for j in 0..top_materials[0].len() {
                // because setting a full tile is not an allowed operation, we pop and push materials to get the desired result
                let xy = to_absolute_pos(i, j);
                // fix len
                while self.len_at(xy) >= tile_heights[i][j] as usize {
                    self.pop_at(xy);
                }
                // put correct top material
                while self.len_at(xy) < tile_heights[i][j] as usize {
                    self.push_at(Block::new(top_materials[i][j]), xy);
                }
            }
        }
    }
    
    pub (crate) fn set_mobs(&mut self, mobs: Vec<Mob>) {
        // destroy all mobs
        self.detach_mobs();
        self.stray_mobs.clear();
        
        for mob in mobs {
            self.place_mob(mob);
        }
    }
}

pub const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
pub type AbsolutePos = (i32, i32);
pub type RelativePos = (i32, i32);  // relative to the player
pub type AbsoluteChunkPos = (i32, i32);

pub fn absolute_to_relative((x, y): AbsolutePos, player: &Player) -> RelativePos {
    (x - player.x, y - player.y)
}

pub fn relative_to_absolute((x, y): RelativePos, player: &Player) -> AbsolutePos {
    (x + player.x, y + player.y)
}
