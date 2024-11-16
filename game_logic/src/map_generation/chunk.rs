use std::mem;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, Deserialize};
use crate::crafting::interactable::{Interactable, InteractableKind};
use crate::map_generation::block::Block;
use crate::map_generation::mobs::mob::Mob;
use crate::map_generation::tile::{randomly_augment, Tile};
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;


#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct Chunk {
    tiles: Vec<Vec<Tile>>,
    /// all mobs on this chunk are stored here
    mobs: Vec<Mob>,
    /// all furnaces, chests, turrets, etc on this chunk are stored here
    /// turrets are special because they have to be extracted from the chunk to act
    interactables: Vec<Interactable>,
    /// length of the side of the square chunk
    size: usize,
}

impl Chunk {
    pub fn new(size: usize, chunk_seed: u64) -> Self {
        let mut chunk_rng = ChaCha8Rng::seed_from_u64(chunk_seed);
        let mut tiles = Vec::new();
        for i in 0..size {
            tiles.push(Vec::new());
            for _ in 0..size {
                tiles[i].push(Self::gen_tile(&mut chunk_rng));
            }
        }
        let mut chunk = Self::from(tiles);
        chunk.add_structures(&mut chunk_rng);
        chunk
    }

    pub fn from(tiles: Vec<Vec<Tile>>) -> Self {
        let size = tiles.len();
        Self {
            tiles,
            size,
            mobs: Vec::new(),
            interactables: Vec::new(),
        }
    }

    /// Indices of the tile within a chunk. Any chunk, not necessarily this one
    ///
    /// # Arguments
    ///
    /// * `x`: absolute x position on the map
    /// * `y`: absolute y position on the map
    pub fn indices_in_chunk(&self, x: i32, y: i32) -> (usize, usize) {
        let mut inner_x = x % self.size as i32;
        let mut inner_y = y % self.size as i32;
        if inner_x < 0 {
            inner_x += self.size as i32;
        }
        if inner_y < 0 {
            inner_y += self.size as i32;
        }
        (inner_x as usize, inner_y as usize)
    }

    /// Randomly generate a Tile (a cell on the field)
    pub fn gen_tile(chunk_rng: &mut impl Rng) -> Tile {
        let mut tile = Tile::make_dirt();
        randomly_augment(&mut tile, &Tile::make_rock,
                         SETTINGS.read().unwrap().field.generation.rock_proba, chunk_rng);
        randomly_augment(&mut tile, &Tile::add_tree,
                         SETTINGS.read().unwrap().field.generation.tree_proba, chunk_rng);
        randomly_augment(&mut tile, &Tile::make_iron,
                         SETTINGS.read().unwrap().field.generation.iron_proba, chunk_rng);
        randomly_augment(&mut tile, &Tile::make_diamond,
                         SETTINGS.read().unwrap().field.generation.diamond_proba, chunk_rng);
        randomly_augment(&mut tile, &Tile::make_full_diamond,
                         0.5, chunk_rng);  // applies only if there is a diamond already
        tile
    }

    pub fn add_mob(&mut self, mob: Mob) {
        self.mobs.push(mob);
    }

    pub fn get_mobs(&self) -> &Vec<Mob> {
        &self.mobs
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn transfer_mobs(&mut self) -> Vec<Mob> {
        mem::take(&mut self.mobs)
    }

    /// Turrets need to be extracted from the chunk to act,
    /// as they need information from other chunks (mobs)
    pub fn transfer_turrets(&mut self) -> Vec<Interactable> {
        let mut turrets = Vec::new();
        let mut i = 0;
        while i < self.interactables.len() {
            if self.interactables[i].get_kind().is_turret() {
                turrets.push(self.interactables.remove(i));
            } else {
                i += 1;
            }
        }
        turrets
    }
}

/// API for Tile interaction, x and y are absolute map positions.
/// Should be called on an appropriate chunk.
impl Chunk {
    pub fn len_at(&self, x: i32, y: i32) -> usize {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].len()
    }
    pub fn top_at(&self, x: i32, y: i32) -> &Block {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].top()
    }
    pub fn top_material_at(&self, x: i32, y: i32) -> Material {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].top_material()
    }
    pub fn non_texture_material_at(&self, x: i32, y: i32) -> Material {
        let inner = self.indices_in_chunk(x, y);
        let i = self.tiles[inner.0][inner.1].len() - 2;
        match self.tiles[inner.0][inner.1].top_material() {
            Material::Texture(_) => self.tiles[inner.0][inner.1].blocks[i].material,
            m => m,
        }
    }
    pub fn push_at(&mut self, block: Block, x: i32, y: i32) {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].push(block)
    }
    pub fn push_all_at(&mut self, blocks: Vec<Block>, x: i32, y: i32) {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].push_all(blocks)
    }
    pub fn pop_at(&mut self, x: i32, y: i32) -> Option<Block> {
        let inner = self.indices_in_chunk(x, y);
        let block = self.tiles[inner.0][inner.1].pop();
        // land mob on this tile
        if self.is_occupied(x, y) {
            for mob in &mut self.mobs {
                if mob.pos.x == x && mob.pos.y == y {
                    mob.pos.z = self.tiles[inner.0][inner.1].len();
                }
            }
        }
        block
    }
    pub fn full_at(&self, x: i32, y: i32) -> bool {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].len() >= 5
    }
    pub fn get_loot_at(&self, x: i32, y: i32) -> &Vec<Storable> {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].get_loot()
    }
    pub fn add_loot_at(&mut self, new: Vec<Storable>, x: i32, y: i32) {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].add_loot(new)
    }
    pub fn gather_loot_at(&mut self, x: i32, y: i32) -> Vec<Storable> {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].gather_loot()
    }
    pub fn is_occupied(&self, x: i32, y: i32) -> bool {
        for m in &self.mobs {
            if m.pos.x == x && m.pos.y == y {
                return true;
            }
        }
        false
    }
    /// Returns true if the mob died
    pub fn damage_mob(&mut self, x: i32, y: i32, damage: i32) -> bool {
        for i in 0..self.mobs.len() {
            if self.mobs[i].pos.x == x && self.mobs[i].pos.y == y {
                return if self.mobs[i].receive_damage(damage) {
                    self.add_loot_at(self.mobs[i].get_kind().loot(), self.mobs[i].pos.x, self.mobs[i].pos.y);

                    self.mobs.remove(i);
                    true
                } else {
                    false
                };
            }
        }
        false
    }
    pub fn get_mob_kind_at(&self, x: i32, y: i32) -> Option<MobKind> {
        for mob in &self.mobs {
            if mob.pos.x == x && mob.pos.y == y {
                return Some(*mob.get_kind());
            }
        }
        None
    }
}

/// API for using the chunk's interactables
impl Chunk {
    pub fn get_interactable_kind_at(&self, x: i32, y: i32) -> Option<InteractableKind> {
        for inter in &self.interactables {
            if inter.get_position().0 == x && inter.get_position().1 == y {
                return Some(inter.get_kind());
            }
        }
        None
    }
    pub fn get_interactable_inventory_at(&self, x: i32, y: i32) -> Option<&Vec<(Storable, u32)>> {
        for inter in &self.interactables {
            if inter.get_position().0 == x && inter.get_position().1 == y {
                return Some(inter.get_inventory().get_all());
            }
        }
        None
    }
    pub fn load_interactable_at(&mut self, x: i32, y: i32, item: Storable, amount: u32) {
        for inter in &mut self.interactables {
            if inter.get_position().0 == x && inter.get_position().1 == y {
                inter.load_item(item, amount);
                return;
            }
        }
        panic!("Tried to load interactable at {}, {} but there was none", x, y)
    }
    pub fn unload_interactable_at(&mut self, x: i32, y: i32, item: &Storable, amount: u32) -> bool {
        for inter in &mut self.interactables {
            if inter.get_position().0 == x && inter.get_position().1 == y {
                return inter.unload_item(item, amount);
            }
        }
        false
    }
    pub fn add_interactable(&mut self, interactable: Interactable) -> bool {
        if self.get_interactable_kind_at(
            interactable.get_position().0, interactable.get_position().1).is_none() {
            self.interactables.push(interactable);
            true
        } else {
            false
        }
    }
    pub fn break_interactable_at(&mut self, x: i32, y: i32) -> InteractableKind {
        for i in 0..self.interactables.len() {
            if self.interactables[i].get_position().0 == x && self.interactables[i].get_position().1 == y {
                let inter = self.interactables.remove(i);
                for (item, amount) in inter.get_inventory().get_all() {
                    self.add_loot_at(vec![item.clone(); *amount as usize], x, y);
                }
                return inter.get_kind();
            }
        }
        panic!("Tried to break interactable at {}, {} but there was none", x, y)
    }
    pub fn get_interactable_targets_at(&self, x: i32, y: i32) -> Vec<MobKind> {
        for inter in &self.interactables {
            if inter.get_position().0 == x && inter.get_position().1 == y {
                return inter.get_targets();
            }
        }
        panic!("Tried to get interactable targets at {}, {} but there was none", x, y)
    }
    pub fn set_interactable_targets_at(&mut self, x: i32, y: i32, targets: Vec<MobKind>) {
        for inter in &mut self.interactables {
            if inter.get_position().0 == x && inter.get_position().1 == y {
                inter.set_targets(targets);
                return;
            }
        }
        panic!("Tried to set interactable targets at {}, {} but there was none", x, y)
    }
}