use std::cell::{RefCell, RefMut};
use rand::random;
use crate::crafting::interactable::Interactable;
use crate::map_generation::block::Block;
use crate::map_generation::mobs::mob::Mob;
use crate::map_generation::tile::{randomly_augment, Tile};
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::map_generation::read_chunk::read_file;
use crate::SETTINGS;

pub struct Chunk {
    tiles: Vec<Vec<Tile>>,
    mobs: Vec<Mob>,
    size: usize,
}

impl Chunk {
    pub fn new(size: usize) -> Self {
        let mut tiles = Vec::new();
        for i in 0..size {
            tiles.push(Vec::new());
            for _ in 0..size {
                tiles[i].push(Self::gen_tile());
            }
        }
        let mobs = Vec::new();
        Self {
            tiles,
            size,
            mobs,
        }
    }

    pub fn from(tiles: Vec<Vec<Tile>>) -> Self {
        let size = tiles.len();
        Self {
            tiles,
            size,
            mobs: Vec::new(),
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
    pub fn gen_tile() -> Tile {
        let mut tile = Tile::make_dirt();
        randomly_augment(&mut tile, &Tile::make_rock,
                         SETTINGS.field.generation.rock_proba);
        randomly_augment(&mut tile, &Tile::add_tree,
                         SETTINGS.field.generation.tree_proba);
        randomly_augment(&mut tile, &Tile::make_iron,
                         SETTINGS.field.generation.iron_proba);
        randomly_augment(&mut tile, &Tile::make_diamond,
                         SETTINGS.field.generation.diamond_proba);
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
        let mut mobs = Vec::new();
        for _ in 0..self.mobs.len() {
            mobs.push(self.mobs.pop().unwrap());
        }
        mobs
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
    pub fn push_at(&mut self, block: Block, x: i32, y: i32) {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].push(block)
    }
    pub fn pop_at(&mut self, x: i32, y: i32) -> Option<Block> {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].pop()
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
    pub fn get_interactable_at(&self, x: i32, y: i32) -> Option<Interactable> {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].get_interactable()
    }
    pub fn add_interactable_at(&mut self, interactable: Interactable, x: i32, y: i32) -> bool {
        let inner = self.indices_in_chunk(x, y);
        self.tiles[inner.0][inner.1].add_interactable(interactable)
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
}