use std::cell::{RefCell, RefMut};
use rand::random;
use crate::map_generation::block::Block;
use crate::map_generation::tile::{randomly_augment, Tile};
use crate::Material;

pub struct Chunk {
    tiles: Vec<Vec<Tile>>,
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
        Self{
            tiles,
            size,
        }
    }

    /// Indices of the tile within a chunk. Any chunk, nit necessarily this one
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
        randomly_augment(&mut tile, &Tile::make_rock, 0.05);
        randomly_augment(&mut tile, &Tile::add_tree, 0.1);
        randomly_augment(&mut tile, &Tile::make_iron, 0.2);
        tile
    }
}

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
}