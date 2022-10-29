use std::cell::{Ref, RefCell, RefMut};
use std::panic;
use std::rc::Rc;
use crate::{Material, Player};
use crate::map_generation::tile::{randomly_augment, Tile};
use crate::map_generation::block::Block;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::chunk_loader::ChunkLoader;
use crate::map_generation::mobs::mob::{Mob, MobKind, Position};


/// The playing grid
pub struct Field {
    /// hashmap for all generated chunks. key: encoded xy position, value: the chunk
    chunk_loader: ChunkLoader,
    /// tiles from these chunks can be accessed
    loaded_chunks: Vec<Vec<Rc<RefCell<Chunk>>>>,
    /// position of the center of the currently loaded chunks
    central_chunk: (i32, i32),
    chunk_size: usize,
    /// how far from the player's chunk the chunks are loaded
    loading_distance: usize,
}

impl Field {
    pub fn new() -> Self {
        let loading_distance = 4;
        let chunk_size = 16;
        let chunk_loader = ChunkLoader::new(loading_distance);
        let loaded_chunks = Vec::new();
        let central_chunk = (0, 0);

        let mut field = Self{
            chunk_loader,
            loaded_chunks,
            central_chunk,
            chunk_size,
            loading_distance,
        };
        field.load(central_chunk.0, central_chunk.1);
        field
    }

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
        let chunk_coord = self.chunk_pos(coord);  // check -64
        let left_top = center - self.loading_distance as i32;
        let idx = chunk_coord - left_top;
        if idx < 0 || idx as usize > self.loading_distance * 2 {
            panic!("Attempted access to an unloaded chunk (idx = {})", idx);
        }
        idx as usize
    }

    /// Chunk's index for this absolute map coordinate
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
    /// * `radius`: how far field from player is included
    ///
    /// returns: (2d Vector: the list of positions)
    pub fn texture_indices(&self, player: &Player, material: Material, radius: usize) -> Vec<(i32, i32)> {
        let r = radius as i32;
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x - r)..=(player.x + r) {
            for j in (player.y - r)..=(player.y + r) {
                if self.get_chunk_immut(i, j).top_at(i, j).material == material {
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
                if self.get_chunk_immut(i, j).len_at(i, j) == height {
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

    /// Load a new set of loaded_chunks around the given chunk
    pub fn load(&mut self, chunk_x: i32, chunk_y:i32) {
        self.chunk_loader.generate_close_chunks(chunk_x, chunk_y);
        self.loaded_chunks = self.chunk_loader.load_around(chunk_x, chunk_y);
        self.central_chunk = (chunk_x, chunk_y);
    }

    pub fn step_mobs(&mut self, player: &Player) {
        let mut mobs = Vec::new();
        for i in 0..self.loaded_chunks.len() {
            for j in 0..self.loaded_chunks[i].len() {
                mobs.extend(self.loaded_chunks[i][j].borrow_mut().transfer_mobs());
            }
        }
        for mut m in mobs {
            m.act(&self, player, self.min_loaded_idx(), self.max_loaded_idx());
            let (x_chunk, y_chunk) = self.chunk_idx_from_pos(m.pos.x, m.pos.y);
            self.loaded_chunks[x_chunk][y_chunk].borrow_mut().add_mob(m);
        }
    }
}

/// API for Tile interaction, x and y are absolute map positions.
impl Field {
    pub fn len_at(&self, x: i32, y: i32) -> usize {
        self.get_chunk_immut(x, y).len_at(x, y)
    }
    pub fn push_at(&mut self, block: Block, x: i32, y: i32) {
        self.get_chunk(x, y).push_at(block, x, y)
    }
    pub fn top_material_at(&self, x: i32, y: i32) -> Material {
        self.get_chunk_immut(x, y).top_material_at(x, y)
    }
    pub fn pop_at(&mut self, x: i32, y: i32) -> Option<Block> {
        self.get_chunk(x, y).pop_at(x, y)
    }
    pub fn full_at(&self, x: i32, y: i32) -> bool {
        self.get_chunk_immut(x, y).full_at(x, y)
    }
    pub fn mob_at(&self, x: i32, y: i32) -> bool {
        self.get_chunk_immut(x, y).mob_at(x, y)
    }
}
