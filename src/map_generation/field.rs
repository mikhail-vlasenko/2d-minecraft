use std::collections::HashMap;
use std::rc::Rc;
use crate::{Material, Player};
use crate::map_generation::tile::{randomly_augment, Tile};
use rand::prelude::*;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::chunk_loader::ChunkLoader;


/// The playing grid
pub struct Field {
    pub tiles: Vec<Vec<Tile>>,
    /// hashmap for all generated chunks. key: encoded xy position, value: the chunk
    pub chunk_loader: ChunkLoader,
    /// tiles from these chunks can be accessed
    pub loaded_chunks: Vec<Vec<Rc<Chunk>>>,
    /// top left corner of the currently loaded chunks (not needed?)
    anchor_coords: (i32, i32),
    chunk_size: usize,
    /// how far from the player's chunk the chunks are loaded
    loading_distance: usize,
}

impl Field {
    pub fn new() -> Self {
        let chunk_size = 16;
        let tiles = Vec::new();
        let chunk_loader = ChunkLoader::new();
        let loaded_chunks = Vec::new();
        let anchor_coords = (0,0);

        let mut field = Self{
            tiles,
            chunk_loader,
            loaded_chunks,
            anchor_coords,
            chunk_size,
            loading_distance: 1,
        };
        field.load(anchor_coords.0, anchor_coords.1);
        field
    }

    /// Display as glyphs
    pub fn render(&self, player: &Player) {
        for i in 0..self.tiles.len() {
            for j in 0..self.tiles[i].len() {
                if i as i32 == player.x && j as i32 == player.y {
                    print!("P");
                } else {
                    print!("{}", self.tiles[i][j]);
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
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x as usize - radius)..=(player.x as usize + radius) {
            for j in (player.y as usize - radius)..=(player.y as usize + radius) {
                if self.tiles[i][j].top().material == material {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
    }

    /// Makes a list of positions of blocks of given height around the player.
    pub fn depth_indices(&self, player: &Player, height: usize, radius: usize) -> Vec<(i32, i32)> {
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x as usize - radius)..=(player.x as usize + radius) {
            for j in (player.y as usize - radius)..=(player.y as usize + radius) {
                if self.tiles[i][j].len() == height {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
    }

    pub fn load(&mut self, chunk_x: i32, chunk_y:i32) {
        self.loaded_chunks = self.chunk_loader.load_around(chunk_x, chunk_y);
    }
}

// impl<'a> Field<'a> {
//     pub fn load_close_chunks(&'a mut self, chunk_x: i32, chunk_y: i32) {
//         for x in 0..=(2 * self.loading_distance) {
//             for y in 0..=(2 * self.loading_distance) {
//                 let curr_x = chunk_x - self.loading_distance as i32 + x as i32;
//                 let curr_y = chunk_y - self.loading_distance as i32 + y as i32;
//                 let key = Self::encode_key(curr_x, curr_y);
//                 self.loaded_chunks[0].push(self.chunks.get(&key).unwrap());
//             }
//         }
//     }
// }

// impl Field<'_> {
//     pub fn load_close_chunks(&mut self, chunk_x: i32, chunk_y: i32) {
//         for x in 0..=(2 * self.loading_distance) {
//             for y in 0..=(2 * self.loading_distance) {
//                 let curr_x = chunk_x - self.loading_distance as i32 + x as i32;
//                 let curr_y = chunk_y - self.loading_distance as i32 + y as i32;
//                 let key = Self::encode_key(curr_x, curr_y);
//                 self.loaded_chunks[0].push(self.chunks.get(&key).unwrap());
//             }
//         }
//     }
// }
