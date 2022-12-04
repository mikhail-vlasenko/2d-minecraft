use std::cell::RefCell;
use std::collections::HashMap;
use crate::map_generation::chunk::Chunk;
use std::rc::Rc;
use crate::map_generation::mobs::spawning::spawn_hostile;


pub struct ChunkLoader {
    chunks: HashMap<i64, Rc<RefCell<Chunk>>>,
    loading_distance: usize,
    chunk_size: usize,
}

impl ChunkLoader {
    pub fn new(loading_distance: usize) -> Self {
        let chunk_size = 16;
        let chunks = HashMap::new();

        let mut loader = Self {
            chunks,
            loading_distance,
            chunk_size,
        };
        loader.generate_close_chunks(0, 0);
        loader
    }
    
    pub fn with_starting_chunk(loading_distance: usize, chunk: Chunk) -> Self {
        let mut loader = Self::new(loading_distance);
        loader.chunks.insert(Self::encode_key(0, 0), Rc::new(RefCell::new(chunk)));
        loader
    }

    pub fn generate_close_chunks(&mut self, chunk_x: i32, chunk_y: i32) {
        for x in (chunk_x - self.loading_distance as i32)..=(chunk_x + self.loading_distance as i32) {
            for y in (chunk_y - self.loading_distance as i32)..=(chunk_y + self.loading_distance as i32) {
                let key = Self::encode_key(x, y);
                if !self.chunks.contains_key(&key) {
                    let generated = Chunk::new(self.chunk_size);
                    // spawn_hostile(&mut generated, x, y);
                    self.chunks.insert(key, Rc::new(RefCell::new(generated)));
                }
            }
        }
    }

    pub fn load_around(&self, chunk_x: i32, chunk_y: i32) -> Vec<Vec<Rc<RefCell<Chunk>>>> {
        let mut loaded = Vec::new();
        for x in 0..=(2 * self.loading_distance) {
            loaded.push(Vec::new());
            for y in 0..=(2 * self.loading_distance) {
                let curr_x = chunk_x - self.loading_distance as i32 + x as i32;
                let curr_y = chunk_y - self.loading_distance as i32 + y as i32;
                let key = Self::encode_key(curr_x, curr_y);
                loaded[x].push(Rc::clone(self.chunks.get(&key).unwrap()));
            }
        }
        loaded
    }

    fn encode_key(x: i32, y: i32) -> i64 {
        let low = x as i64;
        let high = (y as i64) << 32;
        low + high
    }
}