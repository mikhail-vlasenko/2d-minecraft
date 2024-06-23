use std::cell::RefCell;
use std::collections::HashMap;
use crate::map_generation::chunk::Chunk;
use std::rc::Rc;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use crate::SETTINGS;


#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkLoader {
    #[serde(with = "hash_map_serde")]
    chunks: HashMap<i64, Rc<RefCell<Chunk>>>,
    /// in chunks, not tiles
    loading_distance: usize,
    /// in tiles
    chunk_size: usize,
}

impl ChunkLoader {
    pub fn new(loading_distance: usize) -> Self {
        let chunk_size = SETTINGS.read().unwrap().field.chunk_size as usize;
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

    /// chunk_x and chunk_y are in chunk coordinates, not tiles
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

impl PartialEq for ChunkLoader {
    fn eq(&self, other: &Self) -> bool {
        if self.loading_distance != other.loading_distance { return false; }
        if self.chunk_size != other.chunk_size { return false; }
        if self.chunks.len() != other.chunks.len() { return false; }

        for (key, value) in &self.chunks {
            match other.chunks.get(key) {
                Some(other_value) => {
                    if *value.borrow() != *other_value.borrow() {
                        return false;
                    }
                },
                None => return false,
            }
        }
        true
    }
}

mod hash_map_serde {
    use super::*;
    pub fn serialize<S: Serializer>(map: &HashMap<i64, Rc<RefCell<Chunk>>>, serializer: S) -> Result<S::Ok, S::Error> {
        let mut chunks = Vec::new();
        for (idx, chunk) in map.iter() {
            chunks.push((idx.clone(), chunk.borrow().clone()));
        }
        serializer.collect_seq(chunks)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<i64, Rc<RefCell<Chunk>>>, D::Error>
        where D: Deserializer<'de>
    {
        let mut map = HashMap::new();
        for (idx, chunk) in Vec::<(i64, Chunk)>::deserialize(deserializer)? {
            map.insert(idx, Rc::new(RefCell::new(chunk)));
        }
        Ok(map)
    }
}
