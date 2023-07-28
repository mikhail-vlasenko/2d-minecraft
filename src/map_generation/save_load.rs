use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use crate::map_generation::field::Field;

impl Field {
    pub fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    pub fn from_binary_string(data: Vec<u8>) -> Self {
        let mut deserialized: Field = postcard::from_bytes(&data).unwrap();
        let central_chunk = deserialized.get_central_chunk();
        deserialized.load(central_chunk.0, central_chunk.1);
        deserialized
    }
}

pub fn save_game(field: &Field, path: &Path) {
    let serialized = field.to_binary_string();
    let mut file = File::create(path).unwrap();
    file.write_all(&serialized).unwrap();
}

pub fn load_game(path: &Path) -> Field {
    let mut file = File::open(path).unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    let mut field = Field::from_binary_string(data);
    field.load(0, 0);
    field
}
