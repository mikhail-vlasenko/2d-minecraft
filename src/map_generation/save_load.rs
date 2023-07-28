use std::{fs, io};
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;
use crate::character::player::Player;
use crate::map_generation::field::Field;


impl Field {
    pub fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    pub fn from_binary_string(data: &Vec<u8>) -> Self {
        let mut deserialized: Field = postcard::from_bytes(data).unwrap();
        let central_chunk = deserialized.get_central_chunk();
        deserialized.load(central_chunk.0, central_chunk.1);
        deserialized
    }
}

impl Player {
    pub fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    pub fn from_binary_string(data: &Vec<u8>) -> Self {
        postcard::from_bytes(data).unwrap()
    }
}

pub fn save_game(field: &Field, player: &Player, path: &Path) {
    create_dir_all(&path).unwrap();

    let serialized_field = field.to_binary_string();
    let serialized_player = player.to_binary_string();

    let field_path = path.join("field.postcard");
    let player_path = path.join("player.postcard");

    let mut file = File::create(field_path).unwrap();
    file.write_all(&serialized_field).unwrap();

    let mut file = File::create(player_path).unwrap();
    file.write_all(&serialized_player).unwrap();
}

pub fn load_game(path: &Path) -> (Field, Player) {
    let field_path = path.join("field.postcard");
    let player_path = path.join("player.postcard");

    let mut file = File::open(field_path).unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    let field = Field::from_binary_string(&data);

    let mut file = File::open(player_path).unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    let player = Player::from_binary_string(&data);

    (field, player)
}

pub fn get_directories(path: &Path) -> io::Result<Vec<String>> {
    let mut directories = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        // If the entry is a directory, add it to the list
        if path.is_dir() {
            if let Some(filename) = path.file_name() {
                directories.push(filename.to_string_lossy().into_owned());
            }
        }
    }
    directories.sort();

    Ok(directories)
}
