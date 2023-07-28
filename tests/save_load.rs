mod common;

use std::fs::{File, read_to_string, remove_dir_all, remove_file};
use std::io::{Read, Write};
use std::path::Path;
use postcard;
extern crate alloc;
use alloc::vec::Vec;
use minecraft::character::player::Player;
use serde::Serialize;
use crate::common::Data;

use minecraft::map_generation::field::Field;
use minecraft::map_generation::save_load::{load_game, save_game};
use winit::event::VirtualKeyCode::*;


#[test]
fn test_start_field_serde_yaml() {
    // Initialize a Field instance
    let field = Field::new(8, None);
    // Serialize the Field instance to a JSON string
    let serialized = serde_yaml::to_string(&field).unwrap();

    // Deserialize the JSON string back to a Field
    let mut deserialized: Field = serde_yaml::from_str(&serialized).unwrap();

    deserialized.load(0, 0);

    // Check that the deserialized Field is the same as the original
    assert_eq!(field, deserialized);
    // additional checks
    assert_eq!(*field.get_chunk_immut(1, 1), *deserialized.get_chunk_immut(1, 1));
    assert_eq!(field.top_material_at((-5, -7)), deserialized.top_material_at((-5, -7)));
    assert_eq!(field.get_time(), deserialized.get_time());
    deserialized.pop_at((0, 0));
    assert_ne!(field, deserialized);
}

#[test]
fn test_field_serialization_file_postcard() {
    // Initialize a Field instance
    let field = Field::new(8, None);

    // Serialize the Field instance to a byte array using Postcard
    let serialized: Vec<u8> = postcard::to_allocvec(&field).unwrap();

    // Write the serialized data to a file
    let path = Path::new("game_saves/test_field.postcard");
    let mut file = File::create(&path).unwrap();
    file.write_all(&serialized).unwrap();

    // Read the data back from the file
    let mut file = File::open(&path).unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    // Deserialize the byte array back to a Field using Postcard
    let mut deserialized: Field = postcard::from_bytes(&data).unwrap();
    deserialized.load(0, 0);

    // Check that the deserialized Field is the same as the original
    assert_eq!(field, deserialized);
    remove_file(&path).unwrap();
}

#[test]
fn test_full_save_load_functions() {
    let field = Field::new(8, None);
    let player = Player::new(&field);

    let path = Path::new("game_saves/test_save2");
    save_game(&field, &player, &path);
    let (loaded_field, loaded_player) = load_game(&path);
    assert_eq!(field, loaded_field);
    assert_eq!(player, loaded_player);

    let another_field = Field::new(8, None);
    // in an extremely unlikely event these two fields can be equal (because random generation coincided)
    assert_ne!(loaded_field, another_field);

    remove_dir_all(&path).unwrap();
}

#[test]
fn test_save_load_mobs() {
    let mut data = Data::new();
    data.spawn_mobs(10, true);
    data.step_mobs();
    data.step_mobs();

    let path = Path::new("game_saves/test_save3");
    save_game(&data.field, &data.player, &path);
    let (loaded_field, loaded_player) = load_game(&path);
    assert_eq!(data.field, loaded_field);
    assert_eq!(data.player, loaded_player);

    let another_field = Field::new(8, None);
    assert_ne!(loaded_field, another_field);

    remove_dir_all(&path).unwrap();
}

#[test]
fn test_save_load_after_moves(){
    let mut data = Data::new();
    data.spawn_mobs(10, true);
    data.spawn_mobs(3, false);

    for _ in 0..10 {
        data.mine();
        data.random_action();
        data.step_time();
    }

    let path = Path::new("game_saves/test_save4");
    save_game(&data.field, &data.player, &path);
    let (loaded_field, loaded_player) = load_game(&path);
    assert_eq!(data.field, loaded_field);
    assert_eq!(data.player, loaded_player);

    let another_player = Player::new(&loaded_field);
    assert_ne!(loaded_player, another_player);

    remove_dir_all(&path).unwrap();
}
