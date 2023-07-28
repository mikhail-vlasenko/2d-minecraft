mod common;

use std::fs::{File, read_to_string, remove_file};
use std::io::{Read, Write};
use std::path::Path;
use postcard;
extern crate alloc;
use alloc::vec::Vec;
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
    let path = Path::new("game_saves/test_field2.postcard");
    save_game(&field, &path);
    let loaded_field = load_game(&path);
    assert_eq!(field, loaded_field);

    let another_field = Field::new(8, None);
    // in an extremely unlikely event these two fields can be equal (because random generation coincided)
    assert_ne!(loaded_field, another_field);
}
