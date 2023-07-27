mod common;

use serde::Serialize;
use crate::common::Data;

use minecraft::crafting::items::Item;
use minecraft::crafting::items::Item::*;
use minecraft::crafting::material::Material;
use minecraft::crafting::material::Material::*;
use minecraft::crafting::storable::Storable;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::character::player::Player;
use winit::event::VirtualKeyCode::*;


#[test]
fn test_start_field_serde() {
    // Initialize a Field instance
    let field = Field::new(8, None);
    // Serialize the Field instance to a JSON string
    let serialized = serde_yaml::to_string(&field).unwrap();

    println!("{}", serialized[0..1000].to_string());

    // Deserialize the JSON string back to a Field
    let deserialized: Field = serde_yaml::from_str(&serialized).unwrap();

    // Check that the deserialized Field is the same as the original
    assert_eq!(field, deserialized);
}
