use std::fs;
use std::fs::File;
use std::io::Read;
use crate::map_generation::block::Block;
use crate::map_generation::tile::Tile;
use strum::IntoEnumIterator;
use crate::crafting::material::Material;


pub fn read_file(file_path: String) -> Vec<Vec<Tile>> {
    let mut file = File::open(file_path)
        .expect("File not found");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Error while reading file");

    let mut lines = contents.lines();
    let size: usize = lines.next().unwrap().parse().unwrap();

    let mut tiles = Vec::new();
    for i in 0..size {
        tiles.push(Vec::new());
        for _ in 0..size {
            tiles[i].push(Tile::new());
        }
    }

    for _ in 0..5 {
        lines.next();
        for i in 0..size {
            let mut line = lines.next().unwrap().chars();
            for j in 0..size {
                let material = from_glyph(String::from(line.next().unwrap()));
                if material.is_some() {
                    tiles[i][j].push(Block { material: material.unwrap() })
                }
            }
        }
    }
    tiles
}

fn from_glyph(glyph: String) -> Option<Material> {
    if glyph == "e" {
        return None
    }
    for m in Material::iter() {
        if m.glyph() == glyph {
            return Some(m);
        }
    }
    panic!()
}
