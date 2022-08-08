#[macro_use] extern crate text_io;
extern crate core;

use crate::field::Field;
use crate::items::item_by_name;
use crate::material::{Material, materials};
use crate::player::Player;

mod field;
mod tile;
mod block;
mod material;
mod player;
mod inventory;
mod items;

fn main() {
    let mut player = Player::new();
    let mut field = Field::new();
    println!("Welcome to minecraft (very alpha version)");
    loop {
        field.render(&player);
        player.render_inventory();
        println!("input action type");
        let action: String = read!();
        match action.as_str() {
            "w" => {
                println!("input direction");
                let dir: String = read!();
                player.walk(&*dir);
            },
            "m" => {
                println!("input relative coords");
                let x: i32 = read!();
                let y: i32 = read!();
                player.mine(&mut field, x, y)
            },
            "p" => {
                println!("input relative coords and material");
                let x: i32 = read!();
                let y: i32 = read!();
                let material_string: String = read!();
                let material = Material::from_string(&material_string);
                match material {
                    None => println!("unrecognized material"),
                    Some(material) => player.place(&mut field, x, y, material)
                }
            },
            "c" => {
                println!("input the item to craft");
                let item_string: String = read!("{}\n");
                let inferred_item = item_by_name(&item_string);
                match inferred_item {
                    None => println!("unknown item (material != item)"),
                    Some(item) => {
                        if !player.can_craft(&item) {
                            println!("you cannot craft that")
                        } else {
                            player.craft(item);
                            println!("crafting successful")
                        }
                    }
                }
            },
            _ => println!("action not recognized")
        };
    }
}
