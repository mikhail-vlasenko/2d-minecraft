#[macro_use] extern crate text_io;
extern crate core;

use crate::field::Field;
use crate::material::Material;
use crate::player::Player;
use crate::storable::Storable;
use crate::graphics::state::run;

mod field;
mod tile;
mod block;
mod material;
mod player;
mod inventory;
mod items;
mod storable;
mod graphics;



fn main() {
    pollster::block_on(run());

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
                // only try material, cause we want to place it
                let material = Material::try_from(material_string);
                match material {
                    Err(_) => println!("unrecognized material"),
                    Ok(material) => player.place(&mut field, x, y, material)
                }
            },
            "c" => {
                println!("input the item to craft");
                let item_string: String = read!("{}\n");
                let inferred_item = Storable::try_from(item_string);
                match inferred_item {
                    Err(_) => println!("unknown item (material != item)"),
                    Ok(item) => {
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
