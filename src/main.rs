#[macro_use] extern crate text_io;

use crate::field::Field;
use crate::material::{Material, materials};
use crate::player::Player;

mod field;
mod tile;
mod block;
mod material;
mod player;
mod inventory;
mod crafting;
mod items;
mod lol;

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
                let material = match material_string.as_str() {
                    "dirt" => &materials::DIRT,
                    "stone" => &materials::STONE,
                    _ => &materials::AIR,
                };
                player.place(&mut field, x, y, material);
            },
            _ => println!("action not recognized")
        };
    }
}
