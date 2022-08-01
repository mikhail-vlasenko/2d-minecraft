#[macro_use] extern crate text_io;

use crate::field::{Field, init_field};
use crate::material::{Material, materials};
use crate::player::Player;

mod field;
mod tile;
mod block;
mod material;
mod player;

fn main() {
    let mut player = Player{
        x: 4,
        y: 4,
        z: 3,
    };
    let mut field = init_field();
    println!("Welcome to minecraft (very alpha version)");
    loop {
        field.render(&player);
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
