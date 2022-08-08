use crate::Player;
use crate::tile::{randomly_add, Tile};
use rand::prelude::*;

pub struct Field {
    pub tiles: Vec<Vec<Tile>>,
}


impl Field {
    pub fn new() -> Self {
        let mut tiles = Vec::new();
        for i in 0..10 {
            tiles.push(Vec::new());
            for j in 0..10 {
                tiles[i].push(Self::gen_tile());
            }
        }
        Self{
            tiles
        }
    }
    pub fn render(&self, player: &Player) {
        for i in 0..self.tiles.len() {
            for j in 0..self.tiles[i].len() {
                if i as i32 == player.x && j as i32 == player.y {
                    print!("P");
                } else {
                    print!("{}", self.tiles[i][j]);
                }
            }
            println!();
        }
    }

    pub fn gen_tile() -> Tile {
        let num: f32 = random();
        match num {
            _ if num < 0.95 => {
                let mut t = Tile::make_dirt();
                randomly_add(&mut t, &Tile::add_tree, 0.2);
                t
            },
            _ => Tile::make_stone()
        }
    }
}
