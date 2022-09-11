use crate::{Material, Player};
use crate::tile::{randomly_add, Tile};
use rand::prelude::*;

pub struct Field {
    pub tiles: Vec<Vec<Tile>>,
}


impl Field {
    pub fn new() -> Self {
        let init_size = 50;
        let mut tiles = Vec::new();
        for i in 0..init_size {
            tiles.push(Vec::new());
            for j in 0..init_size {
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


    /// Makes a list of positions of blocks of given material around the player.
    /// Useful for rendering if blocks of same type are rendered simultaneously.
    /// Positions are centered on the player.
    /// [positive, positive] is bottom right.
    /// first coord is vertical.
    ///
    /// # Arguments
    ///
    /// * `player`: the player
    /// * `material`: index only blocks of this material
    /// * `radius`: how far field from player is included
    ///
    /// returns: ()
    pub fn texture_indices(&self, player: &Player, material: Material, radius: usize) -> Vec<(i32, i32)> {
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x as usize - radius)..=(player.x as usize + radius) {
            for j in (player.y as usize - radius)..=(player.y as usize + radius) {
                if self.tiles[i][j].top().material == material {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
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
    pub fn extend(&mut self) {

    }
}
