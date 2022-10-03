use crate::{Material, Player};
use crate::map_generation::tile::{randomly_augment, Tile};
use rand::prelude::*;


/// The playing grid
pub struct Field {
    pub tiles: Vec<Vec<Tile>>,

}

impl Field {
    pub fn new() -> Self {
        let init_size = 500;
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

    /// Display as glyphs
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
    /// returns: (2d Vector: the list of positions)
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

    /// Makes a list of positions of blocks of given height around the player.
    pub fn depth_indices(&self, player: &Player, height: usize, radius: usize) -> Vec<(i32, i32)> {
        let mut res: Vec<(i32, i32)> = Vec::new();
        for i in (player.x as usize - radius)..=(player.x as usize + radius) {
            for j in (player.y as usize - radius)..=(player.y as usize + radius) {
                if self.tiles[i][j].len() == height {
                    res.push((i as i32 - player.x, j as i32 - player.y));
                }
            }
        }
        res
    }

    /// Randomly generate a Tile (a cell on the field)
    pub fn gen_tile() -> Tile {
        let num: f32 = random();
        match num {
            _ if num < 0.95 => {
                let mut t = Tile::make_dirt();
                randomly_augment(&mut t, &Tile::add_tree, 0.2);
                t
            },
            _ => Tile::make_stone()
        }
    }
    pub fn extend(&mut self) {

    }
}
