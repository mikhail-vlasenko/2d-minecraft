use rand::random;
use crate::map_generation::tile::{randomly_augment, Tile};

pub struct Chunk{
    pub tiles: Vec<Vec<Tile>>,
}

impl Chunk {
    pub fn new(size: usize) -> Self {
        let mut tiles = Vec::new();
        for i in 0..size {
            tiles.push(Vec::new());
            for _ in 0..size {
                tiles[i].push(Self::gen_tile());
            }
        }
        Self{
            tiles
        }
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
}