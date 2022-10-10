use crate::map_generation::tile::Tile;

pub struct Chunk{
    pub tiles: Vec<Vec<Tile>>,
}

impl Chunk {
    pub fn new(size: u32) -> Self {
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
}