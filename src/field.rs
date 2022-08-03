use crate::block::Block;
use crate::material::materials;
use crate::Player;
use crate::tile::Tile;

pub struct Field<'a> {
    pub tiles: Vec<Vec<Tile<'a>>>,
}


impl Field<'_> {
    pub fn new() -> Self {
        let tile = Tile {
            blocks: vec![Block { material: &materials::BEDROCK },
                         Block { material: &materials::STONE },
                         Block { material: &materials::DIRT },
                         Block { material: &materials::AIR },
                         Block { material: &materials::AIR }],
            top: 2,
        };
        let tree = Tile {
            blocks: vec![Block { material: &materials::BEDROCK },
                         Block { material: &materials::STONE },
                         Block { material: &materials::DIRT },
                         Block { material: &materials::LOG },
                         Block { material: &materials::LOG }],
            top: 4,
        };
        let mut field = Field{
            tiles: vec![vec![tile; 10]; 10],
        };
        field.tiles[5][6] = tree;
        field
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
}
