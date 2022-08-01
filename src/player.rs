use crate::block::Block;
use crate::{Field, Material};
use crate::material::materials;


pub struct Player {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Player {
    pub fn mine(&self, field: &mut Field, x: i32, y: i32) {
        let tile_x = (self.x + x) as usize;
        let tile_y = (self.y + y) as usize;
        let tile = &mut field.tiles[tile_x][tile_y];
        if tile.blocks[tile.top].material == &materials::BEDROCK {
            println!("cannot mine bedrock");
        } else {
            println!("{} mined", tile.blocks[tile.top].material.name);
            tile.blocks[tile.top] = Block { material: &materials::AIR };
            tile.top -= 1;
        }
    }

    pub fn place<'a>(&self, field: &mut Field<'a>, x: i32, y: i32, material: &'a Material) {
        let tile_x = (self.x + x) as usize;
        let tile_y = (self.y + y) as usize;
        let tile = &mut field.tiles[tile_x][tile_y];
        tile.top += 1;
        tile.blocks[tile.top] = Block { material: &material };
        println!("{} placed", material.name);
    }

    pub fn walk(&mut self, direction: &str) {
        match direction {
            "n" => self.x -= 1,
            "e" => self.y += 1,
            "s" => self.x += 1,
            "w" => self.y -= 1,
            _ => println!("unknown direction")
        }
    }
}
