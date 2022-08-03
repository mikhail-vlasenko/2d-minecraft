use crate::block::Block;
use crate::{Field, Material};
use crate::hash_map_storable::into_key;
use crate::inventory::Inventory;
use crate::material::materials;


pub struct Player {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    inventory: Inventory,
}

impl Player {
    pub fn new() -> Self {
        Self {
            x: 4,
            y: 4,
            z: 3,
            inventory: Inventory::new()
        }
    }
    pub fn mine(&mut self, field: &mut Field<'static>, x: i32, y: i32) {
        let tile_x = (self.x + x) as usize;
        let tile_y = (self.y + y) as usize;
        let tile = &mut field.tiles[tile_x][tile_y];
        if tile.blocks[tile.top].material == &materials::BEDROCK {
            println!("cannot mine bedrock");
        } else {
            println!("{} mined", tile.blocks[tile.top].material.name);
            self.inventory.pickup(into_key(tile.blocks[tile.top].clone()));
            tile.blocks[tile.top] = Block { material: &materials::AIR };
            tile.top -= 1;
        }
    }

    pub fn place(&mut self, field: &mut Field<'static>, x: i32, y: i32, material: &'static Material) {
        let tile_x = (self.x + x) as usize;
        let tile_y = (self.y + y) as usize;
        let tile = &mut field.tiles[tile_x][tile_y];
        let placement_block = Block { material: &material };
        if self.inventory.drop(into_key(placement_block.clone())) {
            tile.top += 1;
            tile.blocks[tile.top] = placement_block;
            println!("{} placed", material.name);
        } else {
            println!("You do not have a block of {}", material.name);
        }
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

    pub fn render_inventory(&self) {
        self.inventory.render();
    }
}
