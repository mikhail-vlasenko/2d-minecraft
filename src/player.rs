use crate::block::Block;
use crate::{Field, Material};
use crate::inventory::Inventory;
use crate::material::materials;
use crate::items::{Item, item_by_name, possible_items, Storable};


pub struct Player<'a> {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    inventory: Inventory<'a>,
}

impl<'a> Player<'a> {
    pub fn new() -> Self {
        Self {
            x: 4,
            y: 4,
            z: 3,
            inventory: Inventory::new()
        }
    }
    pub fn mine(&mut self, field: &mut Field, x: i32, y: i32) {
        let tile_x = (self.x + x) as usize;
        let tile_y = (self.y + y) as usize;
        let tile = &mut field.tiles[tile_x][tile_y];
        if tile.blocks[tile.top].material == &materials::BEDROCK {
            println!("cannot mine bedrock");
        } else {
            let mat = tile.blocks[tile.top].material;
            println!("{} mined", mat.name);
            let mined_item = match item_by_name(mat.name) {
                Some(i) => i,
                None => Item::from_material(mat)
            };
            self.inventory.pickup(mined_item, 1);
            tile.blocks[tile.top] = Block { material: &materials::AIR };
            tile.top -= 1;
        }
    }

    pub fn place(&mut self, field: &mut Field, x: i32, y: i32, material: &'static Material) {
        let tile_x = (self.x + x) as usize;
        let tile_y = (self.y + y) as usize;
        let tile = &mut field.tiles[tile_x][tile_y];
        let placement_block = Block { material: &material };
        let mined_item = match item_by_name(material.name) {
            Some(i) => i,
            None => Item::from_material(material)
        };
        if self.inventory.drop(&mined_item, 1) {
            tile.top += 1;
            tile.blocks[tile.top] = placement_block;
            println!("{} placed", material.name);
        } else {
            println!("You do not have a block of {}", material.name);
        }
    }

    pub fn walk(&mut self, direction: &str) {
        match direction {
            "w" => self.x -= 1,
            "a" => self.y -= 1,
            "s" => self.x += 1,
            "d" => self.y += 1,
            _ => println!("unknown direction")
        }
    }

    pub fn can_craft(&self, item: &Item) -> bool {
        let mut possible = true;
        for (req, amount) in item.craft_requirements() {
            if self.inventory.count(&req) < *amount {
                possible = false;
                break
            }
        }
        possible && item.is_craftable()
    }

    pub fn craft(&mut self, item: Item<'a>) {
        assert!(self.can_craft(&item));
        for (req, amount) in item.craft_requirements() {
            self.inventory.drop(req, *amount);
        }
        let craft_yield = item.craft_yield();
        self.inventory.pickup(item, craft_yield)
    }

    pub fn render_inventory(&self) {
        self.inventory.render();
    }
}
