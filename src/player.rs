use std::cell::{Ref, RefMut};
use std::f32::consts::PI;
use crate::map_generation::block::Block;
use crate::{Field, Material};
use crate::inventory::Inventory;
use crate::material::Material::*;
use crate::storable::Storable;


/// The player.
pub struct Player {
    pub x: i32,
    pub y: i32,
    pub z: usize,
    rotation: f32,
    inventory: Inventory,
    // Storables that will be used for the corresponding actions
    // Are set though UI
    pub placement_material: Material,
    pub crafting_item: Storable,
}

impl Player {
    pub fn new(field: &Field) -> Self {
        let mut player = Self {
            x: 0,
            y: 0,
            z: 0,
            rotation: 0.,
            inventory: Inventory::new(),
            placement_material: Plank,
            crafting_item: Storable::M(Plank)
        };
        player.land(field);
        player
    }
    pub fn mine(&mut self, field: &mut Field, delta_x: i32, delta_y: i32) {
        let xx = self.x + delta_x;
        let yy = self.y + delta_y;

        if field.get_chunk_immut(xx, yy).top_at(xx, yy).material == Bedrock {
            println!("cannot mine bedrock");
        } else {
            let mat =  field.get_chunk_immut(xx, yy).top_at(xx, yy).material;
            println!("{} mined", mat.to_string());
            self.inventory.pickup(Storable::M(mat), 1);
            field.pop_at(self.x + delta_x, self.y + delta_y);
        }
    }

    /// Mines a block in front of the player.
    pub fn mine_infront(&mut self, field: &mut Field){
        let (x, y) = self.coords_from_rotation();
        self.mine(field, x, y);
    }

    pub fn place(&mut self, field: &mut Field, delta_x: i32, delta_y: i32, material: Material) {
        if field.full_at(self.x + delta_x, self.y + delta_y) {
            println!("too high to build");
            return
        }

        let placement_block = Block { material };

        if self.inventory.drop(&Storable::M(material), 1) {
            field.push_at(placement_block, self.x + delta_x, self.y + delta_y);
            println!("{} placed", material.to_string());
        } else {
            println!("You do not have a block of {}", material.to_string());
        }
    }

    /// Places a currently selected block in front of the player.
    pub fn place_current(&mut self, field: &mut Field) {
        let (x, y) = self.coords_from_rotation();
        self.place(field, x, y, self.placement_material)
    }

    pub fn walk(&mut self, direction: &str, field: &mut Field) {
        match direction {
            "w" => self.step(field, -1, 0),
            "a" => self.step(field, 0, -1),
            "s" => self.step(field, 1, 0),
            "d" => self.step(field, 0, 1),
            _ => println!("unknown direction")
        }
    }

    /// Moves the Player by (delta_x, delta_y), checking for height conditions.
    ///
    /// # Arguments
    ///
    /// * `field`: the playing field
    /// * `delta_x`:
    /// * `delta_y`:
    fn step(&mut self, field: &mut Field, delta_x: i32, delta_y: i32){
        if field.len_at(self.x + delta_x, self.y + delta_y) <= self.z + 1 {
            self.x += delta_x;
            self.y += delta_y;
            self.land(field);
            let curr_chunk = (field.chunk_pos(self.x), field.chunk_pos(self.y));
            if curr_chunk != field.get_central_chunk() {
                field.load(curr_chunk.0, curr_chunk.1);
                println!("new chunks loaded");
            }
        } else {
            println!("too high to step on")
        }
    }

    /// Sets the z coordinate of the Player
    pub fn land(&mut self, field: &Field) {
        self.z = field.len_at(self.x, self.y);
    }

    fn exists_around(&self, field: &Field, material: &Material) -> bool {
        let x_indices = vec![-1, 0, 1];
        let y_indices = vec![-1, 0, 1];
        for x in x_indices {
            for y in &y_indices {
                if &field.top_material_at(self.x + x, self.y + y) == material {
                    return true;
                }
            }
        }
        return false;
    }

    pub fn can_craft(&self, item: &Storable, field: &Field) -> bool {
        let mut possible = true;
        for (req, amount) in item.craft_requirements() {
            if self.inventory.count(&req) < *amount {
                possible = false;
                break
            }
        }
        let crafter = item.required_crafter();
        let mut crafter_near = false;
        if crafter == None {
            crafter_near = true;
        } else {
            crafter_near = self.exists_around(field, crafter.unwrap());
        }
        possible && item.is_craftable() && crafter_near
    }

    /// If crafting the given item is possible, subtracts the ingredients and adds the item to the inventory.
    /// Else does nothing.
    pub fn craft(&mut self, item: Storable, field: &Field) {
        if !self.can_craft(&item, field){
            println!("cant craft!");
            return
        }
        for (req, amount) in item.craft_requirements() {
            self.inventory.drop(req, *amount);
        }
        let craft_yield = item.craft_yield();
        self.inventory.pickup(item, craft_yield)
    }

    pub fn craft_current(&mut self, field: &Field) {
        self.craft(self.crafting_item, field)
    }

    /// Rotates the Player 90 degrees (counter-)clockwise.
    ///
    /// # Arguments
    ///
    /// * `side`: -1 or 1; 1 is counterclockwise.
    pub fn turn(&mut self, side: i32) {
        self.rotation += side as f32;
    }

    /// Computes delta_x and delta_y that, added to the Player position, give a cell in front of him.
    pub fn coords_from_rotation(&self) -> (i32, i32) {
        (-(self.rotation * PI / 2.).cos() as i32, -(self.rotation * PI / 2.).sin() as i32)
    }

    /// Computes rotation as an integer between 0 and 4, where 0 is up, and 3 is right.
    pub fn get_rotation(&self) -> u32 {
        let modulus = self.rotation as i32 % 4;
        if modulus >= 0 {
            modulus as u32
        } else {
            (modulus + 4) as u32
        }
    }

    pub fn get_inventory(&self) -> &Vec<(Storable, u32)> {
        self.inventory.get_all()
    }

    pub fn render_inventory(&self) {
        self.inventory.render();
    }
}
