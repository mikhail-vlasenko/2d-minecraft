use std::cell::{Ref, RefMut};
use std::cmp::min;
use std::f32::consts::PI;
use crate::crafting::consumable::Consumable;
use crate::map_generation::block::Block;
use crate::map_generation::field::Field;
use crate::crafting::inventory::Inventory;
use crate::crafting::items::Item::Arrow;
use crate::crafting::material::Material;
use crate::crafting::material::Material::*;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::Storable;
use crate::crafting::storable::Storable::{I, RW};

pub const MAX_HP: i32 = 100;

/// The player.
pub struct Player {
    pub x: i32,
    pub y: i32,
    pub z: usize,
    rotation: f32,
    hp: i32,
    inventory: Inventory,

    // Storables that will be used for the corresponding actions
    // Are set though UI
    pub placement_material: Material,
    pub crafting_item: Storable,
    pub consumable: Consumable,
    pub ranged_weapon: RangedWeapon,

    pub message: String,
}

impl Player {
    pub fn new(field: &Field) -> Self {
        let mut player = Self {
            x: 0,
            y: 0,
            z: 0,
            rotation: 0.,
            hp: MAX_HP,
            inventory: Inventory::new(),
            placement_material: Plank,
            crafting_item: Storable::M(Plank),
            consumable: Consumable::Apple,
            ranged_weapon: RangedWeapon::Bow,
            message: String::new(),
        };
        player.land(field);
        player
    }
    /// Returns how much action was spent.
    pub fn mine(&mut self, field: &mut Field, delta_x: i32, delta_y: i32) -> f32 {
        let xx = self.x + delta_x;
        let yy = self.y + delta_y;

        let mat =  field.top_material_at((xx, yy));
        if mat.required_mining_power() > self.get_mining_power() {
            self.add_message(&format!("Need {} mining PWR", mat.required_mining_power()));
            0.
        } else {
            self.add_message(&format!("Mined {}", mat));
            self.inventory.pickup(Storable::M(mat), 1);
            field.pop_at((self.x + delta_x, self.y + delta_y));
            1.
        }
    }

    /// Mines a block in front of the player.
    pub fn mine_infront(&mut self, field: &mut Field) -> f32 {
        let (x, y) = self.coords_from_rotation();
        self.mine(field, x, y)
    }

    pub fn get_mining_power(&self) -> i32 {
        self.inventory.mining_power()
    }

    pub fn place(&mut self, field: &mut Field, delta_x: i32, delta_y: i32, material: Material) -> f32 {
        if field.full_at((self.x + delta_x, self.y + delta_y)) {
            self.add_message(&format!("Too high to build"));
            return 0.;
        }

        let placement_block = Block { material };

        if self.inventory.drop(&Storable::M(material), 1) {
            field.push_at(placement_block, (self.x + delta_x, self.y + delta_y));
            1.
        } else {
            self.add_message(&format!("You do not have a block of {}", material));
            0.
        }
    }

    /// Places a currently selected block in front of the player.
    /// Returns how much action was spent.
    pub fn place_current(&mut self, field: &mut Field) -> f32 {
        let (x, y) = self.coords_from_rotation();
        self.place(field, x, y, self.placement_material)
    }

    pub fn walk(&mut self, direction: &str, field: &mut Field) -> f32 {
        match direction {
            "w" => self.step(field, -1, 0),
            "a" => self.step(field, 0, -1),
            "s" => self.step(field, 1, 0),
            "d" => self.step(field, 0, 1),
            _ => { println!("unknown direction"); 0. },
        }
    }

    /// Moves the Player by (delta_x, delta_y), checking for height conditions.
    ///
    /// # Arguments
    ///
    /// * `field`: the playing field
    /// * `delta_x`:
    /// * `delta_y`:
    ///
    /// returns: how much action was spent.
    fn step(&mut self, field: &mut Field, delta_x: i32, delta_y: i32) -> f32 {
        let new_pos = (self.x + delta_x, self.y + delta_y);
        if field.len_at(new_pos) <= self.z + 1 {
            // fighting
            if field.is_occupied(new_pos) {
                field.damage_mob(new_pos, self.get_melee_damage());
                return 1.
            }
            // movement
            self.x += delta_x;
            self.y += delta_y;
            self.land(field);

            // loot
            let loot = field.gather_loot_at((self.x, self.y));
            if loot.len() > 0 {
                for l in loot {
                    self.add_message(&format!("Looted {}", l));
                    self.pickup(l, 1);
                }
            }

            // chunk loading
            let curr_chunk = (field.chunk_pos(self.x), field.chunk_pos(self.y));
            if curr_chunk != field.get_central_chunk() {
                field.load(curr_chunk.0, curr_chunk.1);
            }
            1.
        } else {
            self.add_message(&"Too high to step on");
            0.
        }
    }

    /// Sets the z coordinate of the Player
    pub fn land(&mut self, field: &Field) {
        self.z = field.len_at((self.x, self.y));
    }

    fn exists_around(&self, field: &Field, material: &Material) -> bool {
        let x_indices = vec![-1, 0, 1];
        let y_indices = vec![-1, 0, 1];
        for x in x_indices {
            for y in &y_indices {
                if &field.top_material_at((self.x + x, self.y + y)) == material {
                    return true;
                }
            }
        }
        return false;
    }

    /// Returns true if the player can craft the item right now.
    pub fn can_craft(&mut self, item: &Storable, field: &Field) -> bool {
        for (req, amount) in item.craft_requirements() {
            if self.inventory.count(&req) < *amount {
                self.add_message(&format!("Need {} of {}, have {}", amount, req, self.inventory.count(&req)));
                return false;
            }
        }

        let crafter = item.required_crafter();
        let crafter_near =
            if crafter == None { true } else {
                self.exists_around(field, crafter.unwrap())
            };
        if !crafter_near {
            self.add_message(&format!("There is no {} nearby", crafter.unwrap()));
        }

        item.is_craftable() && crafter_near
    }

    /// Returns true if the player has all the ingredients to craft the item,
    /// but not necessarily can craft it now.
    pub fn has_all_ingredients(&self, item: &Storable) -> bool {
        if !item.is_craftable() {
            return false;
        }
        for (req, amount) in item.craft_requirements() {
            if self.inventory.count(&req) < *amount {
                return false;
            }
        }
        true
    }

    /// If crafting the given item is possible, subtracts the ingredients and adds the item to the inventory.
    /// Else does nothing.
    pub fn craft(&mut self, item: Storable, field: &Field) -> f32 {
        if !self.can_craft(&item, field){
            return 0.
        }
        for (req, amount) in item.craft_requirements() {
            self.inventory.drop(req, *amount);
        }
        let craft_yield = item.craft_yield();
        self.inventory.pickup(item, craft_yield);
        self.add_message(&format!("Crafted {} of {}", craft_yield, item));
        1.
    }

    pub fn craft_current(&mut self, field: &Field) -> f32 {
        self.craft(self.crafting_item, field)
    }

    pub fn pickup(&mut self, storable: Storable, amount: u32) {
        self.inventory.pickup(storable, amount)
    }

    pub fn consume(&mut self, consumable: Consumable) -> f32 {
        if self.inventory.drop(&Storable::C(consumable), 1) {
            consumable.apply_effect(self);
            1.
        } else {
            self.add_message(&format!("You dont have {}", consumable));
            0.
        }
    }

    pub fn consume_current(&mut self) -> f32 {
        self.consume(self.consumable)
    }

    pub fn shoot(&mut self, field: &mut Field, weapon: RangedWeapon) -> f32 {
        if !self.has(&RW(weapon)) {
            self.add_message(&format!("You do not have {}", weapon));
            return 0.0;
        }
        if !self.inventory.drop(&I(*weapon.ammo()), 1) {
            self.add_message(&format!("No ammo! ({})", weapon.ammo()));
            return 0.0;
        }
        let direction = self.coords_from_rotation();
        let mut curr_tile = (self.x, self. y);
        let height = self.z + 1;
        for _ in 0..weapon.range() {
            curr_tile = (curr_tile.0 + direction.0, curr_tile.1 + direction.1);
            if field.len_at(curr_tile) > height {
                break;
            }
            if field.is_occupied(curr_tile) {
                field.damage_mob(curr_tile, weapon.damage());
                break;
            }
        }
        if weapon.ammo() == &Arrow {
            // arrows can be reused, but sometimes break
            let rng: f32 = rand::random();
            if rng > 0.3 {
                field.add_loot_at(vec![I(Arrow)], curr_tile);
            } else {
                self.add_message(&"Arrow broke");
            }
        }
        1.0
    }

    pub fn shoot_current(&mut self, field: &mut Field) -> f32 {
        self.shoot(field, self.ranged_weapon)
    }

    /// Rotates the Player 90 degrees (counter-)clockwise.
    ///
    /// # Arguments
    ///
    /// * `side`: -1 or 1; 1 is counterclockwise.
    /// returns: how much action was spent.
    pub fn turn(&mut self, side: i32) -> f32 {
        self.rotation += side as f32;
        0.25
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

    pub fn has(&self, storable: &Storable) -> bool {
        self.inventory.contains(storable)
    }

    pub fn inventory_count(&self, storable: &Storable) -> u32 {
        self.inventory.count(storable)
    }

    pub fn get_inventory(&self) -> &Vec<(Storable, u32)> {
        self.inventory.get_all()
    }

    pub fn render_inventory(&self) {
        self.inventory.render();
    }

    pub fn add_message(&mut self, new: &str) {
        if !self.message.is_empty() {
            self.message.push_str("\n");
        }
        self.message.push_str(new);
    }
     pub fn reset_message(&mut self) {
         self.message = String::new();
     }
}

/// HP and damage things
impl Player {
    pub fn receive_damage(&mut self, damage: i32) {
        self.add_message(&format!("You are hit by {}", damage));
        self.hp -= damage
    }

    pub fn heal(&mut self, hp: i32) {
        self.hp = min(MAX_HP, self.hp + hp)
    }

    pub fn get_melee_damage(&self) -> i32 {
        10 + self.inventory.damage_modifier()
    }

    pub fn get_hp(&self) -> i32 {
        self.hp
    }
}
